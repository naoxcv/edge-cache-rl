from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.container import create_catalog
from env.edge_network import EdgeNetwork
from env.request_generator import RequestGenerator
from env.rewards import cache_action_overlap_penalty, score_cache_request


def agent_id(node_id: int) -> str:
    """Convert a numeric node ID to the string agent key."""
    return str(node_id)


def local_obs_size(num_container_types: int, cache_capacity: int) -> int:
    """Per-node local observation dimension.

    Layout: cache slots (C×K) + utilization (1) + request freq (K)
    + pending request one-hot (K) + needs_decision (1).
    """
    return cache_capacity * num_container_types + 2 * num_container_types + 2


def neighbor_feature_size(
    comm_level: int, num_container_types: int, cache_capacity: int
) -> int:
    """Per-neighbor feature width for a communication level."""
    if comm_level <= 0:
        return 0
    if comm_level in (1, 3):
        return num_container_types
    if comm_level == 2:
        return local_obs_size(num_container_types, cache_capacity)
    raise ValueError(f"Unsupported comm_level={comm_level} (expected 0–3)")


def observation_size(
    num_container_types: int,
    *,
    comm_level: int,
    max_neighbors: int,
    cache_capacity: int,
) -> int:
    """Return total observation width given communication level and neighbor count."""
    local = local_obs_size(num_container_types, cache_capacity)
    if comm_level <= 0:
        return local
    return local + max_neighbors * neighbor_feature_size(
        comm_level, num_container_types, cache_capacity
    )


def needs_decision_from_obs(obs: np.ndarray) -> bool:
    """True when the last local feature flags a full-cache miss."""
    return bool(np.asarray(obs).reshape(-1)[-1] > 0.5)


class MultiAgentCachingEnv(gym.Env):
    """Multi-node caching env with per-agent dict observations/rewards.

    Step order (all nodes each timestep):
      1. Score each node's *pending* request against the current caches
         (local hit / same-cluster forward / cloud)
      2. Admit the missed item if needed (see action space)
      3. Record request histories and draw the next pending requests

    Action space is eviction-only: ``Discrete(C+1)`` with C = cache_capacity.
      ``0 … C-1`` — on a full-cache miss, evict that LRU-ordered slot and
      insert the pending request
      ``C`` — reject (do not cache the pending request)
    Hits and non-full misses need no decision: the env ignores the action
    (non-full misses are inserted automatically).

    Communication levels (``comm_level``):
      0 — local state only
      1 — append each *same-cluster* peer's cache binary (padded to max cluster
          degree × K). Inter-cluster bridge neighbors are excluded so the obs
          matches ``forwarding_same_cluster_only``.
      2 — same peer set as Level 1, with full neighbor state per peer
      3 — same layout as Level 1, but neighbor features are included only when the
          agent elects to communicate (Q-margin selective; see agents/multi_agent.py)

    Overlap penalty (``overlap_penalty_weight``): applied once when a container
    is *inserted* and a neighbor already holds it — a nudge, not a per-timestep
    tax on the whole cache.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict | None = None, seed: int = 42) -> None:
        if config is None:
            from configs import load_config

            config = load_config()

        self.config = config
        self.num_nodes = int(config["num_nodes"])
        self.num_container_types = int(config["num_container_types"])
        self.cache_capacity = int(config["cache_capacity"])
        self.observation_window = int(config["observation_window"])
        self.episode_length = int(config["episode_length"])
        self.enable_forwarding = bool(config.get("enable_forwarding", True))
        self.same_cluster_only = bool(config.get("forwarding_same_cluster_only", True))
        self.comm_level = int(config.get("comm_level", 0))
        if self.comm_level not in (0, 1, 2, 3):
            raise ValueError(f"comm_level must be 0–3; got {self.comm_level}")
        self.overlap_penalty_weight = float(config.get("overlap_penalty_weight", 0.0))
        self.comm_penalty_lambda = float(config.get("comm_penalty_lambda", 0.0))
        self.selective_comm_threshold = float(
            config.get("selective_comm_threshold", 0.1)
        )
        self.reject_action = self.cache_capacity
        self.timestep = 0
        self._seed = seed
        self._episode_overlap = 0
        self._episode_overlap_penalty = 0.0
        self._episode_task_reward = 0.0
        self._episode_comm_events = 0
        self._episode_comm_penalty = 0.0
        self._comm_active: dict[int, bool] = {}
        self._pending_requests: dict[int, int | None] = {}

        self.catalog = create_catalog(self.num_container_types, seed=seed)
        self.network = EdgeNetwork(config)
        self.request_generator = RequestGenerator(
            config,
            self.catalog,
            seed=seed,
            cluster_for_node=self.network.cluster_for_node,
        )

        # Neighbor slots for obs / overlap: same-cluster peers only, matching
        # ``forwarding_same_cluster_only``. Graph bridge edges stay in the
        # adjacency for latency, but do not feed the caching policy.
        self.neighbor_lists = {
            i: list(self.network.get_cluster_neighbors(i))
            for i in range(self.num_nodes)
        }
        self.max_neighbors = max((len(v) for v in self.neighbor_lists.values()), default=0)
        self.local_obs_dim = local_obs_size(
            self.num_container_types, self.cache_capacity
        )
        self.neighbor_feat_dim = neighbor_feature_size(
            self.comm_level, self.num_container_types, self.cache_capacity
        )

        self.possible_agents = [agent_id(i) for i in range(self.num_nodes)]
        self.agents = list(self.possible_agents)

        obs_size = observation_size(
            self.num_container_types,
            comm_level=self.comm_level,
            max_neighbors=self.max_neighbors,
            cache_capacity=self.cache_capacity,
        )
        single_obs = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)
        single_act = spaces.Discrete(self.cache_capacity + 1)
        self.observation_spaces = {aid: single_obs for aid in self.possible_agents}
        self.action_spaces = {aid: single_act for aid in self.possible_agents}
        self.observation_space = single_obs
        self.action_space = single_act

    def _obs_size(self) -> int:
        return int(np.prod(self.observation_space.shape))

    def get_observation_space(self, agent_id_: str) -> gym.Space:
        """Return the observation space for a specific agent."""
        return self.observation_spaces[agent_id_]

    def get_action_space(self, agent_id_: str) -> gym.Space:
        """Return the action space for a specific agent."""
        return self.action_spaces[agent_id_]

    def _needs_decision(self, node_id: int) -> bool:
        requested = self._pending_requests.get(node_id)
        if requested is None:
            return False
        node = self.network.nodes[node_id]
        if node.is_cached(int(requested)):
            return False
        return len(node.cache) >= node.cache_capacity

    def _local_observation(self, node_id: int) -> np.ndarray:
        node = self.network.nodes[node_id]
        k = self.num_container_types
        c = self.cache_capacity
        slots = node.get_cache_slots(c, k)
        utilization = np.array(
            [len(node.cache) / max(c, 1)], dtype=np.float32
        )
        freq = node.get_request_freq(k, self.observation_window)
        request = np.zeros(k, dtype=np.float32)
        pending = self._pending_requests.get(node_id)
        if pending is not None and 0 <= int(pending) < k:
            request[int(pending)] = 1.0
        need = np.array(
            [1.0 if self._needs_decision(node_id) else 0.0], dtype=np.float32
        )
        return np.concatenate([slots, utilization, freq, request, need])

    def _neighbor_features(self, node_id: int) -> np.ndarray:
        if self.comm_level <= 0 or self.max_neighbors == 0:
            return np.zeros(0, dtype=np.float32)
        feat_level = 1 if self.comm_level in (1, 3) else 2
        feat_dim = neighbor_feature_size(
            feat_level, self.num_container_types, self.cache_capacity
        )

        parts: list[np.ndarray] = []
        neighbors = self.neighbor_lists[node_id]
        for slot in range(self.max_neighbors):
            if slot < len(neighbors):
                nbr = neighbors[slot]
                node = self.network.nodes[nbr]
                if feat_level == 1:
                    parts.append(node.get_cache_binary(self.num_container_types))
                else:
                    parts.append(self._local_observation(nbr))
            else:
                parts.append(np.zeros(feat_dim, dtype=np.float32))
        return np.concatenate(parts)

    def _get_observation(self, node_id: int) -> np.ndarray:
        local = self._local_observation(node_id)
        if self.comm_level <= 0:
            return local
        return np.concatenate([local, self._neighbor_features(node_id)])

    def _get_observations(self) -> dict[str, np.ndarray]:
        return {agent_id(i): self._get_observation(i) for i in range(self.num_nodes)}

    def _overlap_for_insert(self, node_id: int, container_id: int) -> tuple[float, int]:
        if self.network.nodes[node_id].is_cached(container_id):
            return 0.0, 0
        return cache_action_overlap_penalty(
            self.network,
            node_id,
            container_id,
            self.neighbor_lists[node_id],
            weight=self.overlap_penalty_weight,
        )

    def _admit(self, node_id: int, action: int) -> tuple[float, int]:
        """Insert the pending miss according to eviction/reject; return overlap.

        Call after scoring so the current request is never counted as a hit
        from this admission. Hits and empty pending are no-ops.
        """
        requested = self._pending_requests.get(node_id)
        node = self.network.nodes[node_id]
        if requested is None or node.is_cached(int(requested)):
            return 0.0, 0
        requested = int(requested)

        if len(node.cache) < node.cache_capacity:
            pen, overlap_n = self._overlap_for_insert(node_id, requested)
            node.cache_container(requested)
            return pen, overlap_n

        action = int(action)
        if action < 0 or action >= self.cache_capacity:
            return 0.0, 0
        if action >= len(node.cache):
            return 0.0, 0
        node.evict_slot(action)
        pen, overlap_n = self._overlap_for_insert(node_id, requested)
        node.cache_container(requested)
        return pen, overlap_n

    def record_communication(self, node_id: int, communicated: bool) -> float:
        """Record a Level-3 communication event; return penalty (<= 0).

        Events are counted even when ``comm_penalty_lambda=0`` so selectivity
        can be measured without charging a cost.
        """
        if not communicated:
            return 0.0
        self._episode_comm_events += 1
        if self.comm_penalty_lambda <= 0.0:
            return 0.0
        pen = -self.comm_penalty_lambda
        self._episode_comm_penalty += pen
        return pen

    def _process_request(self, node_id: int, container_id: int | None) -> float:
        return score_cache_request(
            self.network,
            node_id,
            container_id,
            self.config,
            enable_forwarding=self.enable_forwarding,
            same_cluster_only=self.same_cluster_only,
        )

    def _node_rates(self, node_id: int) -> dict[str, float]:
        node = self.network.nodes[node_id]
        total = node.hits + node.misses + node.forwards
        if total == 0:
            return {"hit_rate": 0.0, "forward_rate": 0.0, "cloud_rate": 0.0}
        return {
            "hit_rate": node.hits / total,
            "forward_rate": node.forwards / total,
            "cloud_rate": node.misses / total,
        }

    def _draw_pending_requests(self) -> None:
        generated = self.request_generator.generate()
        self._pending_requests = {i: generated[i] for i in range(self.num_nodes)}

    def network_stats(self) -> dict[str, Any]:
        """Aggregate hit/forward/miss rates and cache diversity across all nodes."""
        hits = sum(n.hits for n in self.network.nodes)
        forwards = sum(n.forwards for n in self.network.nodes)
        misses = sum(n.misses for n in self.network.nodes)
        total = hits + forwards + misses
        caches = {i: list(self.network.nodes[i].cache) for i in range(self.num_nodes)}
        unique = {tuple(sorted(c)) for c in caches.values()}
        return {
            "hits": hits,
            "forwards": forwards,
            "misses": misses,
            "hit_rate": hits / total if total else 0.0,
            "forward_rate": forwards / total if total else 0.0,
            "cloud_rate": misses / total if total else 0.0,
            "caches": caches,
            "cache_diversity": len(unique),
            "comm_level": self.comm_level,
            "max_neighbors": self.max_neighbors,
            "obs_dim": self._obs_size(),
            "n_actions": int(self.action_space.n),
            "overlap_penalty_weight": self.overlap_penalty_weight,
            "overlap_count": self._episode_overlap,
            "overlap_penalty_total": self._episode_overlap_penalty,
            "task_return": self._episode_task_reward,
            "comm_events": self._episode_comm_events,
            "comm_penalty_total": self._episode_comm_penalty,
        }

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Reset all nodes, optionally re-seeding catalog and traffic."""
        if seed is not None:
            self._seed = seed
            self.catalog = create_catalog(self.num_container_types, seed=seed)
            self.request_generator = RequestGenerator(
                self.config,
                self.catalog,
                seed=seed,
                cluster_for_node=self.network.cluster_for_node,
            )

        self.timestep = 0
        self.agents = list(self.possible_agents)
        self.network.reset()
        self.request_generator.reset()
        self._episode_overlap = 0
        self._episode_overlap_penalty = 0.0
        self._episode_task_reward = 0.0
        self._episode_comm_events = 0
        self._episode_comm_penalty = 0.0
        self._comm_active = {}
        self._draw_pending_requests()

        obs = self._get_observations()
        infos = {
            agent_id(i): {
                "timestep": 0,
                "requested": self._pending_requests.get(i),
                "needs_decision": self._needs_decision(i),
                "overlap_penalty": 0.0,
                "task_reward": 0.0,
                **self._node_rates(i),
            }
            for i in range(self.num_nodes)
        }
        return obs, infos

    def step(
        self, action_dict: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Score pending requests, admit misses, then draw the next requests."""
        scored: dict[int, int | None] = dict(self._pending_requests)
        needed = {i: self._needs_decision(i) for i in range(self.num_nodes)}

        task_rewards: dict[int, float] = {}
        for node_id in range(self.num_nodes):
            task_rewards[node_id] = self._process_request(
                node_id, scored.get(node_id)
            )

        rewards: dict[str, float] = {}
        infos: dict[str, dict] = {}
        for node_id in range(self.num_nodes):
            aid = agent_id(node_id)
            action = int(action_dict.get(aid, self.reject_action))
            if not needed[node_id]:
                action = self.reject_action
            pen, overlap_n = self._admit(node_id, action)
            requested = scored.get(node_id)
            if requested is not None:
                self.network.nodes[node_id].record_request(
                    int(requested), self.observation_window
                )

            self._episode_overlap += overlap_n
            self._episode_overlap_penalty += pen
            task_r = task_rewards[node_id]
            self._episode_task_reward += task_r
            total_r = task_r + pen
            rewards[aid] = total_r
            infos[aid] = {
                "timestep": self.timestep + 1,
                "requested": requested,
                "needs_decision": needed[node_id],
                "acted": needed[node_id],
                "action": action,
                "task_reward": task_r,
                "overlap_penalty": pen,
                "overlap_count": overlap_n,
                **self._node_rates(node_id),
            }

        self.timestep += 1
        truncated = self.timestep >= self.episode_length
        terminateds = {aid: False for aid in self.possible_agents}
        truncateds = {aid: truncated for aid in self.possible_agents}
        terminateds["__all__"] = False
        truncateds["__all__"] = truncated

        if truncated:
            self.agents = []
            self._pending_requests = {i: None for i in range(self.num_nodes)}
        else:
            self._draw_pending_requests()

        return self._get_observations(), rewards, terminateds, truncateds, infos

    def render(self) -> None:
        """Print per-node cache contents and aggregate hit/forward/cloud rates."""
        stats = self.network_stats()
        print(
            f"t={self.timestep} level={self.comm_level} hit={stats['hit_rate']:.2f} "
            f"fwd={stats['forward_rate']:.2f} cloud={stats['cloud_rate']:.2f} "
            f"diversity={stats['cache_diversity']}/{self.num_nodes} "
            f"n_actions={stats['n_actions']}"
        )
        for node_id, cache in stats["caches"].items():
            node = self.network.nodes[node_id]
            print(
                f"  node={node_id} cluster={self.network.cluster_for_node[node_id]} "
                f"cache={cache} h/f/m={node.hits}/{node.forwards}/{node.misses}"
            )
