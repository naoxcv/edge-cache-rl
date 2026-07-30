from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.container import create_catalog
from env.edge_network import EdgeNetwork
from env.request_generator import RequestGenerator
from env.rewards import score_cache_request


def agent_id(node_id: int) -> str:
    return str(node_id)


class MultiAgentCachingEnv(gym.Env):
    """Multi-node caching env with per-agent dict observations/rewards.

    Step order (all nodes each timestep):
      1. Apply each node's action
      2. Generate one request per node
      3. Score rewards (local hit / optional forward / cloud)
      4. Update request histories

    Level 0 (``comm_level=0``): each agent sees only local state (shape 2K+1).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict | None = None, seed: int = 42):
        if config is None:
            from configs import load_config

            config = load_config()

        self.config = config
        self.num_nodes = int(config["num_nodes"])
        self.num_container_types = int(config["num_container_types"])
        self.observation_window = int(config["observation_window"])
        self.episode_length = int(config["episode_length"])
        self.enable_forwarding = bool(config.get("enable_forwarding", True))
        self.same_cluster_only = bool(config.get("forwarding_same_cluster_only", True))
        self.comm_level = int(config.get("comm_level", 0))
        self.timestep = 0
        self._seed = seed

        self.catalog = create_catalog(self.num_container_types, seed=seed)
        self.network = EdgeNetwork(config)
        self.request_generator = RequestGenerator(config, self.catalog, seed=seed)

        self.possible_agents = [agent_id(i) for i in range(self.num_nodes)]
        self.agents = list(self.possible_agents)

        obs_size = self._obs_size()
        single_obs = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)
        single_act = spaces.Discrete(2 * self.num_container_types + 1)
        self.observation_spaces = {aid: single_obs for aid in self.possible_agents}
        self.action_spaces = {aid: single_act for aid in self.possible_agents}
        # Compatibility shims for code that expects gym.Env spaces.
        self.observation_space = single_obs
        self.action_space = single_act

    def _obs_size(self) -> int:
        # Level 0: local state only. Levels 1-3 expand this in week 5+.
        return 2 * self.num_container_types + 1

    def get_observation_space(self, agent_id_: str) -> gym.Space:
        return self.observation_spaces[agent_id_]

    def get_action_space(self, agent_id_: str) -> gym.Space:
        return self.action_spaces[agent_id_]

    def _get_observation(self, node_id: int) -> np.ndarray:
        return self.network.nodes[node_id].get_state(
            self.num_container_types, self.observation_window
        )

    def _get_observations(self) -> dict[str, np.ndarray]:
        return {agent_id(i): self._get_observation(i) for i in range(self.num_nodes)}

    def _apply_action(self, node_id: int, action: int) -> None:
        node = self.network.nodes[node_id]
        k = self.num_container_types
        action = int(action)
        if action < k:
            node.cache_container(action)
        elif action < 2 * k:
            node.evict_container(action - k)

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

    def network_stats(self) -> dict[str, Any]:
        hits = sum(n.hits for n in self.network.nodes)
        forwards = sum(n.forwards for n in self.network.nodes)
        misses = sum(n.misses for n in self.network.nodes)
        total = hits + forwards + misses
        return {
            "hits": hits,
            "forwards": forwards,
            "misses": misses,
            "hit_rate": hits / total if total else 0.0,
            "forward_rate": forwards / total if total else 0.0,
            "cloud_rate": misses / total if total else 0.0,
            "caches": {i: list(self.network.nodes[i].cache) for i in range(self.num_nodes)},
        }

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        if seed is not None:
            self._seed = seed
            self.catalog = create_catalog(self.num_container_types, seed=seed)
            self.request_generator = RequestGenerator(
                self.config, self.catalog, seed=seed
            )

        self.timestep = 0
        self.agents = list(self.possible_agents)
        self.network.reset()
        self.request_generator.reset()

        obs = self._get_observations()
        infos = {
            agent_id(i): {"timestep": 0, **self._node_rates(i)}
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
        for aid, action in action_dict.items():
            self._apply_action(int(aid), action)

        requests = self.request_generator.generate()
        rewards: dict[str, float] = {}
        infos: dict[str, dict] = {}

        for node_id in range(self.num_nodes):
            requested = requests[node_id]
            reward = self._process_request(node_id, requested)
            if requested is not None:
                self.network.nodes[node_id].record_request(
                    requested, self.observation_window
                )
            aid = agent_id(node_id)
            rewards[aid] = reward
            infos[aid] = {
                "timestep": self.timestep + 1,
                "requested": requested,
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

        return self._get_observations(), rewards, terminateds, truncateds, infos

    def render(self):
        stats = self.network_stats()
        print(
            f"t={self.timestep} hit={stats['hit_rate']:.2f} "
            f"fwd={stats['forward_rate']:.2f} cloud={stats['cloud_rate']:.2f}"
        )
        for node_id, cache in stats["caches"].items():
            node = self.network.nodes[node_id]
            print(
                f"  node={node_id} cluster={self.network.cluster_for_node[node_id]} "
                f"cache={cache} h/f/m={node.hits}/{node.forwards}/{node.misses}"
            )
