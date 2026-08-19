from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.container import create_catalog
from env.edge_network import EdgeNetwork
from env.edge_node import EdgeNode
from env.multi_agent_caching_env import local_obs_size
from env.request_generator import RequestGenerator
from env.rewards import score_cache_request


class CachingEnv(gym.Env):
    """Single-node Gymnasium wrapper (active node 0) over EdgeNetwork.

    Same eviction-only MDP as MultiAgentCachingEnv: score the pending request,
    then admit on a miss (auto-insert if space, else evict slot or reject).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict | None = None, seed: int = 42) -> None:
        if config is None:
            from configs import load_config

            config = load_config()

        self.config = config
        self.active_node = 0
        self.num_container_types = config["num_container_types"]
        self.cache_capacity = int(config["cache_capacity"])
        self.observation_window = config["observation_window"]
        self.episode_length = config["episode_length"]
        self.enable_forwarding = bool(config.get("enable_forwarding", True))
        self.same_cluster_only = bool(config.get("forwarding_same_cluster_only", True))
        self.reject_action = self.cache_capacity
        self.timestep = 0
        self._seed = seed
        self._pending: int | None = None

        self.catalog = create_catalog(self.num_container_types, seed=seed)
        self.network = EdgeNetwork(config)
        self.request_generator = RequestGenerator(
            config,
            self.catalog,
            seed=seed,
            cluster_for_node=self.network.cluster_for_node,
        )

        obs_size = local_obs_size(self.num_container_types, self.cache_capacity)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.cache_capacity + 1)

    def _active_node(self) -> EdgeNode:
        return self.network.nodes[self.active_node]

    def _needs_decision(self) -> bool:
        if self._pending is None:
            return False
        node = self._active_node()
        if node.is_cached(int(self._pending)):
            return False
        return len(node.cache) >= node.cache_capacity

    def _get_observation(self) -> np.ndarray:
        node = self._active_node()
        k = self.num_container_types
        c = self.cache_capacity
        slots = node.get_cache_slots(c, k)
        utilization = np.array([len(node.cache) / max(c, 1)], dtype=np.float32)
        freq = node.get_request_freq(k, self.observation_window)
        request = np.zeros(k, dtype=np.float32)
        if self._pending is not None and 0 <= int(self._pending) < k:
            request[int(self._pending)] = 1.0
        need = np.array([1.0 if self._needs_decision() else 0.0], dtype=np.float32)
        return np.concatenate([slots, utilization, freq, request, need])

    def _admit(self, action: int) -> None:
        requested = self._pending
        node = self._active_node()
        if requested is None or node.is_cached(int(requested)):
            return
        requested = int(requested)
        if len(node.cache) < node.cache_capacity:
            node.cache_container(requested)
            return
        action = int(action)
        if 0 <= action < len(node.cache):
            node.evict_slot(action)
            node.cache_container(requested)

    def _process_request(self, container_id: int | None) -> float:
        return score_cache_request(
            self.network,
            self.active_node,
            container_id,
            self.config,
            enable_forwarding=self.enable_forwarding,
            same_cluster_only=self.same_cluster_only,
        )

    def _cache_hit_rate(self) -> float:
        node = self._active_node()
        total = node.hits + node.misses + node.forwards
        if total == 0:
            return 0.0
        return node.hits / total

    def _draw_pending(self) -> None:
        requests = self.request_generator.generate()
        self._pending = requests[self.active_node]

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Reset the environment, optionally re-seeding catalog and traffic."""
        super().reset(seed=seed)
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
        self.network.reset()
        self.request_generator.reset()
        self._draw_pending()

        observation = self._get_observation()
        info = {
            "cache_hit_rate": self._cache_hit_rate(),
            "timestep": self.timestep,
            "requested": self._pending,
            "needs_decision": self._needs_decision(),
        }
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Score the pending request, admit if needed, then draw the next request."""
        scored = self._pending
        needed = self._needs_decision()
        reward = self._process_request(scored)
        if needed:
            self._admit(int(action))
        else:
            self._admit(self.reject_action)
        if scored is not None:
            self._active_node().record_request(int(scored), self.observation_window)

        self.timestep += 1
        terminated = False
        truncated = self.timestep >= self.episode_length
        if truncated:
            self._pending = None
        else:
            self._draw_pending()

        observation = self._get_observation()
        info = {
            "cache_hit_rate": self._cache_hit_rate(),
            "timestep": self.timestep,
            "requested": scored,
            "needs_decision": needed,
        }
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """Print a one-line status summary for the active node."""
        node = self._active_node()
        print(
            f"t={self.timestep} node={self.active_node} "
            f"cache={node.cache} hits={node.hits} "
            f"forwards={node.forwards} misses={node.misses} "
            f"hit_rate={self._cache_hit_rate():.2f}"
        )
