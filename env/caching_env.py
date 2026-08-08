from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.container import create_catalog
from env.edge_network import EdgeNetwork
from env.edge_node import EdgeNode
from env.request_generator import RequestGenerator
from env.rewards import score_cache_request


class CachingEnv(gym.Env):
    """Single-node Gymnasium wrapper (active node 0) over EdgeNetwork.

    Step order: apply action → generate requests → score reward → update history.
    Forwarding honors ``enable_forwarding`` and ``forwarding_same_cluster_only``.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict | None = None, seed: int = 42) -> None:
        if config is None:
            from configs import load_config

            config = load_config()

        self.config = config
        self.active_node = 0
        self.num_container_types = config["num_container_types"]
        self.observation_window = config["observation_window"]
        self.episode_length = config["episode_length"]
        self.enable_forwarding = bool(config.get("enable_forwarding", True))
        self.same_cluster_only = bool(config.get("forwarding_same_cluster_only", True))
        self.timestep = 0
        self._seed = seed

        self.catalog = create_catalog(self.num_container_types, seed=seed)
        self.network = EdgeNetwork(config)
        self.request_generator = RequestGenerator(
            config,
            self.catalog,
            seed=seed,
            cluster_for_node=self.network.cluster_for_node,
        )

        obs_size = 2 * self.num_container_types + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2 * self.num_container_types + 1)

    def _active_node(self) -> EdgeNode:
        return self.network.nodes[self.active_node]

    def _get_observation(self) -> np.ndarray:
        return self._active_node().get_state(
            self.num_container_types, self.observation_window
        )

    def _apply_action(self, action: int) -> None:
        node = self._active_node()
        k = self.num_container_types

        if action < k:
            node.cache_container(action)
        elif action < 2 * k:
            node.evict_container(action - k)
        # action == 2*k is no-op

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

        observation = self._get_observation()
        info = {"cache_hit_rate": self._cache_hit_rate(), "timestep": self.timestep}
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply action, generate request, score reward, and advance timestep."""
        self._apply_action(int(action))

        requests = self.request_generator.generate()
        requested = requests[self.active_node]
        reward = self._process_request(requested)

        if requested is not None:
            self._active_node().record_request(requested, self.observation_window)

        self.timestep += 1
        terminated = False
        truncated = self.timestep >= self.episode_length

        observation = self._get_observation()
        info = {
            "cache_hit_rate": self._cache_hit_rate(),
            "timestep": self.timestep,
            "requested": requested,
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
