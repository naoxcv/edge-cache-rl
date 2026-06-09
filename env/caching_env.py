from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.container import create_catalog
from env.edge_network import EdgeNetwork
from env.request_generator import RequestGenerator


class CachingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict | None = None, seed: int = 42):
        if config is None:
            from configs import load_config

            config = load_config()

        self.config = config
        self.active_node = 0
        self.num_container_types = config["num_container_types"]
        self.observation_window = config["observation_window"]
        self.episode_length = config["episode_length"]
        self.timestep = 0
        self._seed = seed

        self.catalog = create_catalog(self.num_container_types, seed=seed)
        self.network = EdgeNetwork(config)
        self.request_generator = RequestGenerator(config, self.catalog, seed=seed)

        obs_size = 2 * self.num_container_types + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2 * self.num_container_types + 1)

    def _active_node(self):
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
        if container_id is None:
            return 0.0

        node = self._active_node()
        if node.is_cached(container_id):
            node.hits += 1
            return self.config["reward_local_hit"]

        neighbor = self.network.find_any_neighbor_with(self.active_node, container_id)
        if neighbor is not None:
            node.forwards += 1
            return self.config["reward_forward_hit"]

        node.misses += 1
        return self.config["reward_cloud_pull"]

    def _cache_hit_rate(self) -> float:
        node = self._active_node()
        total = node.hits + node.misses + node.forwards
        if total == 0:
            return 0.0
        return node.hits / total

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.catalog = create_catalog(self.num_container_types, seed=seed)
            self.request_generator = RequestGenerator(
                self.config, self.catalog, seed=seed
            )

        self.timestep = 0
        self.network.reset()
        self.request_generator.reset()

        observation = self._get_observation()
        info = {"cache_hit_rate": self._cache_hit_rate(), "timestep": self.timestep}
        return observation, info

    def step(self, action: int):
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

    def render(self):
        node = self._active_node()
        print(
            f"t={self.timestep} node={self.active_node} "
            f"cache={node.cache} hits={node.hits} "
            f"forwards={node.forwards} misses={node.misses} "
            f"hit_rate={self._cache_hit_rate():.2f}"
        )
