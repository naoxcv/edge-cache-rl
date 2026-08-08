from __future__ import annotations

import gymnasium as gym
import numpy as np


class RandomTrafficSeedWrapper(gym.Wrapper):
    """Sample a new traffic/catalog seed on every episode reset."""

    def __init__(
        self,
        env: gym.Env,
        *,
        base_seed: int = 42,
        seed_range: int = 10_000,
    ) -> None:
        super().__init__(env)
        self.base_seed = base_seed
        self.seed_range = seed_range
        self._rng = np.random.default_rng(base_seed)
        self._episode_count = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Reset the inner env with a fresh random traffic seed."""
        episode_seed = int(self._rng.integers(0, self.seed_range))
        self._episode_count += 1
        obs, info = self.env.reset(seed=episode_seed)
        info = dict(info)
        info["traffic_seed"] = episode_seed
        info["episode_index"] = self._episode_count
        return obs, info
