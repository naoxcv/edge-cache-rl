from __future__ import annotations

import numpy as np

from env.multi_agent_caching_env import local_obs_size, needs_decision_from_obs


def _capacity(cache_capacity: int | None, cache: list[int] | None) -> int:
    if cache_capacity is not None:
        return int(cache_capacity)
    if cache is not None and len(cache) > 0:
        return len(cache)
    raise ValueError("cache_capacity is required for eviction-only baselines")


def _freqs(observation: np.ndarray, k: int, c: int) -> np.ndarray:
    local = local_obs_size(k, c)
    obs = np.asarray(observation).reshape(-1)[:local]
    start = c * k + 1
    return obs[start : start + k]


class LRUPolicy:
    """Always admit misses; on a full cache, evict LRU slot 0."""

    def act(
        self,
        observation: np.ndarray,
        requested: int | None,
        *,
        cache: list[int] | None = None,
        cache_capacity: int | None = None,
        num_container_types: int | None = None,
    ) -> int:
        """Return the LRU slot (0) when a decision is required, else reject (C)."""
        c = _capacity(cache_capacity, cache)
        if requested is None:
            return c
        if cache is not None:
            if requested in cache or len(cache) < c:
                return c
            return 0
        k = int(num_container_types) if num_container_types is not None else None
        if k is not None:
            local = observation[: local_obs_size(k, c)]
            if not needs_decision_from_obs(local):
                return c
        elif not needs_decision_from_obs(observation):
            return c
        return 0


class LFUPolicy:
    """Always admit misses; on a full cache, evict the least-frequent slot."""

    def act(
        self,
        observation: np.ndarray,
        requested: int | None,
        *,
        cache: list[int] | None = None,
        cache_capacity: int | None = None,
        num_container_types: int | None = None,
    ) -> int:
        """Return the LFU slot index when a decision is required, else reject (C)."""
        c = _capacity(cache_capacity, cache)
        if requested is None:
            return c
        if cache is not None and (requested in cache or len(cache) < c):
            return c

        k = int(num_container_types) if num_container_types is not None else None
        if k is None:
            # Infer K from local layout given C.
            n = min(int(np.asarray(observation).reshape(-1).shape[0]), local_obs_size(64, c))
            if (n - 2) % (c + 2) != 0:
                raise ValueError("num_container_types required to decode LFU obs")
            k = (n - 2) // (c + 2)

        local = np.asarray(observation).reshape(-1)[: local_obs_size(k, c)]
        if cache is None and not needs_decision_from_obs(local):
            return c

        freqs = _freqs(local, k, c)
        items = cache if cache is not None else [
            int(np.argmax(local[slot * k : (slot + 1) * k]))
            for slot in range(c)
            if float(local[slot * k : (slot + 1) * k].sum()) > 0.5
        ]
        if not items:
            return c
        return min(range(len(items)), key=lambda slot: (float(freqs[items[slot]]), slot))
