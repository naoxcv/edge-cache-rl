from __future__ import annotations

import numpy as np


def _num_container_types(observation: np.ndarray) -> int:
    return (len(observation) - 1) // 2


def _noop_action(k: int) -> int:
    return 2 * k


def _cached_ids(observation: np.ndarray) -> list[int]:
    k = _num_container_types(observation)
    return [i for i in range(k) if observation[i] > 0.5]


def _is_cached(observation: np.ndarray, container_id: int) -> bool:
    k = _num_container_types(observation)
    if container_id < 0 or container_id >= k:
        return False
    return observation[container_id] > 0.5


def _cache_full(observation: np.ndarray) -> bool:
    k = _num_container_types(observation)
    return observation[k] >= 1.0 - 1e-6


class LRUPolicy:
    def act(
        self,
        observation: np.ndarray,
        requested: int | None,
        *,
        cache: list[int] | None = None,
    ) -> int:
        k = _num_container_types(observation)

        if requested is None or _is_cached(observation, requested):
            return _noop_action(k)

        if not _cache_full(observation):
            return requested

        # Cache full: cache_container evicts cache[0] (oldest / least recent).
        return requested


class LFUPolicy:
    def _lfu_victim(
        self, observation: np.ndarray, cached: list[int]
    ) -> int:
        k = _num_container_types(observation)
        freqs = observation[k + 1 : 2 * k + 1]
        return min(cached, key=lambda container_id: (freqs[container_id], container_id))

    def act(
        self,
        observation: np.ndarray,
        requested: int | None,
        *,
        cache: list[int] | None = None,
    ) -> int:
        k = _num_container_types(observation)

        if requested is None or _is_cached(observation, requested):
            return _noop_action(k)

        if not _cache_full(observation):
            return requested

        cached = cache if cache is not None else _cached_ids(observation)
        victim = self._lfu_victim(observation, cached)

        if cache is not None and victim == cache[0]:
            return requested

        return k + victim
