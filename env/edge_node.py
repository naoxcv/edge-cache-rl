from __future__ import annotations

import numpy as np

class EdgeNode:
    def __init__(self, node_id: int, cache_capacity: int):
        self.node_id = node_id
        self.cache_capacity = cache_capacity
        self.cache: list[int] = []       # list of container IDs, ordered by recency
        self.cache_set: set[int] = set() # for O(1) lookup
        self.request_history: list[int] = []  # recent request IDs for state vector

        # Counters
        self.hits = 0
        self.misses = 0
        self.forwards = 0

    def is_cached(self, container_id: int) -> bool:
        return container_id in self.cache_set

    def cache_container(self, container_id: int) -> int | None:
        """Add container to cache. Returns evicted container ID if cache was full, else None."""
        if container_id in self.cache_set:
            return None
        evicted = None
        if len(self.cache) >= self.cache_capacity:
            evicted = self.cache.pop(0)
            self.cache_set.remove(evicted)
        self.cache.append(container_id)
        self.cache_set.add(container_id)
        return evicted

    def evict_container(self, container_id: int) -> bool:
        """Manually evict a specific container. Returns True if it was cached."""
        if container_id in self.cache_set:
            self.cache.remove(container_id)
            self.cache_set.remove(container_id)
            return True
        return False

    def record_request(self, container_id: int, observation_window: int) -> None:
        """Append to request history, maintain sliding window."""
        self.request_history.append(container_id)
        if len(self.request_history) > observation_window:
            self.request_history.pop(0)

    def get_state(self, num_container_types: int, observation_window: int) -> np.ndarray:
        """Return observation vector: [cache_binary (K,), utilization (1,), request_freq (K,)]"""
        cache_binary = np.zeros(num_container_types, dtype=np.float32)
        for container_id in self.cache:
            cache_binary[container_id] = 1.0

        utilization = np.array([len(self.cache) / self.cache_capacity], dtype=np.float32)

        request_freq = np.zeros(num_container_types, dtype=np.float32)
        for container_id in self.request_history[-observation_window:]:
            request_freq[container_id] += 1.0
        max_freq = request_freq.max()
        if max_freq > 0:
            request_freq /= max_freq

        return np.concatenate([cache_binary, utilization, request_freq])

    def reset(self):
        """Clear cache, history, counters."""
        self.cache.clear()
        self.cache_set.clear()
        self.request_history.clear()
        self.hits = 0
        self.misses = 0
        self.forwards = 0