from __future__ import annotations

import numpy as np
import pytest

from agents.baselines import LFUPolicy, LRUPolicy
from configs import load_config
from env.edge_node import EdgeNode


@pytest.fixture
def config():
    return load_config()


def _make_observation(
    k: int,
    cached: list[int],
    cache_capacity: int,
    request_counts: list[int] | None = None,
) -> np.ndarray:
    cache_binary = np.zeros(k, dtype=np.float32)
    for container_id in cached:
        cache_binary[container_id] = 1.0

    utilization = np.array([len(cached) / cache_capacity], dtype=np.float32)

    request_freq = np.zeros(k, dtype=np.float32)
    if request_counts is not None:
        for container_id, count in enumerate(request_counts):
            request_freq[container_id] = float(count)
        max_freq = request_freq.max()
        if max_freq > 0:
            request_freq /= max_freq

    return np.concatenate([cache_binary, utilization, request_freq])


def test_lru_noop_when_cached(config):
    k = config["num_container_types"]
    policy = LRUPolicy()
    obs = _make_observation(k, cached=[0], cache_capacity=config["cache_capacity"])

    assert policy.act(obs, requested=0) == 2 * k


def test_lru_caches_when_not_full(config):
    k = config["num_container_types"]
    policy = LRUPolicy()
    obs = _make_observation(k, cached=[0], cache_capacity=config["cache_capacity"])

    assert policy.act(obs, requested=3) == 3


def test_lru_evicts_oldest_entry(config):
    k = 3
    cap = 3
    policy = LRUPolicy()
    node = EdgeNode(0, cap)

    for container_id in [0, 1, 2]:
        node.cache_container(container_id)

    obs = node.get_state(k, observation_window=10)
    action = policy.act(obs, requested=5, cache=node.cache)

    assert action == 5
    node.cache_container(5)
    assert node.cache == [1, 2, 5]
    assert 0 not in node.cache_set


def test_lfu_noop_when_cached(config):
    k = config["num_container_types"]
    policy = LFUPolicy()
    obs = _make_observation(k, cached=[2], cache_capacity=config["cache_capacity"])

    assert policy.act(obs, requested=2) == 2 * k


def test_lfu_caches_when_not_full(config):
    k = config["num_container_types"]
    policy = LFUPolicy()
    obs = _make_observation(k, cached=[1], cache_capacity=config["cache_capacity"])

    assert policy.act(obs, requested=4) == 4


def test_lfu_evicts_least_frequent_entry(config):
    k = 3
    cap = 3
    policy = LFUPolicy()
    node = EdgeNode(0, cap)

    for container_id in [0, 1, 2]:
        node.cache_container(container_id)

    for container_id in [0, 0, 2, 2, 2]:
        node.record_request(container_id, observation_window=10)

    obs = node.get_state(k, observation_window=10)
    action = policy.act(obs, requested=5, cache=node.cache)

    assert action == k + 1
    assert node.evict_container(1) is True
    node.cache_container(5)
    assert node.cache == [0, 2, 5]


def test_lfu_caches_directly_when_lfu_is_oldest(config):
    k = 3
    cap = 3
    policy = LFUPolicy()
    node = EdgeNode(0, cap)

    for container_id in [0, 1, 2]:
        node.cache_container(container_id)

    for container_id in [1, 1, 2, 2]:
        node.record_request(container_id, observation_window=10)

    obs = node.get_state(k, observation_window=10)
    action = policy.act(obs, requested=5, cache=node.cache)

    assert action == 5
