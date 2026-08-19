from __future__ import annotations

import numpy as np
import pytest

from agents.baselines import LFUPolicy, LRUPolicy
from configs import load_config
from env.edge_node import EdgeNode
from env.multi_agent_caching_env import local_obs_size


@pytest.fixture
def config():
    return load_config()


def _make_observation(
    k: int,
    cached: list[int],
    cache_capacity: int,
    request_counts: list[int] | None = None,
    pending: int | None = None,
    needs_decision: bool | None = None,
) -> np.ndarray:
    slots = np.zeros(cache_capacity * k, dtype=np.float32)
    for i, cid in enumerate(cached):
        slots[i * k + cid] = 1.0
    utilization = np.array([len(cached) / cache_capacity], dtype=np.float32)
    request_freq = np.zeros(k, dtype=np.float32)
    if request_counts is not None:
        for container_id, count in enumerate(request_counts):
            request_freq[container_id] = float(count)
        max_freq = request_freq.max()
        if max_freq > 0:
            request_freq /= max_freq
    request = np.zeros(k, dtype=np.float32)
    if pending is not None:
        request[pending] = 1.0
    if needs_decision is None:
        needs_decision = (
            pending is not None
            and pending not in cached
            and len(cached) >= cache_capacity
        )
    need = np.array([1.0 if needs_decision else 0.0], dtype=np.float32)
    obs = np.concatenate([slots, utilization, request_freq, request, need])
    assert obs.shape == (local_obs_size(k, cache_capacity),)
    return obs


def test_lru_reject_when_cached(config):
    k = config["num_container_types"]
    c = config["cache_capacity"]
    policy = LRUPolicy()
    obs = _make_observation(k, cached=[0], cache_capacity=c, pending=0)
    assert policy.act(obs, requested=0, cache=[0], cache_capacity=c) == c


def test_lru_no_decision_when_not_full(config):
    k = config["num_container_types"]
    c = config["cache_capacity"]
    policy = LRUPolicy()
    obs = _make_observation(k, cached=[0], cache_capacity=c, pending=3)
    assert policy.act(obs, requested=3, cache=[0], cache_capacity=c) == c


def test_lru_evicts_oldest_slot(config):
    k = 5
    cap = 3
    policy = LRUPolicy()
    node = EdgeNode(0, cap)
    for container_id in [0, 1, 2]:
        node.cache_container(container_id)

    obs = _make_observation(k, cached=node.cache, cache_capacity=cap, pending=4)
    action = policy.act(
        obs, requested=4, cache=node.cache, cache_capacity=cap, num_container_types=k
    )
    assert action == 0
    node.evict_slot(action)
    node.cache_container(4)
    assert node.cache == [1, 2, 4]
    assert 0 not in node.cache_set


def test_lfu_reject_when_cached(config):
    k = config["num_container_types"]
    c = config["cache_capacity"]
    policy = LFUPolicy()
    obs = _make_observation(k, cached=[2], cache_capacity=c, pending=2)
    assert policy.act(obs, requested=2, cache=[2], cache_capacity=c) == c


def test_lfu_no_decision_when_not_full(config):
    k = config["num_container_types"]
    c = config["cache_capacity"]
    policy = LFUPolicy()
    obs = _make_observation(k, cached=[1], cache_capacity=c, pending=4)
    assert policy.act(obs, requested=4, cache=[1], cache_capacity=c) == c


def test_lfu_evicts_least_frequent_slot(config):
    k = 5
    cap = 3
    policy = LFUPolicy()
    node = EdgeNode(0, cap)
    for container_id in [0, 1, 2]:
        node.cache_container(container_id)
    for container_id in [0, 0, 2, 2, 2]:
        node.record_request(container_id, observation_window=10)

    obs = _make_observation(
        k,
        cached=node.cache,
        cache_capacity=cap,
        request_counts=[2, 0, 3, 0, 0],
        pending=4,
    )
    action = policy.act(
        obs, requested=4, cache=node.cache, cache_capacity=cap, num_container_types=k
    )
    assert action == 1
    assert node.evict_slot(action) == 1
    node.cache_container(4)
    assert node.cache == [0, 2, 4]
