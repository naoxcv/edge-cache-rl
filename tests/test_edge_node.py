import pytest

from configs import load_config
from env.edge_node import EdgeNode


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def small_node():
    return EdgeNode(node_id=0, cache_capacity=3)


def test_edge_node_init(config):
    node = EdgeNode(0, config["cache_capacity"])
    assert node.node_id == 0
    assert node.cache_capacity == config["cache_capacity"]
    assert node.cache == []
    assert node.cache_set == set()
    assert node.request_history == []
    assert node.hits == 0
    assert node.misses == 0
    assert node.forwards == 0


def test_is_cached(small_node):
    assert not small_node.is_cached(0)
    small_node.cache_container(0)
    assert small_node.is_cached(0)
    assert not small_node.is_cached(1)


def test_cache_container_no_eviction_when_not_full(small_node):
    assert small_node.cache_container(0) is None
    assert small_node.cache_container(1) is None
    assert small_node.cache == [0, 1]
    assert small_node.cache_set == {0, 1}


def test_cache_container_returns_none_for_duplicate(small_node):
    small_node.cache_container(0)
    assert small_node.cache_container(0) is None
    assert small_node.cache == [0]


def test_cache_container_evicts_oldest_when_full(small_node):
    for cid in [0, 1, 2]:
        assert small_node.cache_container(cid) is None

    evicted = small_node.cache_container(3)

    assert evicted == 0
    assert small_node.cache == [1, 2, 3]
    assert not small_node.is_cached(0)
    assert small_node.is_cached(3)
    assert small_node.cache_set == {1, 2, 3}


def test_cache_container_evicts_in_fifo_order(small_node):
    for cid in [0, 1, 2]:
        small_node.cache_container(cid)

    assert small_node.cache_container(3) == 0
    assert small_node.cache_container(4) == 1
    assert small_node.cache == [2, 3, 4]


def test_cache_container_full_cache_behavior(config):
    cap = config["cache_capacity"]
    node = EdgeNode(0, cap)

    for i in range(cap):
        assert node.cache_container(i) is None

    evicted = node.cache_container(99)

    assert evicted == 0
    assert len(node.cache) == cap
    assert 99 in node.cache_set
    assert 0 not in node.cache_set


def test_evict_container(small_node):
    small_node.cache_container(0)
    small_node.cache_container(1)

    assert small_node.evict_container(0) is True
    assert small_node.cache == [1]
    assert small_node.cache_set == {1}
    assert not small_node.is_cached(0)

    assert small_node.evict_container(99) is False
    assert small_node.cache == [1]


def test_record_request_sliding_window(small_node, config):
    window = 5
    for i in range(window + 10):
        small_node.record_request(i, window)

    assert len(small_node.request_history) == window
    assert small_node.request_history == list(range(10, window + 10))


def test_get_state_shape_and_range(config):
    k = config["num_container_types"]
    window = config["observation_window"]
    node = EdgeNode(0, config["cache_capacity"])

    node.cache_container(0)
    node.cache_container(2)
    for i in [0, 0, 1, 2, 2, 2]:
        node.record_request(i, window)

    state = node.get_state(k, window)

    assert state.shape == (2 * k + 1,)
    assert state.min() >= 0.0
    assert state.max() <= 1.0
    assert state[0] == 1.0
    assert state[2] == 1.0
    assert state[1] == 0.0
    assert state[k] == pytest.approx(2 / config["cache_capacity"])
    assert state[k + 1 + 2] == 1.0  # container 2 most frequent


def test_reset_clears_everything(small_node, config):
    window = config["observation_window"]
    small_node.cache_container(0)
    small_node.cache_container(1)
    small_node.record_request(0, window)
    small_node.hits = 3
    small_node.misses = 2
    small_node.forwards = 1

    small_node.reset()

    assert small_node.cache == []
    assert small_node.cache_set == set()
    assert small_node.request_history == []
    assert small_node.hits == 0
    assert small_node.misses == 0
    assert small_node.forwards == 0
