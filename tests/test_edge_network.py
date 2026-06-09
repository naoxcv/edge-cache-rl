import numpy as np
import pytest

from configs import load_config
from env.edge_network import EdgeNetwork


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def network(config):
    return EdgeNetwork(config)


def test_cluster_count(network, config):
    clusters = {network.cluster_for_node[i] for i in range(config["num_nodes"])}
    assert len(clusters) == config["num_clusters"]


def test_round_robin_cluster_assignment(network, config):
    for node_id in range(config["num_nodes"]):
        assert network.cluster_for_node[node_id] == node_id % config["num_clusters"]


def test_latency_matrix_symmetric(network):
    assert np.allclose(network.latency_matrix, network.latency_matrix.T)


def test_latency_matrix_diagonal_zero(network):
    assert np.all(np.diag(network.latency_matrix) == 0)


def test_intra_latency_less_than_inter(network, config):
    intra = config["intra_cluster_latency_ms"]
    inter = config["inter_cluster_latency_ms"]
    assert intra < inter

    for i in range(network.num_nodes):
        for j in range(network.num_nodes):
            if i == j:
                continue
            if network.cluster_for_node[i] == network.cluster_for_node[j]:
                assert network.latency_matrix[i, j] == intra
            else:
                assert network.latency_matrix[i, j] == inter


def test_get_neighbors_hierarchical(network):
    # cluster 0: 0, 3, 6, 9 — fully connected + bridge 0-1
    assert network.get_neighbors(0) == [1, 3, 6, 9]

    # cluster 1: 1, 4, 7 — bridges 0-1 and 1-2
    assert network.get_neighbors(1) == [0, 2, 4, 7]


def test_find_any_neighbor_with_returns_none_when_no_neighbor_has_it(network):
    network.nodes[0].cache_container(99)
    assert network.find_any_neighbor_with(0, 99) is None


def test_find_any_neighbor_with_returns_first_matching_neighbor(network):
    network.nodes[1].cache_container(5)
    network.nodes[3].cache_container(5)

    assert network.find_any_neighbor_with(0, 5) == 1


def test_find_any_neighbor_with_ignores_non_neighbors(network):
    network.nodes[2].cache_container(5)
    network.nodes[3].cache_container(5)

    assert network.find_any_neighbor_with(0, 5) == 3


def test_get_forwarding_cost(network, config):
    assert network.get_forwarding_cost(0, 0) == 0.0
    assert network.get_forwarding_cost(0, 3) == config["intra_cluster_latency_ms"]
    assert network.get_forwarding_cost(0, 1) == config["inter_cluster_latency_ms"]


def test_reset_clears_all_nodes(network, config):
    window = config["observation_window"]
    for node in network.nodes:
        node.cache_container(0)
        node.record_request(0, window)
        node.hits = 1

    network.reset()

    for node in network.nodes:
        assert node.cache == []
        assert node.request_history == []
        assert node.hits == 0
