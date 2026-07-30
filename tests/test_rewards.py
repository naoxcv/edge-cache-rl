from __future__ import annotations

from configs import load_config
from env.edge_network import EdgeNetwork
from env.rewards import score_cache_request


def test_score_local_hit():
    config = load_config()
    network = EdgeNetwork(config)
    network.nodes[0].cache_container(3)

    reward = score_cache_request(network, 0, 3, config)

    assert reward == config["reward_local_hit"]
    assert network.nodes[0].hits == 1


def test_score_forward_same_cluster():
    config = load_config()
    network = EdgeNetwork(config)
    peer = network.get_cluster_neighbors(0)[0]
    network.nodes[peer].cache_container(4)

    reward = score_cache_request(
        network, 0, 4, config, enable_forwarding=True, same_cluster_only=True
    )

    assert reward == config["reward_forward_hit"]
    assert network.nodes[0].forwards == 1


def test_score_no_forward_when_disabled():
    config = load_config()
    network = EdgeNetwork(config)
    peer = network.get_cluster_neighbors(0)[0]
    network.nodes[peer].cache_container(4)

    reward = score_cache_request(
        network, 0, 4, config, enable_forwarding=False, same_cluster_only=True
    )

    assert reward == config["reward_cloud_pull"]
    assert network.nodes[0].misses == 1
    assert network.nodes[0].forwards == 0


def test_score_same_cluster_ignores_bridge_neighbor():
    config = load_config()
    network = EdgeNetwork(config)
    # Node 1 is a bridge neighbor of 0 but in a different cluster.
    assert 1 in network.get_neighbors(0)
    assert 1 not in network.get_cluster_neighbors(0)
    network.nodes[1].cache_container(8)

    reward = score_cache_request(
        network, 0, 8, config, enable_forwarding=True, same_cluster_only=True
    )

    assert reward == config["reward_cloud_pull"]
