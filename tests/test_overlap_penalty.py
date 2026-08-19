from __future__ import annotations

from configs import load_config
from env.edge_network import EdgeNetwork
from env.multi_agent_caching_env import MultiAgentCachingEnv, agent_id
from env.rewards import cache_action_overlap_penalty


def test_cache_action_overlap_counts_neighbors_holding_target():
    config = load_config()
    config["num_nodes"] = 3
    config["num_clusters"] = 1
    network = EdgeNetwork(config)
    network.nodes[1].cache_container(1)
    network.nodes[2].cache_container(1)
    network.nodes[2].cache_container(3)

    pen, count = cache_action_overlap_penalty(
        network, 0, 1, [1, 2], weight=0.5
    )
    assert count == 2
    assert pen == -1.0

    pen3, count3 = cache_action_overlap_penalty(
        network, 0, 3, [1, 2], weight=0.5
    )
    assert count3 == 1
    assert pen3 == -0.5


def test_cache_action_overlap_zero_weight():
    config = load_config()
    network = EdgeNetwork(config)
    network.nodes[1].cache_container(0)
    pen, count = cache_action_overlap_penalty(
        network, 0, 0, [1], weight=0.0
    )
    assert pen == 0.0
    assert count == 0


def test_env_applies_overlap_once_on_insert():
    config = load_config()
    config["num_nodes"] = 3
    config["num_clusters"] = 1
    config["comm_level"] = 1
    config["overlap_penalty_weight"] = 0.5
    config["episode_length"] = 5
    config["traffic_pattern"] = "stationary"

    env = MultiAgentCachingEnv(config, seed=42)
    env.reset(seed=42)

    for nid in (1, 2):
        env.network.nodes[nid].cache_container(0)
    env._pending_requests = {0: 0, 1: 1, 2: 2}

    _, rewards, _, _, infos = env.step(
        {agent_id(i): env.reject_action for i in range(3)}
    )

    assert infos[agent_id(0)]["overlap_count"] == 2
    assert infos[agent_id(0)]["overlap_penalty"] == -1.0
    assert (
        infos[agent_id(0)]["task_reward"] + infos[agent_id(0)]["overlap_penalty"]
        == rewards[agent_id(0)]
    )


def test_env_no_overlap_when_already_cached():
    config = load_config()
    config["num_nodes"] = 2
    config["num_clusters"] = 1
    config["overlap_penalty_weight"] = 0.5
    config["episode_length"] = 5
    config["traffic_pattern"] = "stationary"

    env = MultiAgentCachingEnv(config, seed=0)
    env.reset(seed=0)
    env.network.nodes[0].cache_container(0)
    env.network.nodes[1].cache_container(0)
    env._pending_requests = {0: 0, 1: 0}

    _, _, _, _, infos = env.step(
        {agent_id(0): env.reject_action, agent_id(1): env.reject_action}
    )
    assert infos[agent_id(0)]["overlap_penalty"] == 0.0
    assert infos[agent_id(1)]["overlap_penalty"] == 0.0


def test_env_no_overlap_on_reject_when_full():
    config = load_config()
    config["num_nodes"] = 2
    config["num_clusters"] = 1
    config["overlap_penalty_weight"] = 0.5
    config["episode_length"] = 5
    config["traffic_pattern"] = "stationary"

    env = MultiAgentCachingEnv(config, seed=0)
    env.reset(seed=0)
    cap = env.cache_capacity
    for i in range(cap):
        env.network.nodes[0].cache_container(i)
        env.network.nodes[1].cache_container(i)
    pending = cap + 1
    env.network.nodes[1].cache_container(pending)  # neighbor already holds it
    env._pending_requests = {0: pending, 1: 0}

    _, _, _, _, infos = env.step(
        {agent_id(0): env.reject_action, agent_id(1): env.reject_action}
    )
    assert infos[agent_id(0)]["overlap_penalty"] == 0.0
    assert not env.network.nodes[0].is_cached(pending)
