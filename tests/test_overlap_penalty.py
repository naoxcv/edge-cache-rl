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

    # Caching container 1: both neighbors already hold it.
    pen, count = cache_action_overlap_penalty(
        network, 0, 1, [1, 2], weight=0.5
    )
    assert count == 2
    assert pen == -1.0

    # Caching container 3: only neighbor 2 holds it.
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


def test_env_applies_overlap_once_on_cache_action():
    config = load_config()
    config["num_nodes"] = 3
    config["num_clusters"] = 1
    config["comm_level"] = 1
    config["overlap_penalty_weight"] = 0.5
    config["episode_length"] = 5
    config["traffic_pattern"] = "stationary"

    env = MultiAgentCachingEnv(config, seed=42)
    env.reset(seed=42)

    # Pre-seed neighbor caches so a simultaneous cache of 0 is penalized.
    for nid in (1, 2):
        env.network.nodes[nid].cache_container(0)

    actions = {agent_id(0): 0, agent_id(1): env.num_container_types * 2, agent_id(2): env.num_container_types * 2}
    # node 0 caches 0; nodes 1/2 no-op (action = 2K)
    _, rewards, _, _, infos = env.step(actions)

    assert infos[agent_id(0)]["overlap_count"] == 2
    assert infos[agent_id(0)]["overlap_penalty"] == -1.0
    assert infos[agent_id(0)]["task_reward"] + infos[agent_id(0)]["overlap_penalty"] == rewards[agent_id(0)]
    assert infos[agent_id(1)]["overlap_penalty"] == 0.0
    assert infos[agent_id(2)]["overlap_penalty"] == 0.0


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

    # Re-selecting already-cached container 0 must not re-tax.
    _, _, _, _, infos = env.step({agent_id(0): 0, agent_id(1): 0})
    assert infos[agent_id(0)]["overlap_penalty"] == 0.0
    assert infos[agent_id(1)]["overlap_penalty"] == 0.0


def test_env_no_overlap_on_noop_or_evict():
    config = load_config()
    config["num_nodes"] = 2
    config["num_clusters"] = 1
    config["overlap_penalty_weight"] = 0.5
    config["episode_length"] = 5
    config["traffic_pattern"] = "stationary"

    env = MultiAgentCachingEnv(config, seed=0)
    env.reset(seed=0)
    env.network.nodes[0].cache_container(1)
    env.network.nodes[1].cache_container(1)

    noop = env.num_container_types * 2
    evict = env.num_container_types + 1  # evict container 1
    _, _, _, _, infos = env.step({agent_id(0): noop, agent_id(1): evict})
    assert infos[agent_id(0)]["overlap_penalty"] == 0.0
    assert infos[agent_id(1)]["overlap_penalty"] == 0.0
