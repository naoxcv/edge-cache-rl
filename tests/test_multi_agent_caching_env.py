from __future__ import annotations

import numpy as np
import pytest

from configs import load_config
from env.multi_agent_caching_env import MultiAgentCachingEnv, agent_id


@pytest.fixture
def small_config():
    config = load_config()
    config["num_nodes"] = 3
    config["num_clusters"] = 1
    config["episode_length"] = 50
    config["traffic_pattern"] = "stationary"
    config["forwarding_same_cluster_only"] = True
    return config


def test_reset_returns_obs_for_all_agents(small_config):
    env = MultiAgentCachingEnv(small_config, seed=42)
    obs, infos = env.reset(seed=42)

    assert set(obs) == {"0", "1", "2"}
    assert set(infos) == {"0", "1", "2"}
    k = small_config["num_container_types"]
    for aid in obs:
        assert obs[aid].shape == (2 * k + 1,)


def test_step_requires_all_agent_actions(small_config):
    env = MultiAgentCachingEnv(small_config, seed=42)
    obs, _ = env.reset(seed=42)
    noop = 2 * small_config["num_container_types"]
    actions = {aid: noop for aid in obs}

    obs2, rewards, terminateds, truncateds, infos = env.step(actions)

    assert set(rewards) == {"0", "1", "2"}
    assert terminateds["__all__"] is False
    assert truncateds["__all__"] is False
    assert all(isinstance(rewards[a], float) for a in rewards)


def test_episode_truncates(small_config):
    small_config["episode_length"] = 3
    env = MultiAgentCachingEnv(small_config, seed=42)
    obs, _ = env.reset(seed=42)
    noop = 2 * small_config["num_container_types"]

    for _ in range(2):
        obs, _, terminateds, truncateds, _ = env.step({a: noop for a in obs})
        assert truncateds["__all__"] is False

    obs, _, terminateds, truncateds, _ = env.step({a: noop for a in obs})
    assert truncateds["__all__"] is True
    assert env.agents == []


def test_same_cluster_forward_hit(small_config, monkeypatch):
    env = MultiAgentCachingEnv(small_config, seed=42)
    env.reset(seed=42)
    container_id = 4
    env.network.nodes[1].cache_container(container_id)

    def fixed_requests():
        env.request_generator.timestep += 1
        return [container_id] * small_config["num_nodes"]

    monkeypatch.setattr(env.request_generator, "generate", fixed_requests)
    noop = 2 * small_config["num_container_types"]
    _, rewards, _, _, _ = env.step({agent_id(i): noop for i in range(3)})

    # Node 1 local hit; nodes 0 and 2 forward to node 1 (same cluster).
    assert rewards["1"] == small_config["reward_local_hit"]
    assert rewards["0"] == small_config["reward_forward_hit"]
    assert rewards["2"] == small_config["reward_forward_hit"]
    assert env.network.nodes[0].forwards == 1
    assert env.network.nodes[2].forwards == 1


def test_cross_cluster_forward_blocked_by_default():
    config = load_config()
    config["num_nodes"] = 4
    config["num_clusters"] = 2
    config["forwarding_same_cluster_only"] = True
    config["episode_length"] = 5

    env = MultiAgentCachingEnv(config, seed=42)
    env.reset(seed=42)

    # Nodes 0 and 1 are different clusters under round-robin (0%2=0, 1%2=1).
    assert env.network.cluster_for_node[0] != env.network.cluster_for_node[1]
    container_id = 7
    env.network.nodes[1].cache_container(container_id)

    reward = env._process_request(0, container_id)
    assert reward == config["reward_cloud_pull"]
    assert env.network.nodes[0].misses == 1


def test_random_rollout_produces_forward_hits(small_config):
    from agents.multi_agent import evaluate_random_policy

    small_config["episode_length"] = 200
    small_config["num_nodes"] = 3
    small_config["num_clusters"] = 1
    stats = evaluate_random_policy(small_config, num_episodes=2, seed=0)
    assert stats["forwards"] > 0
    assert 0.0 <= stats["forward_rate"] <= 1.0
