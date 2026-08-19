from __future__ import annotations

import numpy as np
import pytest

from configs import load_config
from env.multi_agent_caching_env import (
    MultiAgentCachingEnv,
    agent_id,
    local_obs_size,
)


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
    c = small_config["cache_capacity"]
    local = local_obs_size(k, c)
    for aid in obs:
        assert obs[aid].shape == (local,)
        assert infos[aid]["requested"] is not None


def test_action_space_is_capacity_plus_reject(small_config):
    env = MultiAgentCachingEnv(small_config, seed=42)
    assert env.action_space.n == small_config["cache_capacity"] + 1
    assert env.reject_action == small_config["cache_capacity"]


def test_step_requires_all_agent_actions(small_config):
    env = MultiAgentCachingEnv(small_config, seed=42)
    obs, _ = env.reset(seed=42)
    actions = {aid: env.reject_action for aid in obs}

    obs2, rewards, terminateds, truncateds, infos = env.step(actions)

    assert set(rewards) == {"0", "1", "2"}
    assert terminateds["__all__"] is False
    assert truncateds["__all__"] is False
    assert all(isinstance(rewards[a], float) for a in rewards)


def test_non_full_miss_auto_inserts(small_config):
    env = MultiAgentCachingEnv(small_config, seed=42)
    env.reset(seed=42)
    env._pending_requests = {0: 4, 1: 4, 2: 4}
    env.step({agent_id(i): env.reject_action for i in range(3)})
    for i in range(3):
        assert env.network.nodes[i].is_cached(4)


def test_full_cache_reject_does_not_insert(small_config):
    env = MultiAgentCachingEnv(small_config, seed=42)
    env.reset(seed=42)
    cap = env.cache_capacity
    for i in range(cap):
        env.network.nodes[0].cache_container(i)
    env._pending_requests = {0: cap + 1, 1: 0, 2: 0}
    env.step({agent_id(i): env.reject_action for i in range(3)})
    assert not env.network.nodes[0].is_cached(cap + 1)
    assert env.network.nodes[0].cache == list(range(cap))


def test_full_cache_evicts_chosen_slot(small_config):
    env = MultiAgentCachingEnv(small_config, seed=42)
    env.reset(seed=42)
    cap = env.cache_capacity
    for i in range(cap):
        env.network.nodes[0].cache_container(i)
    victim = env.network.nodes[0].cache[2]
    pending = cap + 3
    env._pending_requests = {0: pending, 1: 0, 2: 0}
    env.step({agent_id(0): 2, agent_id(1): env.reject_action, agent_id(2): env.reject_action})
    assert env.network.nodes[0].is_cached(pending)
    assert not env.network.nodes[0].is_cached(victim)


def test_episode_truncates(small_config):
    small_config["episode_length"] = 3
    env = MultiAgentCachingEnv(small_config, seed=42)
    obs, _ = env.reset(seed=42)

    for _ in range(2):
        obs, _, terminateds, truncateds, _ = env.step(
            {a: env.reject_action for a in obs}
        )
        assert truncateds["__all__"] is False

    obs, _, terminateds, truncateds, _ = env.step(
        {a: env.reject_action for a in obs}
    )
    assert truncateds["__all__"] is True
    assert env.agents == []


def test_same_cluster_forward_hit(small_config):
    env = MultiAgentCachingEnv(small_config, seed=42)
    env.reset(seed=42)
    container_id = 4
    env.network.nodes[1].cache_container(container_id)
    env._pending_requests = {i: container_id for i in range(3)}
    _, rewards, _, _, _ = env.step(
        {agent_id(i): env.reject_action for i in range(3)}
    )

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
    assert stats["n_actions"] == small_config["cache_capacity"] + 1
