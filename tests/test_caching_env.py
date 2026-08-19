import numpy as np
import pytest

from configs import load_config
from env.caching_env import CachingEnv
from env.multi_agent_caching_env import local_obs_size


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def env(config):
    return CachingEnv(config, seed=42)


def test_reset_returns_correct_shape(env, config):
    obs, info = env.reset()

    k = config["num_container_types"]
    c = config["cache_capacity"]
    assert obs.shape == (local_obs_size(k, c),)
    assert env.observation_space.contains(obs)
    assert env.action_space.n == c + 1
    assert info["timestep"] == 0
    assert info["cache_hit_rate"] == 0.0
    assert info["requested"] is not None


def test_step_reject_returns_valid_output(env, config):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.reject_action)

    assert obs.shape == (local_obs_size(config["num_container_types"], config["cache_capacity"]),)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert terminated is False
    assert isinstance(truncated, bool)
    assert 0.0 <= info["cache_hit_rate"] <= 1.0
    assert info["timestep"] == 1


def test_pending_hit_after_insert(env, config):
    env.reset()
    container_id = 5
    env._pending = container_id
    env.step(env.reject_action)
    assert env.network.nodes[0].is_cached(container_id)

    env._pending = container_id
    _, reward, _, _, info = env.step(env.reject_action)
    assert reward == config["reward_local_hit"]
    assert env.network.nodes[0].hits >= 1
    assert info["cache_hit_rate"] > 0.0


def test_uncached_no_neighbor_cloud_pull(env, config):
    env.reset()
    # Empty other caches so a miss cannot forward.
    for node in env.network.nodes:
        node.cache.clear()
        node.cache_set.clear()
    env._pending = 7
    _, reward, _, _, info = env.step(env.reject_action)

    assert reward == config["reward_cloud_pull"]
    assert env.network.nodes[0].misses == 1


def test_forward_hit_reward(env, config):
    env.reset()
    container_id = 9
    neighbor = env.network.get_cluster_neighbors(0)[0]
    env.network.nodes[neighbor].cache_container(container_id)
    env._pending = container_id
    _, reward, _, _, _ = env.step(env.reject_action)

    assert reward == config["reward_forward_hit"]
    assert env.network.nodes[0].forwards == 1


def test_forwarding_disabled_forces_cloud_pull(config):
    config = {**config, "enable_forwarding": False}
    env = CachingEnv(config, seed=42)
    env.reset()
    container_id = 9
    neighbor = env.network.get_cluster_neighbors(0)[0]
    env.network.nodes[neighbor].cache_container(container_id)
    env._pending = container_id
    _, reward, _, _, _ = env.step(env.reject_action)

    assert reward == config["reward_cloud_pull"]
    assert env.network.nodes[0].forwards == 0
    assert env.network.nodes[0].misses == 1


def test_episode_truncates_at_episode_length(config):
    short_config = {**config, "episode_length": 3}
    env = CachingEnv(short_config, seed=42)
    env.reset()

    for step in range(1, 3):
        _, _, terminated, truncated, _ = env.step(env.reject_action)
        assert terminated is False
        assert truncated is False
        assert env.timestep == step

    _, _, terminated, truncated, info = env.step(env.reject_action)
    assert terminated is False
    assert truncated is True
    assert info["timestep"] == 3


def test_cache_hit_rate_in_info(env, config):
    env.reset()
    for _ in range(20):
        _, _, _, _, info = env.step(env.reject_action)
        assert 0.0 <= info["cache_hit_rate"] <= 1.0
