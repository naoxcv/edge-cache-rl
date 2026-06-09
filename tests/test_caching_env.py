import numpy as np
import pytest

from configs import load_config
from env.caching_env import CachingEnv


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def env(config):
    return CachingEnv(config, seed=42)


def test_reset_returns_correct_shape(env, config):
    obs, info = env.reset()

    k = config["num_container_types"]
    assert obs.shape == (2 * k + 1,)
    assert env.observation_space.contains(obs)
    assert info["timestep"] == 0
    assert info["cache_hit_rate"] == 0.0


def test_step_noop_returns_valid_output(env, config):
    env.reset()
    noop = 2 * config["num_container_types"]

    obs, reward, terminated, truncated, info = env.step(noop)

    assert obs.shape == (2 * config["num_container_types"] + 1,)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert terminated is False
    assert isinstance(truncated, bool)
    assert 0.0 <= info["cache_hit_rate"] <= 1.0
    assert info["timestep"] == 1


def test_cache_then_request_local_hit(env, config, monkeypatch):
    env.reset()
    container_id = 5

    def fixed_requests():
        env.request_generator.timestep += 1
        return [container_id] * config["num_nodes"]

    monkeypatch.setattr(env.request_generator, "generate", fixed_requests)

    _, reward, _, _, info = env.step(container_id)

    assert reward == config["reward_local_hit"]
    assert env.network.nodes[0].hits == 1
    assert info["cache_hit_rate"] == 1.0


def test_uncached_no_neighbor_cloud_pull(env, config, monkeypatch):
    env.reset()
    container_id = 7

    def fixed_requests():
        env.request_generator.timestep += 1
        return [container_id] * config["num_nodes"]

    monkeypatch.setattr(env.request_generator, "generate", fixed_requests)

    noop = 2 * config["num_container_types"]
    _, reward, _, _, info = env.step(noop)

    assert reward == config["reward_cloud_pull"]
    assert env.network.nodes[0].misses == 1
    assert info["cache_hit_rate"] == 0.0


def test_forward_hit_reward(env, config, monkeypatch):
    env.reset()
    container_id = 9
    neighbor = env.network.get_neighbors(0)[0]
    env.network.nodes[neighbor].cache_container(container_id)

    def fixed_requests():
        env.request_generator.timestep += 1
        return [container_id] * config["num_nodes"]

    monkeypatch.setattr(env.request_generator, "generate", fixed_requests)

    noop = 2 * config["num_container_types"]
    _, reward, _, _, _ = env.step(noop)

    assert reward == config["reward_forward_hit"]
    assert env.network.nodes[0].forwards == 1


def test_episode_truncates_at_episode_length(config):
    short_config = {**config, "episode_length": 3}
    env = CachingEnv(short_config, seed=42)
    env.reset()

    noop = 2 * config["num_container_types"]
    for step in range(1, 3):
        _, _, terminated, truncated, _ = env.step(noop)
        assert terminated is False
        assert truncated is False
        assert env.timestep == step

    _, _, terminated, truncated, info = env.step(noop)
    assert terminated is False
    assert truncated is True
    assert info["timestep"] == 3


def test_cache_hit_rate_in_info(env, config):
    env.reset()
    noop = 2 * config["num_container_types"]

    for _ in range(20):
        _, _, _, _, info = env.step(noop)
        assert 0.0 <= info["cache_hit_rate"] <= 1.0
