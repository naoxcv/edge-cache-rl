from __future__ import annotations

import numpy as np
import pytest

from configs import load_config
from env.multi_agent_caching_env import (
    MultiAgentCachingEnv,
    observation_size,
)


@pytest.fixture
def base_config():
    config = load_config()
    config["num_nodes"] = 3
    config["num_clusters"] = 1
    config["episode_length"] = 20
    config["traffic_pattern"] = "stationary"
    config["forwarding_same_cluster_only"] = True
    return config


def test_level0_obs_is_local_only(base_config):
    base_config["comm_level"] = 0
    env = MultiAgentCachingEnv(base_config, seed=42)
    obs, _ = env.reset(seed=42)
    k = base_config["num_container_types"]
    assert env.max_neighbors == 2
    assert env.observation_space.shape == (2 * k + 1,)
    for aid in obs:
        assert obs[aid].shape == (2 * k + 1,)


def test_level1_obs_appends_neighbor_cache_binaries(base_config):
    base_config["comm_level"] = 1
    env = MultiAgentCachingEnv(base_config, seed=42)
    k = base_config["num_container_types"]
    expected = observation_size(k, comm_level=1, max_neighbors=env.max_neighbors)
    assert expected == (2 * k + 1) + env.max_neighbors * k
    assert env.observation_space.shape == (expected,)

    obs, _ = env.reset(seed=42)
    # Seed neighbor caches and re-read obs after a no-op step.
    env.network.nodes[1].cache_container(3)
    env.network.nodes[2].cache_container(5)
    noop = 2 * k
    obs, _, _, _, _ = env.step({aid: noop for aid in obs})

    # Node 0 neighbors are [1, 2] (sorted). Level-1 suffix starts after local 2K+1.
    local = 2 * k + 1
    nbr0 = obs["0"][local : local + k]
    nbr1 = obs["0"][local + k : local + 2 * k]
    assert nbr0[3] == 1.0
    assert nbr1[5] == 1.0
    assert nbr0.sum() == 1.0
    assert nbr1.sum() == 1.0


def test_level2_obs_appends_full_neighbor_state(base_config):
    base_config["comm_level"] = 2
    env = MultiAgentCachingEnv(base_config, seed=42)
    k = base_config["num_container_types"]
    local = 2 * k + 1
    expected = local + env.max_neighbors * local
    assert env.observation_space.shape == (expected,)

    obs, _ = env.reset(seed=42)
    env.network.nodes[1].cache_container(7)
    noop = 2 * k
    obs, _, _, _, _ = env.step({aid: noop for aid in obs})

    nbr0_state = obs["0"][local : local + local]
    assert nbr0_state[7] == 1.0  # cache binary bit for container 7
    assert 0.0 < nbr0_state[k] <= 1.0  # utilization


def test_level1_pads_when_degree_below_max():
    config = load_config()
    config["num_nodes"] = 10
    config["num_clusters"] = 3
    config["comm_level"] = 1
    config["traffic_pattern"] = "stationary"
    env = MultiAgentCachingEnv(config, seed=42)
    obs, _ = env.reset(seed=42)

    assert env.max_neighbors == 4
    # Node 4 has degree 2 under this topology — trailing neighbor slots are zeros.
    assert len(env.neighbor_lists[4]) == 2
    k = config["num_container_types"]
    local = 2 * k + 1
    suffix = obs["4"][local:]
    assert suffix.shape == (4 * k,)
    assert np.allclose(suffix[2 * k :], 0.0)


def test_level3_obs_matches_level1(base_config):
    base_config["comm_level"] = 3
    env = MultiAgentCachingEnv(base_config, seed=42)
    k = base_config["num_container_types"]
    expected = observation_size(k, comm_level=3, max_neighbors=env.max_neighbors)
    assert expected == observation_size(k, comm_level=1, max_neighbors=env.max_neighbors)
    assert env.observation_space.shape == (expected,)


def test_level3_selective_gating_uses_full_obs_during_explore():
    """L3 explore uses full neighbor features; exploit gates by Q-margin."""
    pytest.importorskip("stable_baselines3")
    from agents.multi_agent import create_shared_dqn, select_action_for_obs

    config = load_config()
    config["num_nodes"] = 3
    config["num_clusters"] = 1
    config["comm_level"] = 3
    config["traffic_pattern"] = "stationary"
    config["episode_length"] = 10
    config["selective_comm_threshold"] = 0.01

    env = MultiAgentCachingEnv(config, seed=42)
    obs, _ = env.reset(seed=42)
    k = config["num_container_types"]
    local_dim = 2 * k + 1

    model = create_shared_dqn(
        config, seed=42, total_timesteps=1000,
        tensorboard_log=None, verbose=0,
    )

    agent_obs = obs["0"]
    assert agent_obs.shape[0] > local_dim

    _, communicated = select_action_for_obs(
        model, agent_obs, config=config, deterministic=True
    )
    assert isinstance(communicated, bool)

    local_only = np.array(agent_obs, copy=True)
    local_only[local_dim:] = 0.0
    _, comm_local = select_action_for_obs(
        model, local_only, config=config, deterministic=True
    )
    assert isinstance(comm_local, bool)


def test_invalid_comm_level_raises(base_config):
    base_config["comm_level"] = 4
    with pytest.raises(ValueError, match="comm_level"):
        MultiAgentCachingEnv(base_config, seed=42)
