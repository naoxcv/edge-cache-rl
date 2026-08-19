from __future__ import annotations

import numpy as np
import pytest

from configs import load_config
from env.multi_agent_caching_env import (
    MultiAgentCachingEnv,
    local_obs_size,
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


def _reject(env: MultiAgentCachingEnv) -> int:
    return env.reject_action


def test_level0_obs_is_local_only(base_config):
    base_config["comm_level"] = 0
    env = MultiAgentCachingEnv(base_config, seed=42)
    obs, _ = env.reset(seed=42)
    k = base_config["num_container_types"]
    c = base_config["cache_capacity"]
    local = local_obs_size(k, c)
    assert env.max_neighbors == 2
    assert env.observation_space.shape == (local,)
    assert env.action_space.n == c + 1
    for aid in obs:
        assert obs[aid].shape == (local,)


def test_level1_obs_appends_neighbor_cache_binaries(base_config):
    base_config["comm_level"] = 1
    env = MultiAgentCachingEnv(base_config, seed=42)
    k = base_config["num_container_types"]
    c = base_config["cache_capacity"]
    expected = observation_size(
        k, comm_level=1, max_neighbors=env.max_neighbors, cache_capacity=c
    )
    local = local_obs_size(k, c)
    assert expected == local + env.max_neighbors * k
    assert env.observation_space.shape == (expected,)

    obs, _ = env.reset(seed=42)
    env.network.nodes[1].cache_container(3)
    env.network.nodes[2].cache_container(5)
    obs, _, _, _, _ = env.step({aid: _reject(env) for aid in obs})

    nbr0 = obs["0"][local : local + k]
    nbr1 = obs["0"][local + k : local + 2 * k]
    assert nbr0[3] == 1.0
    assert nbr1[5] == 1.0
    assert nbr0.sum() >= 1.0
    assert nbr1.sum() >= 1.0


def test_level2_obs_appends_full_neighbor_state(base_config):
    base_config["comm_level"] = 2
    env = MultiAgentCachingEnv(base_config, seed=42)
    k = base_config["num_container_types"]
    c = base_config["cache_capacity"]
    local = local_obs_size(k, c)
    expected = local + env.max_neighbors * local
    assert env.observation_space.shape == (expected,)

    obs, _ = env.reset(seed=42)
    env.network.nodes[1].cache_container(7)
    obs, _, _, _, _ = env.step({aid: _reject(env) for aid in obs})

    nbr0_state = obs["0"][local : local + local]
    # Slot 0 of neighbor 1 should contain container 7 (and possibly later admits).
    assert nbr0_state[7] == 1.0 or nbr0_state[: c * k].reshape(c, k)[:, 7].sum() >= 1.0
    assert 0.0 < nbr0_state[c * k] <= 1.0  # utilization


def test_level1_pads_when_degree_below_max():
    config = load_config()
    config["num_nodes"] = 10
    config["num_clusters"] = 3
    config["comm_level"] = 1
    config["traffic_pattern"] = "stationary"
    config["forwarding_same_cluster_only"] = True
    env = MultiAgentCachingEnv(config, seed=42)
    obs, _ = env.reset(seed=42)

    assert env.max_neighbors == 3
    assert env.neighbor_lists[4] == [1, 7]
    assert 0 not in env.neighbor_lists[4]
    k = config["num_container_types"]
    c = config["cache_capacity"]
    local = local_obs_size(k, c)
    suffix = obs["4"][local:]
    assert suffix.shape == (3 * k,)
    assert np.allclose(suffix[2 * k :], 0.0)


def test_level1_excludes_inter_cluster_bridge_neighbors():
    """Obs neighbors match forwarding scope: same cluster only."""
    config = load_config()
    config["num_nodes"] = 10
    config["num_clusters"] = 3
    config["comm_level"] = 1
    config["forwarding_same_cluster_only"] = True
    env = MultiAgentCachingEnv(config, seed=42)

    assert 1 in env.network.get_neighbors(0)
    assert env.neighbor_lists[0] == [3, 6, 9]
    assert env.neighbor_lists[1] == [4, 7]
    for node_id, peers in env.neighbor_lists.items():
        home = env.network.cluster_for_node[node_id]
        for nbr in peers:
            assert env.network.cluster_for_node[nbr] == home


def test_level3_obs_matches_level1(base_config):
    base_config["comm_level"] = 3
    env = MultiAgentCachingEnv(base_config, seed=42)
    k = base_config["num_container_types"]
    c = base_config["cache_capacity"]
    expected = observation_size(
        k, comm_level=3, max_neighbors=env.max_neighbors, cache_capacity=c
    )
    assert expected == observation_size(
        k, comm_level=1, max_neighbors=env.max_neighbors, cache_capacity=c
    )
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
    c = config["cache_capacity"]
    local_dim = local_obs_size(k, c)

    model = create_shared_dqn(
        config, seed=42, total_timesteps=1000,
        tensorboard_log=None, verbose=0,
    )

    agent_obs = obs["0"]
    assert agent_obs.shape[0] > local_dim
    assert model.action_space.n == c + 1

    _, communicated = select_action_for_obs(
        model, agent_obs, config=config, deterministic=True
    )
    assert isinstance(communicated, bool)


def test_invalid_comm_level_raises(base_config):
    base_config["comm_level"] = 4
    with pytest.raises(ValueError, match="comm_level"):
        MultiAgentCachingEnv(base_config, seed=42)
