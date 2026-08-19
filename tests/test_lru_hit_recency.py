import pytest

from agents.baselines import LRUPolicy
from configs import load_config
from env.caching_env import CachingEnv


def test_lru_keeps_hot_items_after_hits(config=None):
    config = load_config()
    env = CachingEnv(config, seed=42)
    policy = LRUPolicy()
    obs, _ = env.reset(seed=42)
    node = env.network.nodes[0]

    for _ in range(config["episode_length"]):
        action = policy.act(
            obs,
            env._pending,
            cache=node.cache,
            cache_capacity=env.cache_capacity,
            num_container_types=env.num_container_types,
        )
        obs, _, _, truncated, _ = env.step(action)
        if truncated:
            break

    assert node.hits + node.misses + node.forwards == config["episode_length"]
    total_local = node.hits + node.misses
    assert total_local > 0
    assert node.hits / total_local >= 0.45


def test_hit_moves_requested_container_to_mru():
    config = load_config()
    env = CachingEnv(config, seed=42)
    env.reset(seed=42)
    node = env.network.nodes[0]

    for container_id in [17, 12, 16, 14, 18]:
        node.cache_container(container_id)

    env._pending = 18
    env.step(env.reject_action)
    assert node.cache[-1] == 18
    assert 18 not in node.cache[:-1]
