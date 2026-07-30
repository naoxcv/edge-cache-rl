import pytest

from agents.baselines import LRUPolicy
from configs import load_config
from env.caching_env import CachingEnv


@pytest.fixture
def config():
    return load_config()


def test_lru_keeps_hot_items_after_hits(config):
    env = CachingEnv(config, seed=42)
    policy = LRUPolicy()
    obs, _ = env.reset(seed=42)
    node = env.network.nodes[0]

    for _ in range(config["episode_length"]):
        requests = env.request_generator.generate()
        requested = requests[env.active_node]
        env._process_request(requested)
        if requested is not None:
            node.record_request(requested, env.observation_window)
        action = policy.act(obs, requested, cache=node.cache)
        env._apply_action(action)
        obs = env._get_observation()
        env.timestep += 1

    # Under Zipf α=1.0 / K=20 / C=5, correct LRU is near break-even (~49% hits).
    # Recency updates on hit must keep hit rate competitive (not thrashing).
    assert node.hits + node.misses == config["episode_length"]
    assert node.hits / (node.hits + node.misses) >= 0.45


def test_hit_moves_requested_container_to_mru():
    config = load_config()
    env = CachingEnv(config, seed=42)
    obs, _ = env.reset(seed=42)
    node = env.network.nodes[0]

    for container_id in [17, 12, 16, 14, 18]:
        node.cache_container(container_id)

    env._process_request(18)
    assert node.cache[-1] == 18
    assert 18 not in node.cache[:-1]
