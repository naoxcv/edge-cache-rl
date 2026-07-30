from configs import load_config
from env.caching_env import CachingEnv
from env.wrappers import RandomTrafficSeedWrapper


def test_random_traffic_seed_wrapper_changes_each_reset():
    config = load_config()
    env = RandomTrafficSeedWrapper(CachingEnv(config, seed=0), base_seed=42, seed_range=1000)

    _, info1 = env.reset()
    _, info2 = env.reset()
    _, info3 = env.reset()

    seeds = {info1["traffic_seed"], info2["traffic_seed"], info3["traffic_seed"]}
    assert len(seeds) > 1
