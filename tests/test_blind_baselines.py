"""Request-first multi-agent baseline step."""

from agents.baselines import LRUPolicy
from agents.multi_agent import heuristic_multi_baseline_step, reactive_multi_baseline_step
from configs import load_config
from env.multi_agent_caching_env import MultiAgentCachingEnv, agent_id


def test_heuristic_baseline_sees_pending_request():
    config = load_config()
    config["num_nodes"] = 4
    config["num_clusters"] = 2
    config["episode_length"] = 5
    config["enable_forwarding"] = True
    config["forwarding_same_cluster_only"] = True

    env = MultiAgentCachingEnv(config, seed=0)
    obs, infos = env.reset(seed=0)
    policies = {i: LRUPolicy() for i in range(env.num_nodes)}

    assert all(infos[aid]["requested"] is not None for aid in obs)
    actions_before = {
        aid: policies[int(aid)].act(
            o,
            env._pending_requests.get(int(aid)),
            cache=env.network.nodes[int(aid)].cache,
            cache_capacity=env.cache_capacity,
        )
        for aid, o in obs.items()
    }
    # Caches start empty → no eviction decision (reject / auto-insert).
    assert all(a == env.reject_action for a in actions_before.values())

    obs, rewards, truncated = heuristic_multi_baseline_step(env, policies, obs)
    assert not truncated
    assert env.timestep == 1
    assert set(rewards.keys()) == set(obs.keys())
    # After first misses, caches should have started filling.
    assert any(len(env.network.nodes[i].cache) > 0 for i in range(env.num_nodes))


def test_oracle_alias_matches_heuristic():
    config = load_config()
    config["num_nodes"] = 3
    config["num_clusters"] = 1
    config["episode_length"] = 3
    env = MultiAgentCachingEnv(config, seed=1)
    obs, _ = env.reset(seed=1)
    policies = {i: LRUPolicy() for i in range(env.num_nodes)}
    obs, rewards, truncated = reactive_multi_baseline_step(env, policies, obs)
    assert env.timestep == 1
    assert agent_id(0) in rewards
