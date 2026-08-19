from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.baselines import LFUPolicy, LRUPolicy
from configs import load_config
from env.caching_env import CachingEnv
from run_paths import resolve_model_path


def reactive_baseline_step(
    env: CachingEnv, policy, obs: np.ndarray
) -> tuple[np.ndarray, float, bool]:
    """Score pending request, then admit via eviction-only action."""
    node = env.network.nodes[env.active_node]
    action = policy.act(
        obs,
        env._pending,
        cache=node.cache,
        cache_capacity=env.cache_capacity,
        num_container_types=env.num_container_types,
    )
    obs, reward, _terminated, truncated, _info = env.step(action)
    return obs, reward, truncated


def evaluate_baseline(
    policy_name: str,
    policy,
    config: dict,
    *,
    num_episodes: int,
    seed: int = 42,
) -> dict:
    env = CachingEnv(config, seed=seed)
    obs, _ = env.reset(seed=seed)

    episode_returns: list[float] = []
    episode_return = 0.0

    for _ in range(num_episodes * config["episode_length"]):
        obs, reward, truncated = reactive_baseline_step(env, policy, obs)
        episode_return += reward
        if truncated:
            episode_returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()

    return {
        "policy": policy_name,
        "ep_rew_mean": float(np.mean(episode_returns)),
        "ep_rew_std": float(np.std(episode_returns)),
        "num_episodes": len(episode_returns),
    }


def evaluate_dqn(
    model_path: str,
    config: dict,
    *,
    num_episodes: int,
    seed: int = 42,
) -> dict:
    from stable_baselines3 import DQN

    env = CachingEnv(config, seed=seed)
    model = DQN.load(model_path, env=env)
    obs, _ = env.reset(seed=seed)

    episode_returns: list[float] = []
    episode_return = 0.0

    for _ in range(num_episodes * config["episode_length"]):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, truncated, _ = env.step(int(action))
        episode_return += reward
        if truncated:
            episode_returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()

    return {
        "policy": "DQN",
        "ep_rew_mean": float(np.mean(episode_returns)),
        "ep_rew_std": float(np.std(episode_returns)),
        "num_episodes": len(episode_returns),
    }


def print_result(result: dict, config: dict) -> None:
    ep_len = config["episode_length"]
    mean = result["ep_rew_mean"]
    per_step = mean / ep_len
    print(
        f"{result['policy']:>4}  ep_rew_mean={mean:7.1f}  "
        f"±{result['ep_rew_std']:5.1f}  "
        f"per_step={per_step:+.3f}  "
        f"({result['num_episodes']} episodes × {ep_len} steps)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare episode returns: LRU, LFU, DQN (same episode length)"
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated eval seeds, e.g. 42,0,7 (overrides --seed)",
    )
    parser.add_argument(
        "--dqn-path",
        type=str,
        default="dqn_generalized",
        help="Run name, run dir, or model path (prefers best_model.zip)",
    )
    parser.add_argument("--no-dqn", action="store_true")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default="results", help="Base output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    ep_len = config["episode_length"]
    seeds = (
        [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if args.seeds
        else [args.seed]
    )

    print("Episode return comparison")
    print(f"  episode_length={ep_len}  episodes={args.episodes}  seeds={seeds}")
    print("  LRU/LFU: reactive oracle (score request, then cache)")
    print("  DQN:     act-then-request (same as training)")
    print()

    model_path = None
    if not args.no_dqn:
        model_path = resolve_model_path(args.dqn_path)
        if model_path is None:
            print(f"DQN model not found at {args.dqn_path} (tried .zip too)\n")
        else:
            print(f"Loading DQN from {model_path}\n")

    dqn_vs_lfu: list[float] = []
    for eval_seed in seeds:
        print(f"--- seed={eval_seed} ---")
        results = [
            evaluate_baseline("LRU", LRUPolicy(), config, num_episodes=args.episodes, seed=eval_seed),
            evaluate_baseline("LFU", LFUPolicy(), config, num_episodes=args.episodes, seed=eval_seed),
        ]
        if model_path is not None:
            results.append(
                evaluate_dqn(
                    str(model_path),
                    config,
                    num_episodes=args.episodes,
                    seed=eval_seed,
                )
            )

        for result in results:
            print_result(result, config)

        dqn_mean = next((r["ep_rew_mean"] for r in results if r["policy"] == "DQN"), None)
        if dqn_mean is not None:
            for baseline in ("LRU", "LFU"):
                base_mean = next(r["ep_rew_mean"] for r in results if r["policy"] == baseline)
                delta = dqn_mean - base_mean
                print(f"DQN vs {baseline}: {delta:+.1f} episode return")
            dqn_vs_lfu.append(dqn_mean - next(r["ep_rew_mean"] for r in results if r["policy"] == "LFU"))
        print()

    if len(dqn_vs_lfu) > 1:
        print(
            f"DQN vs LFU across seeds: mean={np.mean(dqn_vs_lfu):+.1f}  "
            f"std={np.std(dqn_vs_lfu):.1f}"
        )


if __name__ == "__main__":
    main()
