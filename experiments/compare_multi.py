from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.baselines import LFUPolicy, LRUPolicy
from agents.multi_agent import (
    evaluate_sb3_dqn,
    reactive_multi_baseline_step,
    resolve_multi_model_path,
)
from configs import load_config
from env.multi_agent_caching_env import MultiAgentCachingEnv


def evaluate_baselines(
    policy_factory,
    policy_name: str,
    config: dict,
    *,
    num_episodes: int,
    seed: int,
) -> dict:
    env = MultiAgentCachingEnv(config, seed=seed)
    obs, _ = env.reset(seed=seed)
    policies = {i: policy_factory() for i in range(env.num_nodes)}

    episode_returns: list[float] = []
    episode_return = 0.0
    episodes_done = 0

    while episodes_done < num_episodes:
        obs, rewards, truncated = reactive_multi_baseline_step(env, policies, obs)
        episode_return += float(sum(rewards.values()))
        if truncated:
            episode_returns.append(episode_return)
            episode_return = 0.0
            episodes_done += 1
            if episodes_done < num_episodes:
                obs, _ = env.reset()
                policies = {i: policy_factory() for i in range(env.num_nodes)}

    stats = env.network_stats()
    return {
        "policy": policy_name,
        "ep_rew_mean": float(np.mean(episode_returns)),
        "ep_rew_std": float(np.std(episode_returns)),
        "num_episodes": len(episode_returns),
        **stats,
    }


def print_result(result: dict, config: dict) -> None:
    print(
        f"{result['policy']:>4}  ep_rew_mean={result['ep_rew_mean']:8.1f}  "
        f"±{result['ep_rew_std']:5.1f}  "
        f"hit={result['hit_rate']:.1%}  "
        f"fwd={result['forward_rate']:.1%}  "
        f"cloud={result['cloud_rate']:.1%}  "
        f"({result['num_episodes']} eps × {config['episode_length']} steps × "
        f"{config['num_nodes']} nodes)"
    )
    caches = list(result["caches"].values())
    unique = {tuple(sorted(c)) for c in caches}
    print(
        f"     cache diversity: {len(unique)} unique cache sets / "
        f"{config['num_nodes']} nodes"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multi-node LRU/LFU/DQN (SB3 shared policy)"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds, e.g. 42,0,7",
    )
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--clusters", type=int, default=None)
    parser.add_argument("--traffic", type=str, default=None)
    parser.add_argument("--locality-factor", type=float, default=None)
    parser.add_argument(
        "--dqn-path",
        type=str,
        default=None,
        help="Run name or model path (prefers best_model.zip)",
    )
    parser.add_argument("--no-dqn", action="store_true")
    parser.add_argument(
        "--no-forwarding",
        action="store_true",
        help="Disable neighbor forwarding (misses go to cloud)",
    )
    parser.add_argument(
        "--comm-level",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Communication level for DQN obs (baselines unchanged)",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default="results", help="Base output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.nodes is not None:
        config["num_nodes"] = args.nodes
    if args.clusters is not None:
        config["num_clusters"] = args.clusters
    if args.traffic is not None:
        config["traffic_pattern"] = args.traffic
    if args.locality_factor is not None:
        config["locality_factor"] = args.locality_factor
    config["forwarding_same_cluster_only"] = True
    config["enable_forwarding"] = not args.no_forwarding
    config["comm_level"] = args.comm_level

    seeds = (
        [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if args.seeds
        else [args.seed]
    )

    fwd_label = "ON" if config["enable_forwarding"] else "OFF"
    print("Multi-node policy comparison")
    print(
        f"  nodes={config['num_nodes']} clusters={config['num_clusters']} "
        f"traffic={config['traffic_pattern']} "
        f"locality={config.get('locality_factor', 0.0)} forwarding={fwd_label} "
        f"comm_level={config['comm_level']} "
        f"episodes={args.episodes} seeds={seeds}"
    )
    print("  LRU/LFU: reactive oracle per node; same-cluster forwarding when ON")
    print("  DQN:     act-then-request shared SB3 policy")
    print()

    model_path = None
    if not args.no_dqn and args.dqn_path:
        model_path = resolve_multi_model_path(args.dqn_path, prefer_best=True)
        if model_path is None:
            print(f"DQN model not found for {args.dqn_path}\n")
        else:
            print(f"Loading DQN from {model_path}\n")

    dqn_vs_lfu: list[float] = []
    for eval_seed in seeds:
        print(f"--- seed={eval_seed} ---")
        results = [
            evaluate_baselines(
                LRUPolicy, "LRU", config, num_episodes=args.episodes, seed=eval_seed
            ),
            evaluate_baselines(
                LFUPolicy, "LFU", config, num_episodes=args.episodes, seed=eval_seed
            ),
        ]
        if model_path is not None:
            results.append(
                evaluate_sb3_dqn(
                    model_path,
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
                base_mean = next(
                    r["ep_rew_mean"] for r in results if r["policy"] == baseline
                )
                print(f"DQN vs {baseline}: {dqn_mean - base_mean:+.1f} episode return")
            dqn_vs_lfu.append(
                dqn_mean
                - next(r["ep_rew_mean"] for r in results if r["policy"] == "LFU")
            )
        print()

    if len(dqn_vs_lfu) > 1:
        print(
            f"DQN vs LFU across seeds: mean={np.mean(dqn_vs_lfu):+.1f}  "
            f"std={np.std(dqn_vs_lfu):.1f}"
        )


if __name__ == "__main__":
    main()
