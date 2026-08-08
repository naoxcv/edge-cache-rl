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
    diversity = result.get("cache_diversity")
    if diversity is None:
        caches = list(result["caches"].values())
        diversity = len({tuple(sorted(c)) for c in caches})
    comms = result.get("comm_events_mean")
    comm_str = f"  comms={comms:.0f}" if comms is not None else ""
    print(
        f"{result['policy']:>8}  ep_rew_mean={result['ep_rew_mean']:8.1f}  "
        f"±{result['ep_rew_std']:5.1f}  "
        f"hit={result['hit_rate']:.1%}  "
        f"fwd={result['forward_rate']:.1%}  "
        f"cloud={result['cloud_rate']:.1%}  "
        f"div={diversity}/{config['num_nodes']}"
        f"{comm_str}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multi-node LRU/LFU/DQN across communication levels"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--traffic", type=str, default="shifting")
    parser.add_argument("--locality-factor", type=float, default=0.3)
    parser.add_argument(
        "--levels",
        type=str,
        default="0,1,2",
        help="Comma-separated comm levels to evaluate, e.g. 0,1,2",
    )
    parser.add_argument(
        "--dqn-paths",
        type=str,
        default="dqn_multi_level0,dqn_multi_level1,dqn_multi_level2",
        help="Comma-separated run names/paths aligned with --levels",
    )
    parser.add_argument("--no-dqn", action="store_true")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--no-forwarding", action="store_true")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default="results", help="Base output directory")
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    dqn_paths = [x.strip() for x in args.dqn_paths.split(",") if x.strip()]
    if not args.no_dqn and len(dqn_paths) != len(levels):
        raise SystemExit("--dqn-paths must have the same length as --levels")

    seeds = (
        [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if args.seeds
        else [args.seed]
    )

    base = load_config(args.config)
    base["num_nodes"] = args.nodes
    base["num_clusters"] = args.clusters
    base["traffic_pattern"] = args.traffic
    base["locality_factor"] = args.locality_factor
    base["forwarding_same_cluster_only"] = True
    base["enable_forwarding"] = not args.no_forwarding

    print("Communication-level comparison")
    print(
        f"  nodes={args.nodes} clusters={args.clusters} traffic={args.traffic} "
        f"locality={args.locality_factor} "
        f"forwarding={'ON' if base['enable_forwarding'] else 'OFF'} "
        f"episodes={args.episodes} seeds={seeds} levels={levels}"
    )
    print()

    resolved: dict[int, Path | None] = {}
    if not args.no_dqn:
        for level, path_arg in zip(levels, dqn_paths):
            resolved[level] = resolve_multi_model_path(path_arg, prefer_best=True)
            label = resolved[level] if resolved[level] else "MISSING"
            print(f"  Level {level}: {path_arg} -> {label}")
        print()

    # Aggregate across seeds: policy_key -> list of metrics
    agg: dict[str, list[dict]] = {}

    for eval_seed in seeds:
        print(f"--- seed={eval_seed} ---")
        # Baselines once per seed (comm_level does not affect reactive baselines).
        config0 = {**base, "comm_level": 0}
        if not args.no_baselines:
            for factory, name in ((LRUPolicy, "LRU"), (LFUPolicy, "LFU")):
                result = evaluate_baselines(
                    factory, name, config0, num_episodes=args.episodes, seed=eval_seed
                )
                print_result(result, config0)
                agg.setdefault(name, []).append(result)

        if not args.no_dqn:
            for level in levels:
                path = resolved.get(level)
                if path is None:
                    print(f"  L{level}-DQN  SKIP (model not found)")
                    continue
                cfg = {**base, "comm_level": level}
                result = evaluate_sb3_dqn(
                    path, cfg, num_episodes=args.episodes, seed=eval_seed
                )
                result["policy"] = f"L{level}-DQN"
                print_result(result, cfg)
                agg.setdefault(result["policy"], []).append(result)
        print()

    print("=== Mean across seeds ===")
    for name, rows in agg.items():
        rets = [r["ep_rew_mean"] for r in rows]
        hits = [r["hit_rate"] for r in rows]
        fwds = [r["forward_rate"] for r in rows]
        clouds = [r["cloud_rate"] for r in rows]
        divs = [r.get("cache_diversity", 0) for r in rows]
        comms = [r.get("comm_events_mean") for r in rows]
        has_comms = any(c is not None for c in comms)
        comm_str = ""
        if has_comms:
            vals = [0.0 if c is None else float(c) for c in comms]
            comm_str = f"  comms={np.mean(vals):.0f}"
        print(
            f"{name:>8}  ret={np.mean(rets):8.1f}±{np.std(rets):5.1f}  "
            f"hit={np.mean(hits):.1%}  fwd={np.mean(fwds):.1%}  "
            f"cloud={np.mean(clouds):.1%}  div={np.mean(divs):.1f}"
            f"{comm_str}"
        )

    if "L0-DQN" in agg and "L1-DQN" in agg:
        l0 = np.mean([r["ep_rew_mean"] for r in agg["L0-DQN"]])
        l1 = np.mean([r["ep_rew_mean"] for r in agg["L1-DQN"]])
        print(f"L1-DQN − L0-DQN: {l1 - l0:+.1f}")
    if "L1-DQN" in agg and "L3-DQN" in agg:
        l1 = np.mean([r["ep_rew_mean"] for r in agg["L1-DQN"]])
        l3 = np.mean([r["ep_rew_mean"] for r in agg["L3-DQN"]])
        l1_comms = np.mean(
            [float(r.get("comm_events_mean") or 0.0) for r in agg["L1-DQN"]]
        )
        l3_comms = np.mean(
            [float(r.get("comm_events_mean") or 0.0) for r in agg["L3-DQN"]]
        )
        # L1 always observes neighbors: N nodes × episode_length decisions.
        l1_always = float(base["num_nodes"] * base["episode_length"])
        print(
            f"L3-DQN − L1-DQN: {l3 - l1:+.1f}  "
            f"comms L3={l3_comms:.0f} vs L1-always-on={l1_always:.0f} "
            f"({100.0 * l3_comms / l1_always:.1f}% of always-on)"
        )

    if "LFU" in agg:
        lfu_ret = np.mean([r["ep_rew_mean"] for r in agg["LFU"]])
        for name, rows in agg.items():
            if name.startswith("L") and name.endswith("-DQN"):
                dqn_ret = np.mean([r["ep_rew_mean"] for r in rows])
                print(f"{name} vs LFU: {dqn_ret - lfu_ret:+.1f}")


if __name__ == "__main__":
    main()
