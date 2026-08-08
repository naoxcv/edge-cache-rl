from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.multi_agent import evaluate_random_policy, train_multi_agent_dqn
from configs import load_config


def _apply_common(config: dict, args: argparse.Namespace) -> None:
    """Apply shared CLI overrides to a config dict."""
    if args.nodes is not None:
        config["num_nodes"] = args.nodes
    if args.clusters is not None:
        config["num_clusters"] = args.clusters
    config["traffic_pattern"] = args.traffic
    if hasattr(args, "locality_factor") and args.locality_factor is not None:
        config["locality_factor"] = float(args.locality_factor)
    if hasattr(args, "comm_level") and args.comm_level is not None:
        config["comm_level"] = int(args.comm_level)
    if hasattr(args, "overlap_penalty") and args.overlap_penalty is not None:
        config["overlap_penalty_weight"] = float(args.overlap_penalty)
    if hasattr(args, "comm_penalty") and args.comm_penalty is not None:
        config["comm_penalty_lambda"] = float(args.comm_penalty)
    if hasattr(args, "gradient_steps") and args.gradient_steps is not None:
        config["gradient_steps"] = int(args.gradient_steps)


def cmd_verify(args: argparse.Namespace) -> None:
    """Run random-action rollout and verify forwarding is nonzero."""
    config = load_config(args.config)
    _apply_common(config, args)
    config["forwarding_same_cluster_only"] = not args.allow_cross_cluster

    stats = evaluate_random_policy(
        config, num_episodes=args.episodes, seed=args.seed
    )
    print("Random-policy multi-node verification")
    print(
        f"  nodes={config['num_nodes']} clusters={config['num_clusters']} "
        f"comm_level={config.get('comm_level', 0)} obs_dim={stats.get('obs_dim')}"
    )
    print(f"  traffic={config['traffic_pattern']} episodes={args.episodes}")
    print(f"  ep_rew_mean={stats['ep_rew_mean']:.1f} +/- {stats['ep_rew_std']:.1f}")
    print(
        f"  network hit={stats['hit_rate']:.1%}  "
        f"forward={stats['forward_rate']:.1%}  "
        f"cloud={stats['cloud_rate']:.1%}"
    )
    print(
        f"  counts hits={stats['hits']} forwards={stats['forwards']} "
        f"misses={stats['misses']} diversity={stats.get('cache_diversity')}"
    )
    print("  per-node caches:")
    for node_id, cache in stats["caches"].items():
        print(f"    node {node_id}: {cache}")

    if stats["forwards"] == 0:
        print("WARNING: zero forward hits -- check cluster topology / cache diversity")
        raise SystemExit(1)
    print("OK: forwarding is nonzero")


def cmd_train(args: argparse.Namespace) -> None:
    """Train shared-policy DQN from scratch."""
    config = load_config(args.config)
    _apply_common(config, args)
    config["forwarding_same_cluster_only"] = True
    if args.seed is not None:
        config["train_seed"] = args.seed

    run_name = args.run_name
    if run_name is None:
        level = config["comm_level"]
        pen = float(config.get("overlap_penalty_weight", 0.0))
        run_name = f"dqn_multi_level{level}"
        if pen > 0:
            run_name = f"{run_name}_ov{pen:g}"

    _, run_paths = train_multi_agent_dqn(
        total_timesteps=args.timesteps,
        config=config,
        run_name=run_name,
        early_stopping=not args.no_early_stopping,
        eval_freq=args.eval_freq,
    )
    print(f"Run directory: {run_paths.root}")
    print(f"  final model: {run_paths.model}")
    print(f"  best model:  {run_paths.best_model}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent verify/train (comm levels 0-3)")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify", help="Random-action rollout; check forwarding")
    verify.add_argument("--episodes", type=int, default=3)
    verify.add_argument("--seed", type=int, default=42)
    verify.add_argument("--nodes", type=int, default=None)
    verify.add_argument("--clusters", type=int, default=None)
    verify.add_argument("--traffic", type=str, default="shifting")
    verify.add_argument("--locality-factor", type=float, default=None)
    verify.add_argument("--comm-level", type=int, default=0, choices=[0, 1, 2, 3])
    verify.add_argument(
        "--allow-cross-cluster",
        action="store_true",
        help="Allow forwarding across cluster bridges (default: same-cluster only)",
    )
    verify.set_defaults(func=cmd_verify)

    train = sub.add_parser(
        "train", help="Train shared-policy SB3 DQN on MultiAgentCachingEnv"
    )
    train.add_argument("--timesteps", type=int, default=100_000)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--run-name", type=str, default=None)
    train.add_argument("--nodes", type=int, default=None)
    train.add_argument("--clusters", type=int, default=None)
    train.add_argument("--traffic", type=str, default="shifting")
    train.add_argument(
        "--locality-factor",
        type=float,
        default=None,
        help="Per-node demand heterogeneity (0=identical, 1=independent)",
    )
    train.add_argument("--comm-level", type=int, default=0, choices=[0, 1, 2, 3])
    train.add_argument(
        "--overlap-penalty",
        type=float,
        default=None,
        help="Override overlap_penalty_weight (one-shot cost when caching a neighbor-held container)",
    )
    train.add_argument(
        "--comm-penalty",
        type=float,
        default=None,
        help="Override comm_penalty_lambda (Level-3 cost per communication event)",
    )
    train.add_argument(
        "--gradient-steps",
        type=int,
        default=None,
        help="Gradient steps per train call (default: num_nodes)",
    )
    train.add_argument("--eval-freq", type=int, default=None)
    train.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping on eval plateau",
    )
    train.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
