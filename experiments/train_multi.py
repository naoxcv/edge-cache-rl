from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.multi_agent import evaluate_random_policy, train_multi_agent_dqn
from configs import load_config


def cmd_verify(args: argparse.Namespace) -> None:
    config = load_config()
    if args.nodes is not None:
        config["num_nodes"] = args.nodes
    if args.clusters is not None:
        config["num_clusters"] = args.clusters
    config["traffic_pattern"] = args.traffic
    config["forwarding_same_cluster_only"] = not args.allow_cross_cluster

    stats = evaluate_random_policy(
        config, num_episodes=args.episodes, seed=args.seed
    )
    print("Random-policy multi-node verification")
    print(f"  nodes={config['num_nodes']} clusters={config['num_clusters']}")
    print(f"  traffic={config['traffic_pattern']} episodes={args.episodes}")
    print(f"  ep_rew_mean={stats['ep_rew_mean']:.1f} ± {stats['ep_rew_std']:.1f}")
    print(
        f"  network hit={stats['hit_rate']:.1%}  "
        f"forward={stats['forward_rate']:.1%}  "
        f"cloud={stats['cloud_rate']:.1%}"
    )
    print(
        f"  counts hits={stats['hits']} forwards={stats['forwards']} "
        f"misses={stats['misses']}"
    )
    print("  per-node caches:")
    for node_id, cache in stats["caches"].items():
        print(f"    node {node_id}: {cache}")

    if stats["forwards"] == 0:
        print("WARNING: zero forward hits — check cluster topology / cache diversity")
        raise SystemExit(1)
    print("OK: forwarding is nonzero")


def cmd_train(args: argparse.Namespace) -> None:
    config = load_config()
    if args.nodes is not None:
        config["num_nodes"] = args.nodes
    if args.clusters is not None:
        config["num_clusters"] = args.clusters
    config["traffic_pattern"] = args.traffic
    config["forwarding_same_cluster_only"] = True

    _, run_paths = train_multi_agent_dqn(
        total_timesteps=args.timesteps,
        config=config,
        run_name=args.run_name,
        pretrained_path=args.pretrained,
        early_stopping=not args.no_early_stopping,
        eval_freq=args.eval_freq,
    )
    print(f"Run directory: {run_paths.root}")
    print(f"  final model: {run_paths.model}")
    print(f"  best model:  {run_paths.best_model}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent (Level 0) verify/train")
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify", help="Random-action rollout; check forwarding")
    verify.add_argument("--episodes", type=int, default=3)
    verify.add_argument("--seed", type=int, default=42)
    verify.add_argument("--nodes", type=int, default=None)
    verify.add_argument("--clusters", type=int, default=None)
    verify.add_argument("--traffic", type=str, default="shifting")
    verify.add_argument(
        "--allow-cross-cluster",
        action="store_true",
        help="Allow forwarding across cluster bridges (default: same-cluster only)",
    )
    verify.set_defaults(func=cmd_verify)

    train = sub.add_parser(
        "train", help="Train shared-policy SB3 DQN Level 0 on MultiAgentCachingEnv"
    )
    train.add_argument("--timesteps", type=int, default=100_000)
    train.add_argument("--run-name", type=str, default="dqn_multi_level0")
    train.add_argument("--nodes", type=int, default=None)
    train.add_argument("--clusters", type=int, default=None)
    train.add_argument("--traffic", type=str, default="shifting")
    train.add_argument("--eval-freq", type=int, default=None)
    train.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Warm-start from SB3 zip/run name (e.g. dqn_shifting)",
    )
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
