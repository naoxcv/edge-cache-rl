from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.single_agent import train_single_node_dqn
from configs import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train single-node DQN on CachingEnv")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results", help="Base output directory")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run directory name under results/runs/ (default: timestamped)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Deprecated alias for --run-name (e.g. results/dqn_generalized)",
    )
    parser.add_argument(
        "--no-randomize-traffic",
        action="store_true",
        help="Train on a fixed traffic seed every episode (old behaviour)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=None,
        help="Evaluate every N steps; 0 disables eval and early stopping",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping on eval plateau",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    _, run_paths = train_single_node_dqn(
        total_timesteps=args.timesteps,
        config=config,
        seed=args.seed,
        run_name=args.run_name,
        save_path=args.save_path,
        randomize_traffic=not args.no_randomize_traffic,
        eval_freq=args.eval_freq if args.eval_freq is not None else None,
        early_stopping=not args.no_early_stopping,
    )
    print(f"Run directory: {run_paths.root}")
    print(f"  final model: {run_paths.model}")
    print(f"  best model:  {run_paths.best_model}")


if __name__ == "__main__":
    main()
