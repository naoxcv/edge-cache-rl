from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.multi_agent import evaluate_independent_dqn, evaluate_sb3_dqn
from configs import load_config
from env.multi_agent_caching_env import agent_id
from run_paths import resolve_model_path


def load_idqn_models(run_dir: Path, *, prefer_best: bool) -> dict[str, DQN]:
    """Load one SB3 DQN per node from an IDQN run directory."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config snapshot: {config_path}")

    import yaml

    with config_path.open() as f:
        config = yaml.safe_load(f)

    n_nodes = int(config["num_nodes"])
    models: dict[str, DQN] = {}
    for node_id in range(n_nodes):
        aid = agent_id(node_id)
        stem = f"best_model_node{aid}" if prefer_best else f"model_node{aid}"
        path = run_dir / f"{stem}.zip"
        if prefer_best and not path.exists():
            path = run_dir / f"model_node{aid}.zip"
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint for node {node_id}: {path}")
        models[aid] = DQN.load(str(path))
    return models


def summarize(rows: list[dict], label: str) -> dict:
    rets = [r["ep_rew_mean"] for r in rows]
    tasks = [r.get("task_return_mean", r["ep_rew_mean"]) for r in rows]
    hits = [r["hit_rate"] for r in rows]
    fwds = [r["forward_rate"] for r in rows]
    clouds = [r["cloud_rate"] for r in rows]
    return {
        "label": label,
        "return_mean": float(np.mean(rets)),
        "return_std": float(np.std(rets)),
        "task_mean": float(np.mean(tasks)),
        "hit_mean": float(np.mean(hits)),
        "fwd_mean": float(np.mean(fwds)),
        "cloud_mean": float(np.mean(clouds)),
        "n_seeds": len(rows),
    }


def print_row(s: dict) -> None:
    print(
        f"{s['label']:>24}  ret={s['return_mean']:8.1f}±{s['return_std']:5.1f}  "
        f"task={s['task_mean']:8.1f}  hit={s['hit_mean']:.1%}  "
        f"fwd={s['fwd_mean']:.1%}  cloud={s['cloud_mean']:.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare IDQN vs shared-policy DQN baselines"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seeds", type=str, default="42,0,7")
    parser.add_argument(
        "--idqn-runs",
        type=str,
        required=True,
        help="Comma-separated run dirs under results/runs/ or absolute paths",
    )
    parser.add_argument(
        "--shared-runs",
        type=str,
        default="",
        help="Optional comma-separated shared-policy run names aligned with --idqn-runs",
    )
    parser.add_argument("--prefer-best", action="store_true", default=True)
    parser.add_argument("--prefer-final", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    prefer_best = not args.prefer_final
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    idqn_run_args = [x.strip() for x in args.idqn_runs.split(",") if x.strip()]
    shared_run_args = (
        [x.strip() for x in args.shared_runs.split(",") if x.strip()]
        if args.shared_runs
        else []
    )

    results: dict[str, object] = {"seeds": seeds, "conditions": []}

    for i, run_arg in enumerate(idqn_run_args):
        run_dir = Path(run_arg)
        if not run_dir.is_absolute():
            run_dir = ROOT / "results" / "runs" / run_arg
        if not run_dir.exists():
            print(f"SKIP missing IDQN run: {run_dir}")
            continue

        import yaml

        with (run_dir / "config.yaml").open() as f:
            config = dict(yaml.safe_load(f))

        models = load_idqn_models(run_dir, prefer_best=prefer_best)
        ckpt = "best" if prefer_best else "final"
        label = f"IDQN-L{config.get('comm_level', 0)} ({ckpt})"
        print(f"\n=== {label} — {run_dir.name} ===")

        per_seed: list[dict] = []
        for seed in seeds:
            result = evaluate_independent_dqn(
                models, config, num_episodes=args.episodes, seed=seed
            )
            result["eval_seed"] = seed
            per_seed.append(result)
            print(
                f"  seed={seed}  ret={result['ep_rew_mean']:8.1f}  "
                f"hit={result['hit_rate']:.1%}  fwd={result['forward_rate']:.1%}  "
                f"cloud={result['cloud_rate']:.1%}"
            )

        summary = summarize(per_seed, label)
        print_row(summary)
        results["conditions"].append(
            {"run": run_dir.name, "type": "idqn", "summary": summary, "per_seed": per_seed}
        )

        if i < len(shared_run_args) and shared_run_args[i]:
            shared_path = resolve_model_path(shared_run_args[i], prefer_best=True)
            if shared_path is None:
                print(f"  Shared-policy model missing: {shared_run_args[i]}")
                continue
            shared_label = f"Shared-L{config.get('comm_level', 0)}"
            print(f"\n--- {shared_label} — {shared_run_args[i]} ---")
            shared_rows: list[dict] = []
            for seed in seeds:
                cfg = dict(config)
                cfg["comm_level"] = int(shared_run_args[i].split("level")[-1][0])
                result = evaluate_sb3_dqn(
                    shared_path, cfg, num_episodes=args.episodes, seed=seed
                )
                result["eval_seed"] = seed
                shared_rows.append(result)
                print(
                    f"  seed={seed}  ret={result['ep_rew_mean']:8.1f}  "
                    f"hit={result['hit_rate']:.1%}  fwd={result['forward_rate']:.1%}  "
                    f"cloud={result['cloud_rate']:.1%}"
                )
            shared_summary = summarize(shared_rows, shared_label)
            print_row(shared_summary)
            delta = shared_summary["return_mean"] - summary["return_mean"]
            print(f"  Shared − IDQN: {delta:+.1f}")
            results["conditions"].append(
                {
                    "run": shared_run_args[i],
                    "type": "shared",
                    "summary": shared_summary,
                    "per_seed": shared_rows,
                    "delta_vs_idqn": delta,
                }
            )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
