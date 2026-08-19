#!/usr/bin/env python3
"""Week-7 experiment sweeps: scalability, traffic robustness, locality.

Trains missing from-scratch DQN runs, then evaluates with mean ± std across seeds.
Reuses existing Experiment-1 checkpoints (10 nodes / shifting / loc 0.3).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.multi_agent import evaluate_sb3_dqn, resolve_multi_model_path, train_multi_agent_dqn
from configs import load_config

RESULTS_DIR = ROOT / "results" / "data" / "week7"
DEFAULT_SEEDS = (42, 0, 7)


@dataclass(frozen=True)
class Condition:
    experiment: str
    nodes: int
    clusters: int
    traffic: str
    locality: float
    level: int

    @property
    def run_name(self) -> str:
        # Keep Experiment-1 names for the canonical 10/shifting/loc0.3 cells.
        if (
            self.nodes == 10
            and self.clusters == 3
            and self.traffic == "shifting"
            and abs(self.locality - 0.3) < 1e-9
        ):
            return f"dqn_evict_level{self.level}_scratch_loc0.3"
        # Keep one decimal so 0.0 stays "0.0" (":g" would emit "0").
        loc = f"{self.locality:.1f}"
        return (
            f"dqn_evict_level{self.level}_scratch_"
            f"n{self.nodes}_c{self.clusters}_{self.traffic}_loc{loc}"
        )


def clusters_for(nodes: int) -> int:
    """Keep ~3–5 nodes per cluster, matching the 10/3 baseline ratio."""
    return {5: 2, 10: 3, 25: 5, 50: 10}[nodes]


def timesteps_for(nodes: int) -> int:
    # Larger N writes more transitions/step and uses gradient_steps=N; shorten a bit.
    if nodes <= 5:
        return 300_000
    if nodes <= 10:
        return 400_000
    if nodes <= 25:
        return 300_000
    return 250_000


def eval_freq_for(nodes: int) -> int:
    return max(20_000, timesteps_for(nodes) // 16)


def build_conditions(experiment: str) -> list[Condition]:
    levels = (0, 1, 2, 3)
    if experiment == "exp1":
        return [
            Condition("exp1", 10, 3, "shifting", 0.3, level) for level in levels
        ]
    if experiment == "scalability":
        out: list[Condition] = []
        for nodes in (5, 10, 25, 50):
            for level in levels:
                out.append(
                    Condition(
                        "scalability",
                        nodes,
                        clusters_for(nodes),
                        "shifting",
                        0.3,
                        level,
                    )
                )
        return out
    if experiment == "traffic":
        out = []
        for traffic in ("stationary", "shifting", "bursty"):
            for level in levels:
                out.append(Condition("traffic", 10, 3, traffic, 0.3, level))
        return out
    if experiment == "locality":
        out = []
        for loc in (0.0, 0.2, 0.3, 0.4, 0.6, 0.8):
            for level in levels:
                out.append(Condition("locality", 10, 3, "shifting", loc, level))
        return out
    raise ValueError(f"Unknown experiment={experiment}")


def make_config(cond: Condition, config_path: str = "configs/default.yaml") -> dict:
    cfg = load_config(config_path)
    cfg.update(
        {
            "num_nodes": cond.nodes,
            "num_clusters": cond.clusters,
            "traffic_pattern": cond.traffic,
            "locality_factor": cond.locality,
            "comm_level": cond.level,
            "overlap_penalty_weight": 0.0,
            "comm_penalty_lambda": 0.0,
            "forwarding_same_cluster_only": True,
            "enable_forwarding": True,
        }
    )
    return cfg


def ensure_trained(cond: Condition, *, force: bool = False, config_path: str = "configs/default.yaml") -> Path:
    path = resolve_multi_model_path(cond.run_name, prefer_best=True)
    if path is not None and not force:
        print(f"  SKIP train {cond.run_name} (exists: {path})")
        return path

    cfg = make_config(cond, config_path=config_path)
    steps = timesteps_for(cond.nodes)
    print(
        f"  TRAIN {cond.run_name}  nodes={cond.nodes} level={cond.level} "
        f"traffic={cond.traffic} loc={cond.locality} steps={steps}"
    )
    _, run_paths = train_multi_agent_dqn(
        total_timesteps=steps,
        config=cfg,
        run_name=cond.run_name,
        early_stopping=True,
        eval_freq=eval_freq_for(cond.nodes),
        verbose=1,
    )
    best = run_paths.root / "best_model.zip"
    if not best.exists():
        best = run_paths.model
    return best


def evaluate_condition(
    cond: Condition,
    model_path: Path,
    *,
    seeds: tuple[int, ...],
    episodes: int,
    config_path: str = "configs/default.yaml",
) -> list[dict]:
    rows: list[dict] = []
    cfg = make_config(cond, config_path=config_path)
    for seed in seeds:
        result = evaluate_sb3_dqn(
            model_path, cfg, num_episodes=episodes, seed=seed
        )
        always_on = float(cond.nodes * cfg["episode_length"])
        if cond.level in (1, 2):
            # Always observe neighbors — count every decision as a communication event.
            comms = always_on
        else:
            comms = float(result.get("comm_events_mean") or 0.0)
        row = {
            "experiment": cond.experiment,
            "run_name": cond.run_name,
            "nodes": cond.nodes,
            "clusters": cond.clusters,
            "traffic": cond.traffic,
            "locality": cond.locality,
            "level": cond.level,
            "seed": seed,
            "return": float(result["ep_rew_mean"]),
            "hit": float(result["hit_rate"]),
            "fwd": float(result["forward_rate"]),
            "cloud": float(result["cloud_rate"]),
            "div": float(result.get("cache_diversity") or 0.0),
            "comms": comms,
            "comms_frac": (comms / always_on) if always_on > 0 else 0.0,
        }
        rows.append(row)
        print(
            f"    seed={seed}  L{cond.level}  ret={row['return']:8.1f}  "
            f"hit={row['hit']:.1%}  fwd={row['fwd']:.1%}  "
            f"cloud={row['cloud']:.1%}  comms={row['comms']:.0f}"
        )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    keys = [
        "experiment",
        "nodes",
        "clusters",
        "traffic",
        "locality",
        "level",
        "run_name",
    ]
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)

    summaries = []
    for key, group in groups.items():
        summary = {k: v for k, v in zip(keys, key)}
        for metric in ("return", "hit", "fwd", "cloud", "div", "comms", "comms_frac"):
            vals = np.array([g[metric] for g in group], dtype=np.float64)
            summary[f"{metric}_mean"] = float(vals.mean())
            summary[f"{metric}_std"] = float(vals.std())
        summaries.append(summary)
    return summaries


def write_outputs(experiment: str, rows: list[dict], summaries: list[dict], results_dir: Path | None = None) -> None:
    out_dir = results_dir if results_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{experiment}_raw.json"
    sum_path = out_dir / f"{experiment}_summary.csv"
    with raw_path.open("w") as f:
        json.dump(rows, f, indent=2)
    if summaries:
        with sum_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
    print(f"Wrote {raw_path}")
    print(f"Wrote {sum_path}")


def print_summary_table(summaries: list[dict]) -> None:
    print()
    print("=== Mean ± std across seeds ===")
    header = (
        f"{'exp':>12} {'N':>3} {'traf':>10} {'loc':>4} {'L':>2}  "
        f"{'return':>16}  {'hit':>11}  {'fwd':>11}  {'cloud':>11}  {'comms':>14}"
    )
    print(header)
    for s in sorted(
        summaries,
        key=lambda r: (r["experiment"], r["nodes"], r["traffic"], r["locality"], r["level"]),
    ):
        print(
            f"{s['experiment']:>12} {s['nodes']:>3} {s['traffic']:>10} "
            f"{s['locality']:4.1f} {s['level']:>2}  "
            f"{s['return_mean']:7.1f}±{s['return_std']:5.1f}  "
            f"{s['hit_mean']:5.1%}±{s['hit_std']:4.1%}  "
            f"{s['fwd_mean']:5.1%}±{s['fwd_std']:4.1%}  "
            f"{s['cloud_mean']:5.1%}±{s['cloud_std']:4.1%}  "
            f"{s['comms_mean']:6.0f}±{s['comms_std']:5.0f}"
        )


def run_experiment(
    experiment: str,
    *,
    seeds: tuple[int, ...],
    episodes: int,
    train: bool,
    force_train: bool,
    config_path: str = "configs/default.yaml",
    results_dir: Path | None = None,
) -> None:
    conditions = build_conditions(experiment)
    print(f"Week-7 {experiment}: {len(conditions)} conditions, seeds={seeds}")
    rows: list[dict] = []
    for cond in conditions:
        print(f"\n--- {cond.run_name} ---")
        if train:
            model_path = ensure_trained(cond, force=force_train, config_path=config_path)
        else:
            model_path = resolve_multi_model_path(cond.run_name, prefer_best=True)
            if model_path is None:
                print(f"  MISSING model for {cond.run_name}; skip eval")
                continue
        rows.extend(
            evaluate_condition(cond, model_path, seeds=seeds, episodes=episodes, config_path=config_path)
        )
    summaries = summarize(rows)
    write_outputs(experiment, rows, summaries, results_dir=results_dir)
    print_summary_table(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Week-7 experiment sweeps")
    parser.add_argument(
        "experiment",
        choices=["exp1", "scalability", "traffic", "locality", "all"],
    )
    parser.add_argument("--seeds", type=str, default="42,0,7")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Evaluate existing checkpoints only",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain even if a checkpoint exists",
    )
    parser.add_argument("--seed", type=int, default=42, help="Single eval seed")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default=None, help="Base results directory (overrides default results/week7)")
    args = parser.parse_args()
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    results_dir = Path(args.output_dir) / "week7" if args.output_dir is not None else None
    experiments = (
        ["exp1", "locality", "traffic", "scalability"]
        if args.experiment == "all"
        else [args.experiment]
    )
    for experiment in experiments:
        run_experiment(
            experiment,
            seeds=seeds,
            episodes=args.episodes,
            train=not args.no_train,
            force_train=args.force_train,
            config_path=args.config,
            results_dir=results_dir,
        )


if __name__ == "__main__":
    main()
