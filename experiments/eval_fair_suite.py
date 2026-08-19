#!/usr/bin/env python3
"""Fair multi-node evaluation: request-first LRU/LFU + shared/IDQN checkpoints.

Heuristics and DQN both see the pending request and choose an eviction slot
(or reject). Writes CSV/JSON under results/data/fair_eval_evict/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Line-buffered progress when piped to tee/log files.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import yaml
from stable_baselines3 import DQN

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.baselines import LFUPolicy, LRUPolicy
from agents.multi_agent import (
    evaluate_independent_dqn,
    evaluate_sb3_dqn,
    resolve_multi_model_path,
)
from configs import load_config
from env.multi_agent_caching_env import agent_id
from experiments.compare_comm_levels import evaluate_baselines
from experiments.week7_sweep import (
    Condition,
    build_conditions,
    clusters_for,
    make_config,
)

OUT_DIR = ROOT / "results" / "data" / "fair_eval_evict"
DEFAULT_SEEDS = (42, 0, 7)


@dataclass(frozen=True)
class EvalSpec:
    policy: str
    kind: str  # baseline | shared | idqn
    run_name: str | None = None
    comm_level: int | None = None
    checkpoint: str = "best"  # best | final


def baseline_specs() -> list[EvalSpec]:
    return [
        EvalSpec("LRU", "baseline"),
        EvalSpec("LFU", "baseline"),
    ]


def shared_specs(levels: tuple[int, ...]) -> list[EvalSpec]:
    specs: list[EvalSpec] = []
    for level in levels:
        for ckpt in ("best", "final"):
            specs.append(
                EvalSpec(
                    f"Shared-L{level}-{ckpt}",
                    "shared",
                    comm_level=level,
                    checkpoint=ckpt,
                )
            )
    return specs


def idqn_specs() -> list[EvalSpec]:
    return [
        EvalSpec("IDQN-L0-best", "idqn", "dqn_idqn_level0_400k_gs10_loc0.3", 0, "best"),
        EvalSpec("IDQN-L0-final", "idqn", "dqn_idqn_level0_400k_gs10_loc0.3", 0, "final"),
        EvalSpec("IDQN-L1-best", "idqn", "dqn_idqn_level1_400k_gs10_loc0.3", 1, "best"),
        EvalSpec("IDQN-L1-final", "idqn", "dqn_idqn_level1_400k_gs10_loc0.3", 1, "final"),
    ]


def run_name_for(cond: Condition) -> str:
    return cond.run_name


def load_idqn(run_name: str, *, checkpoint: str) -> dict[str, DQN]:
    run_dir = ROOT / "results" / "runs" / run_name
    with (run_dir / "config.yaml").open() as f:
        cfg = yaml.safe_load(f)
    n = int(cfg["num_nodes"])
    prefer_best = checkpoint == "best"
    models: dict[str, DQN] = {}
    for i in range(n):
        stem = f"best_model_node{i}" if prefer_best else f"model_node{i}"
        path = run_dir / f"{stem}.zip"
        if prefer_best and not path.exists():
            path = run_dir / f"model_node{i}.zip"
        if not path.exists():
            raise FileNotFoundError(path)
        models[agent_id(i)] = DQN.load(str(path))
    return models


def resolve_shared(cond: Condition, checkpoint: str) -> Path | None:
    run = run_name_for(cond)
    if checkpoint == "best":
        return resolve_multi_model_path(run, prefer_best=True)
    run_dir = ROOT / "results" / "runs" / run
    final = run_dir / "model.zip"
    return final if final.exists() else None


def model_matches_env(model_path: Path, cfg: dict) -> bool:
    """True when checkpoint obs width matches the current env (post same-cluster obs)."""
    from env.multi_agent_caching_env import MultiAgentCachingEnv

    model = DQN.load(str(model_path))
    env = MultiAgentCachingEnv(cfg, seed=0)
    return int(model.observation_space.shape[0]) == int(env.observation_space.shape[0])


def eval_spec(
    spec: EvalSpec,
    cfg: dict,
    *,
    cond: Condition | None,
    seeds: tuple[int, ...],
    episodes: int,
) -> list[dict]:
    rows: list[dict] = []
    for seed in seeds:
        if spec.kind == "baseline":
            factory = LRUPolicy if spec.policy == "LRU" else LFUPolicy
            result = evaluate_baselines(
                factory, spec.policy, cfg, num_episodes=episodes, seed=seed, oracle=False
            )
        elif spec.kind == "shared":
            assert cond is not None
            path = resolve_shared(cond, spec.checkpoint)
            if path is None:
                continue
            result = evaluate_sb3_dqn(path, cfg, num_episodes=episodes, seed=seed)
            result["policy"] = spec.policy
        elif spec.kind == "idqn":
            assert spec.run_name is not None
            models = load_idqn(spec.run_name, checkpoint=spec.checkpoint)
            idqn_cfg = dict(cfg)
            idqn_cfg["comm_level"] = int(spec.comm_level or 0)
            result = evaluate_independent_dqn(
                models, idqn_cfg, num_episodes=episodes, seed=seed
            )
            result["policy"] = spec.policy
        else:
            raise ValueError(spec.kind)

        row = {
            "policy": spec.policy,
            "kind": spec.kind,
            "checkpoint": spec.checkpoint,
            "experiment": cond.experiment if cond else "canonical",
            "nodes": cfg["num_nodes"],
            "clusters": cfg["num_clusters"],
            "traffic": cfg["traffic_pattern"],
            "locality": float(cfg["locality_factor"]),
            "comm_level": spec.comm_level if spec.comm_level is not None else -1,
            "seed": seed,
            "return": float(result["ep_rew_mean"]),
            "hit": float(result["hit_rate"]),
            "fwd": float(result["forward_rate"]),
            "cloud": float(result["cloud_rate"]),
            "div": float(result.get("cache_diversity") or 0.0),
        }
        if spec.kind == "shared" and cond is not None:
            always_on = float(cond.nodes * cfg["episode_length"])
            if cond.level in (1, 2):
                row["comms"] = always_on
            else:
                row["comms"] = float(result.get("comm_events_mean") or 0.0)
            row["comms_frac"] = row["comms"] / always_on if always_on else 0.0
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    keys = (
        "policy",
        "kind",
        "checkpoint",
        "experiment",
        "nodes",
        "clusters",
        "traffic",
        "locality",
        "comm_level",
    )
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)

    out: list[dict] = []
    for key, group in groups.items():
        summary = {k: v for k, v in zip(keys, key)}
        for metric in ("return", "hit", "fwd", "cloud", "div"):
            vals = np.array([g[metric] for g in group], dtype=np.float64)
            summary[f"{metric}_mean"] = float(vals.mean())
            summary[f"{metric}_std"] = float(vals.std())
        if "comms" in group[0]:
            vals = np.array([g.get("comms", 0.0) for g in group], dtype=np.float64)
            summary["comms_mean"] = float(vals.mean())
            summary["comms_std"] = float(vals.std())
        out.append(summary)
    return sorted(
        out,
        key=lambda r: (
            r["experiment"],
            r["nodes"],
            r["traffic"],
            r["locality"],
            r["policy"],
        ),
    )


def print_table(summaries: list[dict], *, experiment: str | None = None) -> None:
    rows = summaries
    if experiment:
        rows = [s for s in rows if s["experiment"] == experiment]
    print()
    print(f"=== Fair eval (request-first LRU/LFU, C+1 actions) — {experiment or 'all'} ===")
    print(
        f"{'policy':>20} {'ret':>16}  {'hit':>11}  {'fwd':>11}  "
        f"{'cloud':>11}  {'N':>3} traf loc"
    )
    for s in rows:
        print(
            f"{s['policy']:>20} "
            f"{s['return_mean']:7.1f}±{s['return_std']:5.1f}  "
            f"{s['hit_mean']:5.1%}  {s['fwd_mean']:5.1%}  {s['cloud_mean']:5.1%}  "
            f"{s['nodes']:>3} {s['traffic']:>8} {s['locality']:.1f}"
        )


def eval_experiment(
    experiment: str,
    *,
    seeds: tuple[int, ...],
    episodes: int,
    include_idqn: bool,
    levels: tuple[int, ...],
) -> list[dict]:
    all_rows: list[dict] = []
    conditions = build_conditions(experiment)
    # Unique physical configs for baselines (comm_level irrelevant).
    seen: set[tuple] = set()
    for cond in conditions:
        key = (cond.nodes, cond.clusters, cond.traffic, cond.locality)
        if key not in seen:
            seen.add(key)
            cfg = make_config(
                Condition(experiment, cond.nodes, cond.clusters, cond.traffic, cond.locality, 0)
            )
            print(f"\nBaselines {experiment} N={cond.nodes} {cond.traffic} loc={cond.locality}")
            for spec in baseline_specs():
                all_rows.extend(
                    eval_spec(spec, cfg, cond=cond, seeds=seeds, episodes=episodes)
                )

    for cond in conditions:
        if cond.level not in levels:
            continue
        cfg = make_config(cond)
        path = resolve_shared(cond, "best")
        if path is None:
            print(f"  SKIP shared L{cond.level} missing {cond.run_name}")
            continue
        if not model_matches_env(path, cfg):
            print(
                f"  SKIP shared L{cond.level} {cond.run_name} "
                f"(checkpoint obs != env obs; retrain required)"
            )
            continue
        print(f"\nShared L{cond.level} {cond.run_name}")
        for ckpt in ("best", "final"):
            spec = EvalSpec(
                f"Shared-L{cond.level}-{ckpt}",
                "shared",
                comm_level=cond.level,
                checkpoint=ckpt,
            )
            rows = eval_spec(spec, cfg, cond=cond, seeds=seeds, episodes=episodes)
            if rows:
                all_rows.extend(rows)

    if include_idqn and experiment == "exp1":
        cfg = make_config(Condition("exp1", 10, 3, "shifting", 0.3, 0))
        print("\nIDQN canonical exp1")
        for spec in idqn_specs():
            try:
                all_rows.extend(
                    eval_spec(spec, cfg, cond=None, seeds=seeds, episodes=episodes)
                )
            except FileNotFoundError as e:
                print(f"  SKIP {spec.policy}: {e}")

    return all_rows


def write_outputs(name: str, rows: list[dict], summaries: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = OUT_DIR / f"{name}_raw.json"
    summary = OUT_DIR / f"{name}_summary.csv"
    raw.write_text(json.dumps(rows, indent=2))
    if summaries:
        fieldnames: list[str] = []
        seen: set[str] = set()
        for s in summaries:
            for k in s:
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
        with summary.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(summaries)
    print(f"Wrote {raw}")
    print(f"Wrote {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair eval suite with request-first baselines")
    parser.add_argument(
        "suite",
        choices=["exp1", "scalability", "traffic", "locality", "all", "canonical"],
        help="canonical = exp1 + IDQN only (fast)",
    )
    parser.add_argument("--seeds", type=str, default="42,0,7")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--levels", type=str, default="0,1,2,3")
    parser.add_argument("--no-idqn", action="store_true")
    args = parser.parse_args()

    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    levels = tuple(int(x.strip()) for x in args.levels.split(",") if x.strip())
    include_idqn = not args.no_idqn

    suites = (
        ["exp1", "scalability", "traffic", "locality"]
        if args.suite == "all"
        else (["exp1"] if args.suite == "canonical" else [args.suite])
    )

    combined: list[dict] = []
    for experiment in suites:
        rows = eval_experiment(
            experiment,
            seeds=seeds,
            episodes=args.episodes,
            include_idqn=include_idqn and experiment == "exp1",
            levels=levels,
        )
        summaries = summarize(rows)
        write_outputs(experiment, rows, summaries)
        print_table(summaries, experiment=experiment)
        combined.extend(rows)

    if len(suites) > 1:
        all_summaries = summarize(combined)
        write_outputs("all", combined, all_summaries)


if __name__ == "__main__":
    main()
