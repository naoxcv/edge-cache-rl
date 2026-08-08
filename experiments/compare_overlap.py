from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.multi_agent import evaluate_sb3_dqn, resolve_multi_model_path
from configs import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DQN conditions with/without overlap penalty + communication"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Single eval seed")
    parser.add_argument("--seeds", type=str, default="42,0,7")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default="results", help="Base output directory")
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--traffic", type=str, default="shifting")
    parser.add_argument("--locality-factor", type=float, default=0.3)
    parser.add_argument(
        "--conditions",
        type=str,
        default="L0:0:0:dqn_multi_level0,"
        "L0+pen:0:0.3:dqn_multi_level0_ov0.3,"
        "L1+pen:1:0.3:dqn_multi_level1_ov0.3,"
        "L2+pen:2:0.3:dqn_multi_level2_ov0.3",
        help="Comma list of name:comm_level:overlap_weight:run_name",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    conditions = []
    for raw in args.conditions.split(","):
        raw = raw.strip()
        if not raw:
            continue
        name, level, weight, run = raw.split(":")
        conditions.append(
            {
                "name": name,
                "comm_level": int(level),
                "overlap_penalty_weight": float(weight),
                "run": run,
            }
        )

    print("Overlap-penalty communication comparison (action-time penalty)")
    print(
        f"  nodes={args.nodes} clusters={args.clusters} traffic={args.traffic} "
        f"locality={args.locality_factor} episodes={args.episodes} seeds={seeds}"
    )
    print()

    resolved = {}
    for cond in conditions:
        path = resolve_multi_model_path(cond["run"], prefer_best=True)
        resolved[cond["name"]] = path
        print(
            f"  {cond['name']}: level={cond['comm_level']} "
            f"ov={cond['overlap_penalty_weight']} -> {path or 'MISSING'}"
        )
    print()

    agg: dict[str, list[dict]] = {c["name"]: [] for c in conditions}

    for seed in seeds:
        print(f"--- seed={seed} ---")
        for cond in conditions:
            path = resolved[cond["name"]]
            if path is None:
                print(f"  {cond['name']:>8}  SKIP")
                continue
            cfg = load_config(args.config)
            cfg["num_nodes"] = args.nodes
            cfg["num_clusters"] = args.clusters
            cfg["traffic_pattern"] = args.traffic
            cfg["locality_factor"] = args.locality_factor
            cfg["comm_level"] = cond["comm_level"]
            cfg["overlap_penalty_weight"] = cond["overlap_penalty_weight"]
            cfg["forwarding_same_cluster_only"] = True
            cfg["enable_forwarding"] = True

            result = evaluate_sb3_dqn(
                path, cfg, num_episodes=args.episodes, seed=seed
            )
            agg[cond["name"]].append(result)
            print(
                f"  {cond['name']:>8}  "
                f"task={result['task_return_mean']:8.1f}  "
                f"pen={result['overlap_penalty_mean']:8.1f}  "
                f"tot={result['ep_rew_mean']:8.1f}  "
                f"hit={result['hit_rate']:.1%}  fwd={result['forward_rate']:.1%}  "
                f"cloud={result['cloud_rate']:.1%}  "
                f"div={result.get('cache_diversity', '?')}/{args.nodes}"
            )
        print()

    print("=== Mean across seeds ===")
    print(
        f"{'name':>8}  {'taskRet':>10}  {'penalty':>10}  {'total':>10}  "
        f"{'hit':>7}  {'fwd':>7}  {'cloud':>7}  {'div':>6}"
    )
    means = {}
    for cond in conditions:
        rows = agg[cond["name"]]
        if not rows:
            continue
        means[cond["name"]] = {
            "task": float(np.mean([r["task_return_mean"] for r in rows])),
            "pen": float(np.mean([r["overlap_penalty_mean"] for r in rows])),
            "tot": float(np.mean([r["ep_rew_mean"] for r in rows])),
            "hit": float(np.mean([r["hit_rate"] for r in rows])),
            "fwd": float(np.mean([r["forward_rate"] for r in rows])),
            "cloud": float(np.mean([r["cloud_rate"] for r in rows])),
            "div": float(np.mean([r.get("cache_diversity", 0) for r in rows])),
        }
        m = means[cond["name"]]
        print(
            f"{cond['name']:>8}  {m['task']:10.1f}  {m['pen']:10.1f}  {m['tot']:10.1f}  "
            f"{m['hit']:6.1%}  {m['fwd']:6.1%}  {m['cloud']:6.1%}  {m['div']:6.1f}"
        )

    if "L0" in means and "L0+pen" in means and "L1+pen" in means:
        print()
        print("Key contrasts (task return / diversity):")
        print(
            f"  L1+pen − L0+pen:  "
            f"task={means['L1+pen']['task'] - means['L0+pen']['task']:+.1f}  "
            f"pen={means['L1+pen']['pen'] - means['L0+pen']['pen']:+.1f}  "
            f"div={means['L1+pen']['div'] - means['L0+pen']['div']:+.1f}  "
            f"(communication value under penalty)"
        )
        print(
            f"  L0+pen − L0:      "
            f"task={means['L0+pen']['task'] - means['L0']['task']:+.1f}  "
            f"pen={means['L0+pen']['pen'] - means['L0']['pen']:+.1f}  "
            f"div={means['L0+pen']['div'] - means['L0']['div']:+.1f}  "
            f"(penalty without visibility)"
        )
        if "L2+pen" in means:
            print(
                f"  L2+pen − L1+pen:  "
                f"task={means['L2+pen']['task'] - means['L1+pen']['task']:+.1f}  "
                f"pen={means['L2+pen']['pen'] - means['L1+pen']['pen']:+.1f}  "
                f"div={means['L2+pen']['div'] - means['L1+pen']['div']:+.1f}  "
                f"(richer neighbor state)"
            )


if __name__ == "__main__":
    main()
