#!/usr/bin/env python3
"""Measure local-only Q-margins on eviction decisions for L3 threshold calibration."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.multi_agent import _local_dim, _q_margin, _q_values, resolve_multi_model_path
from configs import load_config
from env.multi_agent_caching_env import MultiAgentCachingEnv, needs_decision_from_obs
from stable_baselines3 import DQN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="dqn_evict_level3_scratch_loc0.3")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = resolve_multi_model_path(args.run, prefer_best=True)
    if path is None:
        raise SystemExit(f"missing model {args.run}")
    cfg = load_config()
    cfg.update(
        {
            "num_nodes": 10,
            "num_clusters": 3,
            "traffic_pattern": "shifting",
            "locality_factor": 0.3,
            "comm_level": 3,
            "forwarding_same_cluster_only": True,
        }
    )
    model = DQN.load(str(path), device="cpu")
    env = MultiAgentCachingEnv(cfg, seed=args.seed)
    local_dim = _local_dim(cfg)
    margins: list[float] = []
    abs_q1: list[float] = []
    n_steps = 0
    n_decisions = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed if ep == 0 else None)
        while True:
            actions = {}
            for aid, agent_obs in obs.items():
                n_steps += 1
                local = np.array(agent_obs, copy=True)
                local[local_dim:] = 0.0
                if needs_decision_from_obs(local[:local_dim]):
                    n_decisions += 1
                    m = _q_margin(model, local)
                    margins.append(m)
                    q = _q_values(model, local)
                    abs_q1.append(float(q.max().detach()))
                action, _ = model.predict(agent_obs, deterministic=True)
                actions[aid] = int(action)
            obs, _, terms, truncs, _ = env.step(actions)
            if terms.get("__all__") or truncs.get("__all__"):
                break

    arr = np.asarray(margins, dtype=np.float64)
    q1 = np.asarray(abs_q1, dtype=np.float64)
    print(f"model={path}")
    print(f"steps={n_steps} decisions={n_decisions} ({100 * n_decisions / max(n_steps, 1):.1f}%)")
    print(
        f"Qmax: mean={q1.mean():.2f} p50={np.median(q1):.2f} "
        f"p10={np.percentile(q1, 10):.2f} p90={np.percentile(q1, 90):.2f}"
    )
    print(
        f"margin: mean={arr.mean():.3f} p10={np.percentile(arr, 10):.3f} "
        f"p25={np.percentile(arr, 25):.3f} p50={np.median(arr):.3f} "
        f"p75={np.percentile(arr, 75):.3f} p90={np.percentile(arr, 90):.3f} "
        f"p99={np.percentile(arr, 99):.3f}"
    )
    print(f"rel margin (m/|Qmax|): median={(arr / np.maximum(np.abs(q1), 1e-6)).mean():.4f}")
    print()
    print(f"{'threshold':>12} {'%decisions':>12} {'comms/ep':>10} {'%all-steps':>12}")
    for t in (0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        frac = float((arr < t).mean())
        comms_ep = frac * n_decisions / args.episodes
        print(f"{t:12.2f} {100 * frac:11.1f}% {comms_ep:10.0f} {100 * comms_ep / 10000:11.2f}%")


if __name__ == "__main__":
    main()
