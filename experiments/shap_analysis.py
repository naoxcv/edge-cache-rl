#!/usr/bin/env python3
"""SHAP explainability: L0 vs L1 eviction decisions at locality=0.3.

Collects observations where the shared-policy DQN chooses an *evict* action,
explains the Q-value of that action with GradientExplainer, and compares
feature-group importance (local request freqs vs neighbor cache binaries).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.multi_agent import resolve_multi_model_path, select_action_for_obs
from configs import load_config
from env.multi_agent_caching_env import MultiAgentCachingEnv, local_obs_size
from stable_baselines3 import DQN

FIGURES = ROOT / "results" / "figures"
WEEK8 = ROOT / "results" / "data" / "week8"


def feature_names(env: MultiAgentCachingEnv, node_id: int = 0) -> list[str]:
    """Human-readable labels for one node's observation vector."""
    k = env.num_container_types
    c = env.cache_capacity
    names = [f"slot[{s}]_c[{i}]" for s in range(c) for i in range(k)]
    names.append("local_util")
    names.extend(f"local_req_freq[{i}]" for i in range(k))
    names.extend(f"pending_req[{i}]" for i in range(k))
    names.append("needs_decision")
    if env.comm_level <= 0:
        return names
    neighbors = env.neighbor_lists[node_id]
    feat_k = k if env.comm_level in (1, 3) else env.local_obs_dim
    for slot in range(env.max_neighbors):
        if slot < len(neighbors):
            nid = neighbors[slot]
            prefix = f"nbr[{nid}]"
        else:
            prefix = f"nbr_pad[{slot}]"
        if env.comm_level == 2:
            names.extend(f"{prefix}_local[{i}]" for i in range(feat_k))
        else:
            names.extend(f"{prefix}_cache[{i}]" for i in range(k))
    return names


def feature_groups(k: int, obs_dim: int, cache_capacity: int) -> dict[str, slice]:
    local = local_obs_size(k, cache_capacity)
    slot_end = cache_capacity * k
    util_end = slot_end + 1
    freq_end = util_end + k
    req_end = freq_end + k
    groups = {
        "local_cache": slice(0, slot_end),
        "local_util": slice(slot_end, util_end),
        "local_req_freq": slice(util_end, freq_end),
        "pending_request": slice(freq_end, req_end),
        "needs_decision": slice(req_end, local),
    }
    if obs_dim > local:
        groups["neighbor_cache"] = slice(local, obs_dim)
    return groups


def make_config(comm_level: int, locality: float, config_path: str = "configs/default.yaml") -> dict:
    cfg = load_config(config_path)
    cfg.update(
        {
            "num_nodes": 10,
            "num_clusters": 3,
            "traffic_pattern": "shifting",
            "locality_factor": locality,
            "comm_level": comm_level,
            "overlap_penalty_weight": 0.0,
            "comm_penalty_lambda": 0.0,
            "forwarding_same_cluster_only": True,
            "enable_forwarding": True,
        }
    )
    return cfg


def collect_eviction_obs(
    model: DQN,
    config: dict,
    *,
    seed: int,
    max_steps: int,
    max_samples: int,
    background_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out greedily; return (background_obs, eviction_obs, eviction_actions)."""
    env = MultiAgentCachingEnv(config, seed=seed)
    c = env.cache_capacity
    obs, _ = env.reset(seed=seed)

    all_obs: list[np.ndarray] = []
    evict_obs: list[np.ndarray] = []
    evict_actions: list[int] = []

    steps = 0
    while steps < max_steps and len(evict_obs) < max_samples:
        actions: dict[str, int] = {}
        for aid, agent_obs in obs.items():
            action, _ = select_action_for_obs(
                model, agent_obs, config=config, deterministic=True
            )
            actions[aid] = action
            all_obs.append(np.asarray(agent_obs, dtype=np.float32))
            if 0 <= action < c:
                evict_obs.append(np.asarray(agent_obs, dtype=np.float32))
                evict_actions.append(action)
        obs, _, _, truncateds, _ = env.step(actions)
        steps += 1
        if truncateds.get("__all__", False):
            obs, _ = env.reset()

    if not all_obs:
        raise RuntimeError("No observations collected")
    if not evict_obs:
        raise RuntimeError("No eviction actions collected — try more steps")

    all_arr = np.stack(all_obs)
    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(all_arr), size=min(background_size, len(all_arr)), replace=False)
    background = all_arr[bg_idx]
    return background, np.stack(evict_obs[:max_samples]), np.asarray(evict_actions[:max_samples])


def explain_evictions(
    model: DQN,
    background: np.ndarray,
    eviction_obs: np.ndarray,
    eviction_actions: np.ndarray,
) -> np.ndarray:
    """Mean |SHAP| per feature for Q(chosen eviction action).

    Uses a differentiable wrapper that gathers Q[a] for each sample's action,
    then GradientExplainer. Falls back to KernelExplainer if needed.
    """
    import shap

    device = model.device
    q_net = model.policy.q_net
    q_net.eval()

    actions_t = torch.as_tensor(eviction_actions, dtype=torch.long, device=device)

    class ChosenQ(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            q = q_net(x)
            # During explainer init, batch size may differ from actions_t;
            # use first-action broadcast only for background — for explain
            # we rebind actions. Safer: return full Q and post-select.
            return q

    wrapper = ChosenQ().to(device)
    bg_t = torch.as_tensor(background, dtype=torch.float32, device=device)
    explainer = shap.GradientExplainer(wrapper, bg_t)

    obs_t = torch.as_tensor(eviction_obs, dtype=torch.float32, device=device)
    # shap_values shape: (n_samples, obs_dim, n_actions) for multi-output
    shap_vals = explainer.shap_values(obs_t)
    arr = np.asarray(shap_vals)
    if arr.ndim == 3:
        # (n, features, actions) or (n, actions, features) depending on shap version
        if arr.shape[-1] == model.action_space.n:
            chosen = arr[np.arange(len(eviction_actions)), :, eviction_actions]
        elif arr.shape[1] == model.action_space.n:
            chosen = arr[np.arange(len(eviction_actions)), eviction_actions, :]
        else:
            raise RuntimeError(f"Unexpected SHAP shape {arr.shape}")
    elif arr.ndim == 2:
        chosen = arr
    else:
        raise RuntimeError(f"Unexpected SHAP shape {arr.shape}")

    return np.mean(np.abs(chosen), axis=0)


def group_importance(mean_abs: np.ndarray, groups: dict[str, slice]) -> dict[str, float]:
    raw = {name: float(np.sum(mean_abs[sl])) for name, sl in groups.items()}
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def plot_group_comparison(
    l0_groups: dict[str, float],
    l1_groups: dict[str, float],
    out_path: Path,
) -> None:
    labels = ["local_cache", "local_util", "local_req_freq", "neighbor_cache"]
    x = np.arange(len(labels))
    w = 0.35
    l0 = [l0_groups.get(g, 0.0) for g in labels]
    l1 = [l1_groups.get(g, 0.0) for g in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, l0, w, label="L0 (no comm)", color="#4C78A8")
    ax.bar(x + w / 2, l1, w, label="L1 (neighbor cache)", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Local cache\nbits", "Local\nutilization", "Local request\nfrequencies", "Neighbor\ncache bits"]
    )
    ax.set_ylabel("Share of mean |SHAP| (eviction Q)")
    ax.set_ylim(0, max(max(l0), max(l1), 0.01) * 1.25)
    ax.set_title("SHAP feature-group importance on eviction decisions\n(locality=0.3, shifting)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top_features(
    mean_abs: np.ndarray,
    names: list[str],
    title: str,
    out_path: Path,
    top_n: int = 20,
) -> None:
    idx = np.argsort(mean_abs)[::-1][:top_n]
    vals = mean_abs[idx][::-1]
    labs = [names[i] for i in idx][::-1]
    colors = ["#F58518" if n.startswith("nbr") else "#4C78A8" for n in labs]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labs, fontsize=8)
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_level(
    level: int,
    run_name: str,
    *,
    locality: float,
    seed: int,
    max_steps: int,
    max_samples: int,
    background_size: int,
    config_path: str = "configs/default.yaml",
) -> dict:
    path = resolve_multi_model_path(run_name, prefer_best=True)
    if path is None:
        raise FileNotFoundError(f"Missing model for {run_name}")
    config = make_config(level, locality, config_path=config_path)
    model = DQN.load(str(path), device="cpu")
    print(f"L{level}: loaded {path}")

    background, evict_obs, evict_actions = collect_eviction_obs(
        model,
        config,
        seed=seed,
        max_steps=max_steps,
        max_samples=max_samples,
        background_size=background_size,
    )
    print(
        f"L{level}: background={len(background)} eviction_samples={len(evict_obs)} "
        f"({100 * len(evict_obs) / max(1, max_steps * config['num_nodes']):.1f}% of decisions)"
    )

    mean_abs = explain_evictions(model, background, evict_obs, evict_actions)
    groups = feature_groups(
        int(config["num_container_types"]),
        mean_abs.shape[0],
        int(config["cache_capacity"]),
    )
    group_share = group_importance(mean_abs, groups)

    env = MultiAgentCachingEnv(config, seed=seed)
    names = feature_names(env, node_id=0)
    # Names for node 0; neighbor labels are approximate for shared-policy aggregate.
    if len(names) != len(mean_abs):
        names = [f"f[{i}]" for i in range(len(mean_abs))]
        # Rebuild generic L1 names without node-specific nbr ids
        k = config["num_container_types"]
        names = [f"local_cache[{i}]" for i in range(k)]
        names.append("local_util")
        names.extend(f"local_req_freq[{i}]" for i in range(k))
        if level > 0:
            max_n = env.max_neighbors
            for slot in range(max_n):
                names.extend(f"nbr_slot[{slot}]_cache[{i}]" for i in range(k))

    print(f"L{level} group shares:")
    for g, v in group_share.items():
        print(f"  {g:16s} {v:6.1%}")

    return {
        "level": level,
        "run_name": run_name,
        "n_evictions": int(len(evict_obs)),
        "group_share": group_share,
        "mean_abs_shap": mean_abs.tolist(),
        "feature_names": names,
        "top_features": [
            {"name": names[i], "mean_abs_shap": float(mean_abs[i])}
            for i in np.argsort(mean_abs)[::-1][:30]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP L0 vs L1 eviction analysis")
    parser.add_argument("--locality", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--background", type=int, default=64)
    parser.add_argument(
        "--l0-run",
        default="dqn_evict_level0_scratch_loc0.3",
    )
    parser.add_argument(
        "--l1-run",
        default="dqn_evict_level1_scratch_loc0.3",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default=None, help="Base results directory (overrides default results/)")
    args = parser.parse_args()

    if args.output_dir is not None:
        out_base = Path(args.output_dir)
        figures_dir = out_base / "figures"
        week8_dir = out_base / "week8"
    else:
        figures_dir = FIGURES
        week8_dir = WEEK8

    figures_dir.mkdir(parents=True, exist_ok=True)
    week8_dir.mkdir(parents=True, exist_ok=True)

    l0 = run_level(
        0,
        args.l0_run,
        locality=args.locality,
        seed=args.seed,
        max_steps=args.max_steps,
        max_samples=args.max_samples,
        background_size=args.background,
        config_path=args.config,
    )
    l1 = run_level(
        1,
        args.l1_run,
        locality=args.locality,
        seed=args.seed,
        max_steps=args.max_steps,
        max_samples=args.max_samples,
        background_size=args.background,
        config_path=args.config,
    )

    plot_group_comparison(
        l0["group_share"],
        l1["group_share"],
        figures_dir / "shap_l0_vs_l1_eviction_groups.png",
    )
    plot_top_features(
        np.asarray(l0["mean_abs_shap"]),
        l0["feature_names"],
        "L0 — top features for eviction Q",
        figures_dir / "shap_l0_eviction_top_features.png",
    )
    plot_top_features(
        np.asarray(l1["mean_abs_shap"]),
        l1["feature_names"],
        "L1 — top features for eviction Q (orange = neighbor)",
        figures_dir / "shap_l1_eviction_top_features.png",
    )

    summary = {
        "locality": args.locality,
        "seed": args.seed,
        "l0": {
            "group_share": l0["group_share"],
            "n_evictions": l0["n_evictions"],
            "top_features": l0["top_features"][:15],
        },
        "l1": {
            "group_share": l1["group_share"],
            "n_evictions": l1["n_evictions"],
            "top_features": l1["top_features"][:15],
        },
    }
    out_json = week8_dir / "shap_l0_vs_l1.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("\n=== Side-by-side (share of mean |SHAP| on eviction Q) ===")
    print(f"{'group':16s} {'L0':>8s} {'L1':>8s}")
    for g in ("local_cache", "local_util", "local_req_freq", "neighbor_cache"):
        print(
            f"{g:16s} {l0['group_share'].get(g, 0):8.1%} {l1['group_share'].get(g, 0):8.1%}"
        )
    print(f"\nWrote {out_json}")
    print(f"Wrote {figures_dir / 'shap_l0_vs_l1_eviction_groups.png'}")
    print(f"Wrote {figures_dir / 'shap_l0_eviction_top_features.png'}")
    print(f"Wrote {figures_dir / 'shap_l1_eviction_top_features.png'}")


if __name__ == "__main__":
    main()
