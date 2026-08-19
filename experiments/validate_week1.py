from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.baselines import LFUPolicy, LRUPolicy
from configs import load_config
from env.caching_env import CachingEnv
from run_paths import figure_path

NUM_STEPS = 10_000
PLOT_PATH = figure_path("week1_baselines.png")
HIT_RATE_BOUNDS = (0.15, 0.50)


def baseline_step(env: CachingEnv, policy, obs: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """Request-first eviction step matching CachingEnv.step."""
    node = env.network.nodes[env.active_node]
    action = policy.act(
        obs,
        env._pending,
        cache=node.cache,
        cache_capacity=env.cache_capacity,
        num_container_types=env.num_container_types,
    )
    obs, reward, _terminated, truncated, _info = env.step(action)
    return obs, reward, truncated


def run_policy(policy_name: str, policy, config: dict, seed: int = 42) -> dict:
    env = CachingEnv(config, seed=seed)
    obs, _ = env.reset(seed=seed)

    hits = 0
    forwards = 0
    misses = 0
    cumulative_rewards: list[float] = []
    total_reward = 0.0

    for _ in range(NUM_STEPS):
        obs, reward, truncated = baseline_step(env, policy, obs)
        total_reward += reward
        cumulative_rewards.append(total_reward)

        if reward == config["reward_local_hit"]:
            hits += 1
        elif reward == config["reward_forward_hit"]:
            forwards += 1
        elif reward == config["reward_cloud_pull"]:
            misses += 1

        if truncated:
            obs, _ = env.reset()

    total = hits + forwards + misses
    return {
        "policy": policy_name,
        "hits": hits,
        "forwards": forwards,
        "misses": misses,
        "hit_rate": hits / total,
        "forward_rate": forwards / total,
        "cloud_rate": misses / total,
        "cumulative_rewards": cumulative_rewards,
    }


def print_metrics(result: dict) -> None:
    print(f"\n{result['policy']}")
    print(f"  cache hit rate:   {result['hit_rate']:.1%}")
    print(f"  forward rate:     {result['forward_rate']:.1%}")
    print(f"  cloud pull rate:  {result['cloud_rate']:.1%}")
    print(
        f"  counts: hits={result['hits']} "
        f"forwards={result['forwards']} misses={result['misses']}"
    )


def sanity_check(result: dict) -> None:
    hit_rate = result["hit_rate"]
    if hit_rate == 0.0 or hit_rate == 1.0:
        raise RuntimeError(
            f"{result['policy']} hit rate is {hit_rate:.1%}; expected a mix of outcomes"
        )
    low, high = HIT_RATE_BOUNDS
    if not (low <= hit_rate <= high):
        print(
            f"  warning: hit rate {hit_rate:.1%} outside expected "
            f"{low:.0%}-{high:.0%} range"
        )


def plot_results(lru: dict, lfu: dict, out_path: Path | None = None) -> None:
    dest = out_path if out_path is not None else PLOT_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(lru["cumulative_rewards"], label="LRU")
    plt.plot(lfu["cumulative_rewards"], label="LFU")
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative reward")
    plt.title("Week 1 Baselines")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(dest, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Week-1 baseline validation")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Base output directory for plots")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.output_dir is not None:
        plot_path = Path(args.output_dir) / "figures" / "week1_baselines.png"
    else:
        plot_path = PLOT_PATH

    print("Running week-1 baseline validation")
    print(f"  steps: {NUM_STEPS}")
    print(
        f"  K={config['num_container_types']} "
        f"C={config['cache_capacity']} "
        f"nodes={config['num_nodes']}"
    )

    lru = run_policy("LRU", LRUPolicy(), config, seed=args.seed)
    lfu = run_policy("LFU", LFUPolicy(), config, seed=args.seed)

    print_metrics(lru)
    print_metrics(lfu)

    sanity_check(lru)
    sanity_check(lfu)

    plot_results(lru, lfu, out_path=plot_path)
    print(f"\nSaved plot to {plot_path}")


if __name__ == "__main__":
    main()
