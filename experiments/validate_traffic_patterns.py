from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import load_config
from env.container import create_catalog
from env.request_generator import RequestGenerator
from run_paths import figure_path

SHIFT_PLOT = figure_path("traffic_shift_distribution.png")
BURST_PLOT = figure_path("traffic_burst_spikes.png")
SAMPLES = 20_000
SHIFT_INTERVAL = 500


def _node0_counts(gen: RequestGenerator, steps: int) -> Counter:
    counts: Counter = Counter()
    for _ in range(steps):
        counts[gen.generate()[0]] += 1
    return counts


def plot_shift_distribution() -> None:
    config = load_config()
    config["traffic_pattern"] = "shifting"
    config["shift_interval"] = SHIFT_INTERVAL

    catalog = create_catalog(config["num_container_types"], seed=42)
    gen = RequestGenerator(config, catalog, seed=42)

    before = _node0_counts(gen, SHIFT_INTERVAL)
    after = _node0_counts(gen, SHIFT_INTERVAL)

    container_ids = sorted(before.keys())
    x = np.arange(len(container_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, [before[cid] for cid in container_ids], width, label="before shift")
    ax.bar(x + width / 2, [after[cid] for cid in container_ids], width, label="after shift")
    ax.set_xlabel("Container ID")
    ax.set_ylabel("Request count")
    ax.set_title(
        f"Request distribution before/after popularity shift "
        f"({SHIFT_INTERVAL} steps each, shifting pattern)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(container_ids, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    SHIFT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SHIFT_PLOT, dpi=150)
    plt.close(fig)
    print(f"Saved {SHIFT_PLOT}")


def plot_burst_spikes() -> None:
    config = load_config()
    config["traffic_pattern"] = "bursty"
    config["burst_probability"] = 0.05
    config["burst_multiplier"] = 10

    catalog = create_catalog(config["num_container_types"], seed=42)
    gen = RequestGenerator(config, catalog, seed=42)

    steps = 2_000
    per_step_dominant: list[int] = []

    for _ in range(steps):
        requests = gen.generate()
        step_counts = Counter(requests)
        dominant_count = max(step_counts.values())
        per_step_dominant.append(dominant_count)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(per_step_dominant, linewidth=0.8)
    axes[0].axhline(1, color="gray", linestyle="--", linewidth=0.8, label="typical Zipf step")
    axes[0].axhline(
        config["burst_multiplier"],
        color="crimson",
        linestyle="--",
        linewidth=0.8,
        label="burst_multiplier",
    )
    axes[0].set_ylabel("Max requests for one container")
    axes[0].set_title("Burst spikes (dominant container count per timestep)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    burst_threshold = max(3, config["burst_multiplier"] // 2)
    burst_steps = [i for i, c in enumerate(per_step_dominant) if c >= burst_threshold]
    axes[1].eventplot([burst_steps], orientation="horizontal", colors="crimson")
    axes[1].set_yticks([0])
    axes[1].set_yticklabels(["burst"])
    axes[1].set_xlabel("Timestep")
    axes[1].set_title(f"Burst timesteps (dominant count ≥ {burst_threshold})")
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    BURST_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(BURST_PLOT, dpi=150)
    plt.close(fig)
    print(f"Saved {BURST_PLOT} ({len(burst_steps)} burst timesteps in {steps})")


def main() -> None:
    plot_shift_distribution()
    plot_burst_spikes()


if __name__ == "__main__":
    main()
