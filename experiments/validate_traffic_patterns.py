from __future__ import annotations

import argparse
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


def plot_shift_distribution(config_path: str = "configs/default.yaml", shift_plot: Path | None = None, seed: int = 42) -> None:
    if shift_plot is None:
        shift_plot = SHIFT_PLOT
    config = load_config(config_path)
    config["traffic_pattern"] = "shifting"
    config["shift_interval"] = SHIFT_INTERVAL

    catalog = create_catalog(config["num_container_types"], seed=seed)
    gen = RequestGenerator(config, catalog, seed=seed)

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
    shift_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(shift_plot, dpi=150)
    plt.close(fig)
    print(f"Saved {shift_plot}")


def plot_burst_spikes(config_path: str = "configs/default.yaml", burst_plot: Path | None = None, seed: int = 42) -> None:
    if burst_plot is None:
        burst_plot = BURST_PLOT
    config = load_config(config_path)
    config["traffic_pattern"] = "bursty"
    config["burst_probability"] = 0.05
    config["burst_multiplier"] = 10

    catalog = create_catalog(config["num_container_types"], seed=seed)
    gen = RequestGenerator(config, catalog, seed=seed)

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
    burst_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(burst_plot, dpi=150)
    plt.close(fig)
    print(f"Saved {burst_plot} ({len(burst_steps)} burst timesteps in {steps})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate traffic patterns (shift + burst)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Base output directory for plots")
    args = parser.parse_args()

    if args.output_dir is not None:
        out_base = Path(args.output_dir) / "figures"
        shift_out = out_base / "traffic_shift_distribution.png"
        burst_out = out_base / "traffic_burst_spikes.png"
    else:
        shift_out = None
        burst_out = None

    plot_shift_distribution(config_path=args.config, shift_plot=shift_out, seed=args.seed)
    plot_burst_spikes(config_path=args.config, burst_plot=burst_out, seed=args.seed)


if __name__ == "__main__":
    main()
