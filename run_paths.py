from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"
RUNS_DIR = RESULTS_ROOT / "runs"
FIGURES_DIR = RESULTS_ROOT / "figures"


@dataclass(frozen=True)
class RunPaths:
    """Canonical layout for one training run under results/runs/<name>/."""

    root: Path

    @property
    def model(self) -> Path:
        """Path to the final model checkpoint."""
        return self.root / "model.zip"

    @property
    def best_model(self) -> Path:
        """Path to the best model checkpoint (by eval reward)."""
        return self.root / "best_model.zip"

    @property
    def evaluations(self) -> Path:
        """Path to the evaluation results archive."""
        return self.root / "evaluations.npz"

    @property
    def config_snapshot(self) -> Path:
        """Path to the saved training configuration."""
        return self.root / "config.yaml"

    @property
    def tensorboard(self) -> Path:
        """Path to the TensorBoard log directory."""
        return self.root / "tensorboard"

    @classmethod
    def create(cls, run_name: str | None = None) -> RunPaths:
        """Create a RunPaths for the given name, defaulting to a timestamped name."""
        if run_name is None:
            run_name = f"dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        root = RUNS_DIR / run_name
        root.mkdir(parents=True, exist_ok=True)
        return cls(root=root)

    def ensure_dirs(self) -> None:
        """Create the run root and TensorBoard directories if they don't exist."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.tensorboard.mkdir(parents=True, exist_ok=True)


def figure_path(name: str) -> Path:
    """Return the path for a named figure file, creating the figures directory if needed."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / name


def resolve_run_name(path_arg: str) -> str:
    """Normalize CLI paths like results/runs/foo or results/foo into a run name."""
    path = Path(path_arg)
    parts = path.parts
    if "runs" in parts:
        idx = parts.index("runs")
        return "/".join(parts[idx + 1 :]) or path.name
    name = path.name
    if name.endswith("_best"):
        name = name[: -len("_best")]
    return name


def resolve_model_path(
    path_arg: str | Path,
    *,
    prefer_best: bool = True,
) -> Path | None:
    """Resolve a model .zip from a run directory, run name, or legacy flat path."""
    path = Path(path_arg)
    if not path.is_absolute():
        path = ROOT / path

    run_name = resolve_run_name(str(path_arg))
    run_root = RUNS_DIR / run_name
    candidates: list[Path] = []

    if path.suffix == ".zip":
        candidates.append(path)
    candidates.append(path.with_suffix(".zip"))

    if path.is_dir():
        if prefer_best:
            candidates.append(path / "best_model.zip")
        candidates.append(path / "model.zip")

    if prefer_best:
        candidates.append(run_root / "best_model.zip")
    candidates.append(run_root / "model.zip")
    candidates.append(RESULTS_ROOT / f"{run_name}_best" / "best_model.zip")
    candidates.append(RESULTS_ROOT / f"{run_name}.zip")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None
