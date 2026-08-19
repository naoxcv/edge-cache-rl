from pathlib import Path

from run_paths import ROOT, RunPaths, resolve_model_path, resolve_run_name


def test_run_paths_layout():
    paths = RunPaths.create("test_run")
    assert paths.root == ROOT / "results" / "runs" / "test_run"
    assert paths.model.name == "model.zip"
    assert paths.best_model.name == "best_model.zip"
    assert paths.tensorboard.name == "tensorboard"


def test_resolve_run_name_strips_legacy_suffix():
    assert resolve_run_name("results/dqn_generalized_best") == "dqn_generalized"
    assert resolve_run_name("results/runs/foo") == "foo"


def test_resolve_model_path_prefers_run_best(tmp_path, monkeypatch):
    run_root = tmp_path / "results" / "runs" / "demo"
    run_root.mkdir(parents=True)
    best = run_root / "best_model.zip"
    final = run_root / "model.zip"
    best.write_text("best")
    final.write_text("final")

    monkeypatch.setattr("run_paths.ROOT", tmp_path)
    monkeypatch.setattr("run_paths.RUNS_DIR", tmp_path / "results" / "runs")

    assert resolve_model_path("demo") == best
    assert resolve_model_path("demo", prefer_best=False) == final


def test_resolve_model_path_legacy_flat_zip(tmp_path, monkeypatch):
    legacy = tmp_path / "results" / "old_model.zip"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy")

    monkeypatch.setattr("run_paths.ROOT", tmp_path)
    monkeypatch.setattr("run_paths.RUNS_DIR", tmp_path / "results" / "runs")
    monkeypatch.setattr("run_paths.RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr("run_paths.PRETRAINED_DIR", tmp_path / "pretrained_models")

    assert resolve_model_path("old_model") == legacy


def test_resolve_model_path_pretrained_fallback(tmp_path, monkeypatch):
    pretrained = tmp_path / "pretrained_models" / "demo" / "best_model.zip"
    pretrained.parent.mkdir(parents=True)
    pretrained.write_text("pretrained")

    monkeypatch.setattr("run_paths.ROOT", tmp_path)
    monkeypatch.setattr("run_paths.RUNS_DIR", tmp_path / "results" / "runs")
    monkeypatch.setattr("run_paths.RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr("run_paths.PRETRAINED_DIR", tmp_path / "pretrained_models")

    assert resolve_model_path("demo") == pretrained
    assert resolve_run_name("pretrained_models/demo") == "demo"
