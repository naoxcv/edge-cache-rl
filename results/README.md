# Results layout

All generated artifacts live under `results/`. Model `.zip` files and TensorBoard
event dirs are **gitignored**; keep configs / eval curves / figures in the tree.

## Directory tree

```
results/
  figures/                  # static plots (not tied to one training run)
    week1_baselines.png
    traffic_*.png
    shap_*.png
  data/                     # experiment CSVs and JSON summaries
    week7/                  # scalability / traffic / locality sweeps
    week8/                  # SHAP analysis JSON
  runs/
    <run_name>/             # one directory per training run
      config.yaml           # config snapshot at train start
      model.zip             # final checkpoint (gitignored)
      best_model.zip        # best multi-seed eval checkpoint (gitignored)
      evaluations.npz       # eval curve (timesteps, per-seed means)
      tensorboard/          # TensorBoard logs (gitignored)
```

## Canonical runs

| Run name | Meaning |
|----------|---------|
| `dqn_generalized` | Single-node DQN (stationary / general) |
| `dqn_shifting` | Single-node DQN trained on shifting traffic |
| `dqn_bursty` | Single-node DQN trained on bursty traffic |
| `dqn_multi_level*_scratch_*` | **Trained from scratch** — comparable across levels |
| `dqn_multi_level3_scratch_*` | Level 3 selective (Q-margin); report return **and** comm events |

**Comm-level comparisons use `_scratch_` runs.** All levels are trained from scratch
under identical settings for fair comparison.

**Traffic locality:** `locality_factor=0` reproduces identical node rankings;
`0.3` is the clustered heterogeneous-demand default; `1` gives independent rankings.

## Week-7 sweeps

```bash
# Full train+eval matrix (locality, traffic, scalability)
./experiments/week7_run_all.sh

# Evaluate only (requires checkpoints)
python experiments/week7_sweep.py all --no-train --seeds 42,0,7 --episodes 20
```

Artifacts: `results/data/week7/{exp1,scalability,traffic,locality}_{raw.json,summary.csv}`

## Week-8 SHAP + dashboard

```bash
python experiments/shap_analysis.py
# figures: results/figures/shap_*.png
# json:    results/data/week8/shap_l0_vs_l1.json

cd dashboard/frontend && npm install && npm run build && cd ../..
python -m dashboard.server   # http://127.0.0.1:8000
```
