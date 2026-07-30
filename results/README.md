# Results layout

All generated artifacts live under `results/`. Model `.zip` files and TensorBoard
event dirs are **gitignored**; keep configs / eval curves / figures in the tree.

## Directory tree

```
results/
  figures/                  # static plots (not tied to one training run)
    week1_baselines.png
    traffic_*.png
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
| `dqn_multi_level0` | **Multi-node Level 0 baseline** — warm-start from `dqn_shifting` |

## Training

```bash
# Single-node
python experiments/train_single.py --timesteps 1000000 --run-name dqn_generalized

# Multi-node Level 0 (warm-start)
python experiments/train_multi.py train \
  --nodes 3 --clusters 1 --timesteps 100000 \
  --run-name dqn_multi_level0 --pretrained dqn_shifting
```

## Evaluation

```bash
python experiments/compare_policies.py --dqn-path dqn_generalized --seeds 42,0,7
python experiments/compare_multi.py --dqn-path dqn_multi_level0 --nodes 3 --clusters 1 --seeds 42,0,7
```

`resolve_model_path()` still accepts legacy flat paths if present locally.
