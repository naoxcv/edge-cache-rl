# Pretrained models (canonical Exp1)

Shared-policy DQN checkpoints for **10 nodes / shifting / locality=0.3** (eviction-only
`Discrete(C+1)` MDP). Used by `compare_comm_levels.py`, SHAP analysis, and the demo.

| Run | Comm level | Exp1 return (best) |
|-----|------------|--------------------|
| `dqn_evict_level0_scratch_loc0.3` | L0 local only | 3783 |
| `dqn_evict_level1_scratch_loc0.3` | L1 always-on neighbor cache | **4240** |
| `dqn_evict_level2_scratch_loc0.3` | L2 full neighbor obs | 3699 |
| `dqn_evict_level3_scratch_loc0.3` | L3 selective (Q-margin) | 3511 |

Each run folder contains `best_model.zip` and `config.yaml`.

## Where to unzip

From the repo root (same directory as `README.md`, `agents/`, `experiments/`):

```bash
cd edge-cache-rl
unzip edge-cache-rl-pretrained-exp1.zip
```

The archive should produce this layout:

```
edge-cache-rl/
  pretrained_models/
    dqn_evict_level0_scratch_loc0.3/best_model.zip
    dqn_evict_level1_scratch_loc0.3/best_model.zip
    ...
```

Evaluation scripts load checkpoints from `pretrained_models/<run_name>/` automatically.

## Verify

```bash
python experiments/compare_comm_levels.py --config configs/default.yaml
```
