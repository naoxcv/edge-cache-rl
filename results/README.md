# Results layout

Published summaries and figures live here. Training checkpoints and eval curves
stay local under `results/runs/` (gitignored) or ship via the GitHub release zip
under `pretrained_models/`.

```
results/
  figures/                 # SHAP plots
  data/
    fair_eval_evict/       # Exp1 summary CSV (canonical table in README)
    week7/                 # scalability / traffic / locality summary CSVs
    week8/                 # SHAP JSON
  runs/                    # local only — configs, *.zip, evaluations.npz
```

Reproduce Exp1 numbers with pretrained models + `compare_comm_levels.py` (see root README).
