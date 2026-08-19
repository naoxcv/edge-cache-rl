# Communication-Aware Multi-Agent RL for Edge Container Caching

Edge nodes learn cooperative container caching under four communication levels (L0–L3), using a shared-policy DQN with request-first eviction actions (`Discrete(C+1)`). **Research question:** how does inter-agent communication affect caching performance, and can selective communication recover full-coordination benefits at lower bandwidth? **Headline result:** on the canonical 10-node shifting workload (locality=0.3), **L1 (always-on neighbor cache summaries) beats the LFU heuristic by 4.4%** episode return (4240 vs 4060) while cutting cloud pulls to 19.8%. **SHAP analysis on eviction decisions confirms L1 agents rely on neighbor cache state** (~51% of feature importance vs 0% at L0), shifting weight away from purely local signals.

## Setup

```bash
git clone <repo-url>
cd edge-cache-rl
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -v           # verify: 109 tests pass
```

Requires Python 3.9+. Tested on macOS ARM64.

### Pretrained models

Download `edge-cache-rl-pretrained-exp1.zip` from the [GitHub release](https://github.com/<org>/edge-cache-rl/releases) and unzip at the **repo root** so `pretrained_models/` sits next to `experiments/` and `configs/`. See `pretrained_models/README.md`.

## Reproduce the key result

```bash
python experiments/compare_comm_levels.py --config configs/default.yaml
```

Evaluates LRU, LFU, and shared-policy DQN checkpoints (`dqn_evict_level{0,1,2,3}_scratch_loc0.3`) on 10 nodes, shifting traffic, locality=0.3, across seeds 42/0/7 (20 episodes each).

### SHAP explainability (L0 vs L1 eviction decisions)

```bash
python experiments/shap_analysis.py --config configs/default.yaml
```

Outputs `results/data/week8/shap_l0_vs_l1.json` and figures under `results/figures/`.

### Demo dashboard

```bash
cd dashboard/frontend && npm install && npm run build && cd ../..
python -m dashboard.server   # http://127.0.0.1:8000
```

## Main results

### Experiment 1 — Communication levels (10 nodes, shifting, locality=0.3)

3 seeds × 20 episodes; mean ± std. Checkpoints: `best_model.zip` for DQN runs.

| Policy | Return | Hit | Fwd | Cloud | Comms/ep |
|--------|--------|-----|-----|-------|----------|
| **Shared-L1** | **4240 ± 151** | 44.1% | 36.1% | **19.8%** | 10,000 |
| LFU | 4060 ± 73 | **56.9%** | 17.8% | 25.3% | — |
| Shared-L0 | 3783 ± 108 | 49.2% | 26.2% | 24.5% | 0 |
| LRU | 3717 ± 92 | 48.2% | 27.2% | 24.6% | — |
| Shared-L2 | 3699 ± 62 | 47.6% | 27.9% | 24.5% | 10,000 |
| Shared-L3 | 3511 ± 9 | 44.2% | 31.1% | 24.7% | 3318 |

### SHAP feature importance (eviction Q, locality=0.3)

| Feature group | L0 share | L1 share |
|---------------|----------|----------|
| Local cache bits | **53.4%** | 25.3% |
| Local request freqs | 22.2% | 4.7% |
| Neighbor cache bits | — | **50.8%** |

## Project structure

```
agents/       Shared-policy DQN, LRU/LFU baselines, training helpers
configs/      default.yaml — network, traffic, reward, and training hyperparameters
dashboard/    FastAPI + React live demo (L0/L1/L3 DQN + LFU heuristic)
env/          Edge nodes, network topology, request generator, Gymnasium envs
experiments/  Training, evaluation, SHAP analysis, and sweep scripts
pretrained_models/  Shareable Exp1 checkpoints (best_model.zip + config per run)
results/      Checkpoints (runs/), figures, and experiment CSVs/JSON
tests/        pytest suite (109 tests)
run_paths.py  Canonical paths for training artifacts
```

## Citation

This project extends prior work from Professor Genya Ishigaki's Interconnect Lab at SJSU:

- Jayaram et al., "Explainable DRL for Edge Container Caching," IEEE Globecom 2023
- Chen et al., "Multi-Agent Scaling for Edge Caching," IEEE ICCCN 2024
- Shah et al., "Communication-Efficient State Sharing in NDN," IEEE ICCCN 2026
