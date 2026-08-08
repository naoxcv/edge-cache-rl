# Communication-Aware Multi-Agent RL for Edge Container Caching

Multi-agent reinforcement learning system where edge computing nodes learn cooperative
container caching policies under four communication levels. Each node runs a shared-policy
DQN that decides which containers to cache locally. The research question: **how does
varying inter-agent communication affect caching performance, and can selective
communication recover full-coordination benefits at a fraction of the bandwidth?**

Key finding: Level 1 (neighbor cache summaries) beats Level 0 (no communication) by +272
episode return under heterogeneous demand (locality=0.3). Level 3 (Q-margin selective)
matches Level 1 within -1% while communicating on only 7.3% of decisions.

## Setup

```bash
git clone <repo-url>
cd edge-cache-rl
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -v           # verify: 101 tests pass
```

Requires Python 3.9+. Tested on macOS ARM64.

## Reproducing key results

### Train L0-L3 from scratch (canonical 10-node, shifting, locality=0.3)

```bash
for LEVEL in 0 1 2 3; do
  python experiments/train_multi.py train \
    --nodes 10 --clusters 3 --traffic shifting --locality-factor 0.3 \
    --comm-level "$LEVEL" --overlap-penalty 0 --timesteps 400000 \
    --run-name "dqn_multi_level${LEVEL}_scratch_loc0.3"
done
```

### Compare communication levels

```bash
python experiments/compare_comm_levels.py \
  --nodes 10 --clusters 3 --locality-factor 0.3 --seeds 42,0,7 --levels 0,1,2,3 \
  --dqn-paths dqn_multi_level0_scratch_loc0.3,dqn_multi_level1_scratch_loc0.3,dqn_multi_level2_scratch_loc0.3,dqn_multi_level3_scratch_loc0.3
```

### Run full Week-7 experiment matrix

```bash
./experiments/week7_run_all.sh
python experiments/week7_sweep.py all --no-train --seeds 42,0,7 --episodes 20
```

### SHAP explainability (L0 vs L1)

```bash
python experiments/shap_analysis.py
```

### Demo dashboard

```bash
cd dashboard/frontend && npm install && npm run build && cd ../..
python -m dashboard.server   # http://127.0.0.1:8000
```

## Main results (3 seeds x 20 episodes, mean +/- std)

### Experiment 1 -- Communication levels (10 nodes, shifting, locality=0.3)

| Level | Return | Hit | Fwd | Cloud | Comms |
|-------|--------|-----|-----|-------|-------|
| L0 | 2536+/-399 | 47.6% | 20.1% | 32.3% | 0 |
| L1 | **2808+/-102** | 44.1% | 26.6% | 29.3% | 10,000 (always) |
| L2 | 2299+/-161 | 40.9% | 27.5% | 31.6% | 10,000 (always) |
| L3 | 2779+/-259 | 50.3% | 18.2% | 31.6% | 735 (7.3%) |

### Experiment 2 -- Scalability (per-node return, shifting, locality=0.3)

| N | L0/node | L1/node | L2/node | L3/node | L3 comms |
|---|---------|---------|---------|---------|----------|
| 5 | 214+/-27 | **264+/-34** | 143+/-42 | 261+/-15 | 6.2% |
| 10 | 254+/-40 | **281+/-10** | 230+/-16 | 278+/-26 | 7.3% |
| 25 | **335+/-32** | 323+/-48 | 180+/-40 | 324+/-9 | 8.7% |
| 50 | 347+/-1 | **393+/-14** | 207+/-53 | 374+/-5 | 8.6% |

### Experiment 3 -- Traffic patterns (10 nodes, locality=0.3)

| Traffic | L0 | L1 | L2 | L3 | L3 comms |
|---------|----|----|----|----|----------|
| stationary | 2726+/-317 | 2932+/-129 | 2067+/-329 | **3157+/-20** | 6.4% |
| shifting | 2536+/-399 | **2808+/-102** | 2299+/-161 | 2779+/-259 | 7.3% |
| bursty | 2587+/-281 | **2801+/-83** | 1958+/-426 | 2596+/-115 | 9.1% |

### SHAP feature importance (eviction Q, locality=0.3)

| Feature group | L0 share | L1 share |
|---------------|----------|----------|
| Local cache bits | **60.4%** | 25.6% |
| Local request freqs | 37.3% | 22.8% |
| Neighbor cache bits | -- | **51.5%** |

## Project structure

```
env/                Edge nodes, network topology, request generator, Gymnasium envs
agents/             LRU/LFU baselines, single-agent DQN, shared-policy multi-agent DQN
configs/            default.yaml with all tunable parameters
experiments/        Training, evaluation, SHAP analysis, and sweep scripts
dashboard/          FastAPI + React live demo (L0/L1/L3 toggle)
tests/              pytest suite (101 tests)
results/
  figures/          Static plots (SHAP, traffic validation, baselines)
  data/             Experiment CSVs and JSON summaries (week7/, week8/)
  runs/             Per-run checkpoints, configs, eval curves (models gitignored)
run_paths.py        Canonical paths for training artifacts
```

## Citation

This project extends prior work from Professor Genya Ishigaki's Interconnect Lab at SJSU:

- Jayaram et al., "Explainable DRL for Edge Container Caching," IEEE Globecom 2023
- Chen et al., "Multi-Agent Scaling for Edge Caching," IEEE ICCCN 2024
- Shah et al., "Communication-Efficient State Sharing in NDN," IEEE ICCCN 2026
