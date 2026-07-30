# edge-cache-rl

Gymnasium edge-caching RL under Zipf traffic (stationary / shifting / bursty).
Single-node SB3 DQN through Week 3; Week 4 multi-node Level 0 uses a **shared-policy**
SB3 DQN with same-cluster forwarding.

## Setup

```bash
git clone <repo-url>
cd edge-cache-rl
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Project layout

```
env/           Edge nodes, network, request generator, CachingEnv, MultiAgentCachingEnv
agents/        LRU/LFU, single-agent SB3 DQN, multi-agent shared-policy SB3 DQN
configs/       default.yaml (defaults to shifting traffic)
experiments/   train/compare/validate scripts (single + multi)
tests/         pytest suite
results/       figures/ and runs/<name>/ (model zips gitignored; see results/README.md)
```

## Quick start

```python
from configs import load_config
from env.caching_env import CachingEnv

config = load_config()
env = CachingEnv(config, seed=42)
obs, info = env.reset()

noop = 2 * config["num_container_types"]
obs, reward, terminated, truncated, info = env.step(noop)
env.render()
```

## Environment interface

Defaults: `K=20` container types, `C=5` cache slots per node, `10` nodes / `3` clusters
(single-node mode uses node 0). Default traffic is **shifting**.

### Observation (`Box(0, 1, shape=(2K+1,))`) — Level 0

| Slice | Size | Meaning |
|-------|------|---------|
| `obs[0:K]` | K | Cache binary — 1.0 if container `i` is cached |
| `obs[K]` | 1 | Cache utilization (`len(cache) / C`) |
| `obs[K+1:2K+1]` | K | Request frequency over the observation window, normalized to [0, 1] |

### Action (`Discrete(2K+1)`)

| Action | Effect |
|--------|--------|
| `0 .. K-1` | Cache container `k` (evicts oldest if full) |
| `K .. 2K-1` | Evict container `k` |
| `2K` | No-op |

### Reward

| Outcome | Reward |
|---------|--------|
| Local cache hit | +1.0 |
| Forward hit (neighbor has container) | +0.5 |
| Cloud pull | -1.0 |
| No request | 0.0 |

Forwarding is controlled by `enable_forwarding` and `forwarding_same_cluster_only`
(both default true). Scoring is shared via `env/rewards.py`.

### Step order (RL)

Each `step(action)` runs: **apply action → generate request → score reward → update history**.

The agent does not see the current request before acting; it must infer demand from the observation history.

### `info` dict

- `cache_hit_rate` — local hits / total requests so far on the active node
- `timestep` — steps elapsed in the current episode
- `requested` — container ID requested this step (for debugging)

## Baselines

`agents/baselines.py` provides oracle LRU and LFU policies. They take the upcoming request as input and use reactive caching: **score request first, then update cache**.

```python
from agents.baselines import LRUPolicy, LFUPolicy

policy = LRUPolicy()
action = policy.act(observation, requested, cache=node.cache)
```

## Run tests

```bash
pytest tests/ -v
```

## Train DQN (single-node)

```bash
python experiments/train_single.py --timesteps 1000000 --run-name dqn_generalized
python experiments/compare_policies.py --dqn-path dqn_generalized --seeds 42,0,7
```

Artifacts: `results/runs/<run_name>/` (`best_model.zip`, `model.zip`, `evaluations.npz`, `tensorboard/`).

## Multi-node (Week 4, Level 0)

`MultiAgentCachingEnv` steps all N nodes each timestep. Level 0 baseline is a **shared-policy**
SB3 DQN warm-started from single-node `dqn_shifting` (zero-shot transfer; from-scratch multi
train underperformed). Canonical run: `results/runs/dqn_multi_level0/`.

```bash
# Verify forwarding with random actions
python experiments/train_multi.py verify --nodes 3 --clusters 1 --traffic shifting
python experiments/train_multi.py verify --nodes 10 --clusters 3 --traffic shifting

# Multi-node LRU/LFU (+ optional DQN)
python experiments/compare_multi.py --nodes 10 --clusters 3 --episodes 20 --seeds 42,0,7
python experiments/compare_multi.py --nodes 3 --clusters 1 --dqn-path dqn_multi_level0 --seeds 42,0,7

# Ablate forwarding
python experiments/compare_multi.py --nodes 3 --clusters 1 --no-forwarding --no-dqn

# Warm-start Level 0 from single-node shifting DQN
python experiments/train_multi.py train \
  --nodes 3 --clusters 1 --timesteps 100000 \
  --run-name dqn_multi_level0 --traffic shifting \
  --pretrained dqn_shifting
```

## Validation plots

```bash
python experiments/validate_week1.py
python experiments/validate_traffic_patterns.py
```

Figures land in `results/figures/`.

### Week-1 single-node baseline sketch (K=20, C=5, Zipf α=1.0, seed=42)

| Policy | Cache hit rate | Forward rate | Cloud pull rate |
|--------|----------------|--------------|-----------------|
| LRU | 45.1% | 0.0% | 54.9% |
| LFU | 59.1% | 0.0% | 40.9% |

Forward rate is 0% in that table because week 1 ran a single active node with empty neighbor caches.

## Configuration

Edit `configs/default.yaml` or load programmatically:

```python
from configs import load_config
config = load_config()
```
