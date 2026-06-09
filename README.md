# edge-cache-rl

Gymnasium environment for edge container caching under Zipf traffic. Week 1 delivers the simulation stack, LRU/LFU baselines, and a validation script.

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
env/           Edge nodes, network, request generator, CachingEnv
agents/        LRU and LFU baseline policies
configs/       default.yaml and load_config()
experiments/   validate_week1.py
tests/         pytest suite
results/       validation plots (generated)
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

Defaults: `K=20` container types, `C=5` cache slots per node, `10` nodes (single-node mode uses node 0).

### Observation (`Box(0, 1, shape=(2K+1,))`)

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

## Run validation

Runs LRU and LFU for 10,000 timesteps and saves a cumulative-reward plot.

```bash
python experiments/validate_week1.py
```

Output plot: `results/week1_baselines.png`

### Baseline results (K=20, C=5, Zipf α=1.0, seed=42)

| Policy | Cache hit rate | Forward rate | Cloud pull rate |
|--------|----------------|--------------|-----------------|
| LRU | 45.1% | 0.0% | 54.9% |
| LFU | 59.1% | 0.0% | 40.9% |

Forward rate is 0% because week 1 runs a single active node; neighbor caches are empty.

## Configuration

Edit `configs/default.yaml` or load programmatically:

```python
from configs import load_config
config = load_config()
```
