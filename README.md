# Communication-Aware Multi-Agent RL for Edge Container Caching

## Overview

Edge nodes with limited storage learn cooperative container caching policies using multi-agent reinforcement learning. The study varies one factor, the amount of information each node's agent has about its neighbors' caches (communication levels L0 through L3), and measures the effect on caching performance.

**Research question.** How does the degree of inter-agent communication affect caching performance, and can selective communication recover the benefits of full communication at lower bandwidth?

**Headline result.** On the 10-node shifting-demand workload (locality 0.3), agents that see neighbor cache contents (L1) beat the LFU heuristic by 4.4% in episode return (4240 vs 4060), cutting cloud fetches from 25.3% to 19.8% of requests. SHAP analysis attributes 50.8% of L1's eviction-decision importance to neighbor cache state, versus 0% at the no-communication baseline, confirming the agent learns to use communication.

## Table of Contents

1. [The caching problem](#1-the-caching-problem)
2. [Environment](#2-environment)
3. [Communication levels](#3-communication-levels)
4. [Design decisions](#4-design-decisions)
5. [Results](#5-results)
6. [Limitations](#6-limitations)
7. [Setup and reproduction](#7-setup-and-reproduction)
8. [Project structure](#8-project-structure)
9. [Related work](#9-related-work)
10. [License](#10-license)

---

## 1. The caching problem

Each edge node holds a cache of C = 5 containers drawn from a catalog of K = 20 types. Requests arrive one per node per timestep, following a Zipf popularity distribution that varies per node and shifts over time. When a request for container *k* arrives at node *i*, one of three things happens:

| Outcome | Condition | Reward |
|---------|-----------|--------|
| Local hit | *k* is cached at node *i* | +1.0 |
| Forward hit | *k* is cached at a same-cluster neighbor | +0.5 |
| Cloud pull | no neighbor has *k* | −1.0 |

The reward tiers encode the cost structure: serving locally is cheapest, forwarding to a neighbor is acceptable, fetching from the cloud is most expensive. The agent manages its cache to maximize local and forward hits and minimize cloud pulls.

Communication matters because of specialization. If every node caches the same popular items, forwarding never helps, since neighbors hold identical contents. The better strategy is for nodes in a cluster to cache different items and forward to each other. A node can only do this if it knows what its neighbors already hold, which is exactly what the communication levels provide.

## 2. Environment

The environment is a custom [Gymnasium](https://gymnasium.farama.org/)-compatible simulator. All parameters live in `configs/default.yaml`.

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `num_nodes` | 10 | Edge nodes, one agent each |
| `num_clusters` | 3 | Node clusters (sizes 4, 3, 3) |
| `cache_capacity` (C) | 5 | Containers per node |
| `num_container_types` (K) | 20 | Catalog size |
| `zipf_alpha` | 1.0 | Popularity skew (top 5 of 20 draw ~64% of requests) |
| `locality_factor` | 0.3 | Per-node demand divergence (0 = identical, 1 = independent) |
| `episode_length` | 1000 | Timesteps per episode |
| `shift_interval` | 500 | Timesteps between demand shifts |
| `observation_window` | 50 | Recent-request history length in the state |

**Requests and return.** One request per node per timestep produces 10,000 scored requests per episode. Reported return is the summed reward across all nodes over an episode, so its scale is in the thousands.

**Per-node demand.** Each node derives its Zipf ranking from a global ranking through a two-stage shuffle: a cluster-level permutation (about 6 rank swaps at locality 0.3) shared across a cluster, plus a smaller per-node permutation (about 2 swaps). Nodes within a cluster get similar but not identical demand; nodes in different clusters get more distinct demand. At locality 0.3, mean top-5 popularity overlap is about **0.73 within clusters** and **0.41–0.56 across clusters** (seed-dependent). This heterogeneity is what makes intra-cluster forwarding worthwhile.

**Demand shifts.** Every 500 timesteps the global container-to-popularity mapping is reshuffled and per-node rankings are rematerialized. Under this non-stationarity, a frequency-counting heuristic like LFU carries stale counts after a shift, while a learning agent watching its recent-request window can adapt.

**Scope of the model.** The environment uses fixed-size slot-based caching, categorical forwarding cost through the reward tiers rather than a per-link latency graph, and same-cluster-only forwarding. These choices isolate the caching-policy decision from confounds such as bin-packing and latency-graph effects, and they match the abstraction used in the prior lab work this project extends (Jayaram et al. 2023, Chen et al. 2024). Relaxing them is future work.

## 3. Communication levels

Every agent observation includes a 142-dimensional local state (for K = 20, C = 5):

| Block | Size | Encoding |
|-------|------|----------|
| Cache slots | C×K = 100 | LRU-ordered one-hot per slot |
| Utilization | 1 | fraction of cache in use |
| Request frequencies | K = 20 | normalized counts over the last 50 steps |
| Pending request | K = 20 | one-hot of the requested container |
| `needs_decision` flag | 1 | 1.0 on a full-cache miss requiring an eviction choice |

The communication level sets what is appended to this local state:

| Level | Obs dim | Appended information |
|-------|---------|---------------------|
| L0 (none) | 142 | nothing; local state only |
| L1 (cache summaries) | 202 | each same-cluster neighbor's K-bit cache-presence vector, node-id ordered, zero-padded to max cluster degree 3 |
| L2 (full neighbor state) | 568 | each neighbor's entire 142-dim local vector |
| L3 (selective) | 202 | L1 layout, but neighbor information is received only when the agent is uncertain |

L3 measures uncertainty by the Q-margin, the gap between the best and second-best action's Q-value computed on the local-only observation. When the margin is below `selective_comm_threshold` (0.1), the agent requests neighbor state; otherwise it acts on local information. The intent is to approach L1's performance while communicating less often.

**Action space.** Actions are `Discrete(C+1) = 6`. A decision is required only on a full-cache miss (`needs_decision = 1`): actions 0 through 4 evict the corresponding LRU-ordered slot and insert the requested container, and action 5 rejects the insert. On a local hit the reward is scored and the action is ignored. On a miss with spare capacity the container is inserted automatically with no decision. This request-first formulation keeps the action space small and matches how the LRU and LFU baselines operate.

## 4. Design decisions

**Shared policy, not independent agents.** All nodes share one DQN trained on pooled transitions (one replay transition written per node per env step). Sharing lets the policy learn communication features from the combined experience of nodes with different demand profiles. An independent-DQN path (`train-idqn` / `compare_idqn.py`) is retained as an ablation; **all reported Exp1 numbers use the shared policy.**

**Eviction-only action space.** An earlier version allowed caching or evicting any of K containers at any timestep, a 41-action space that was mostly irrelevant choices. Restricting decisions to forced full-cache misses (6 actions) was the change that let a learned policy exceed the heuristics.

**Symmetric information for the comparison.** The agent sees the pending request in its observation, and the heuristics act on the same request. Neither side has an information edge. An earlier protocol gave the heuristics a reactive-oracle advantage; that was corrected.

**Same-cluster forwarding and communication.** Forwarding and cache sharing occur within a cluster only, modeling locality among nearby nodes. Cross-cluster forwarding is future work.

**No overlap penalty in final runs.** A penalty for caching an item a neighbor already holds was tested to force specialization. It is disabled in the final results (`overlap_penalty_weight: 0.0`) because, once per-node demand heterogeneity was in place, agents specialized on their own where it helped. The code path remains for reproducibility.

**Best-checkpoint selection.** Models are evaluated every 20,000 steps (mean return over 5 episodes across seeds 42, 0, 7) and the best is saved. This matters because these policies can degrade after their peak (see [Limitations](#6-limitations)).

## 5. Results

### Experiment 1: communication levels (10 nodes, shifting, locality 0.3)

Three seeds, 20 episodes each, mean ± std, best-checkpoint models.

**Comms/ep:** for L1 and L2 this is the always-on bandwidth budget (`num_nodes × episode_length` = 10,000 neighbor-obs slots per episode). For L3 it is the count of selective communication events (`record_communication`). L0 and baselines send no neighbor state.

| Policy | Return | Hit | Fwd | Cloud | Comms/ep |
|--------|--------|-----|-----|-------|----------|
| **Shared-L1** | **4240 ± 151** | 44.1% | 36.1% | **19.8%** | 10,000 |
| LFU | 4060 ± 73 | **56.9%** | 17.8% | 25.3% | — |
| Shared-L0 | 3783 ± 108 | 49.2% | 26.2% | 24.5% | 0 |
| LRU | 3717 ± 92 | 48.2% | 27.2% | 24.6% | — |
| Shared-L2 | 3699 ± 62 | 47.6% | 27.9% | 24.5% | 10,000 |
| Shared-L3 | 3511 ± 9 | 44.2% | 31.1% | 24.7% | 3,318 |

L1 achieves the highest return with the second-lowest local hit rate. It gives up local hits (44.1% against LFU's 56.9%) to raise forwarding (36.1% against 17.8%) and lower cloud pulls (19.8% against 25.3%). This is the specialization effect. L2 carries more neighbor information than L1 but scores lower, indicating the extra state is noise for the eviction decision. L3's Q-margin gating did not reproduce L1's gains under this action space; see [Limitations](#6-limitations).

### SHAP feature importance: eviction decisions (L0 vs L1)

Share of mean |SHAP| on the eviction Q-value, by feature group (from `results/data/week8/shap_l0_vs_l1.json`).

| Feature group | L0 | L1 |
|---------------|-----|-----|
| Local cache bits | 53.4% | 25.3% |
| Local request freqs | 22.2% | 4.7% |
| Pending request (one-hot) | 22.4% | 17.6% |
| Neighbor cache bits | — | 50.8% |

L0 decides almost entirely on local signals. At L1, neighbor cache bits account for half of eviction-decision importance, with pending request also material; the agent uses communication rather than ignoring the extra inputs.

## 6. Limitations

**The RL margin is modest and formulation-dependent.** L1 beats LFU by 4.4%, and only under the eviction-only action space. The earlier wide action space lost to the heuristics.

**Policies degrade after their peak.** Best-checkpoint and final-checkpoint returns differ (Shared-L1 best 4240, final 3806; data in `results/data/fair_eval_evict/exp1_summary.csv`). The policies do not converge to a stable optimum within the 400k-step budget. Reported numbers are best-checkpoint, and this dependence on early stopping is a real caveat.

**L3 selective communication underperformed.** With threshold 0.1, L3 communicates on about one-third of always-on slots (~3,318 of 10,000) but still trails L1 (3511 vs 4240). Intermittent neighbor access appears insufficient for the coordination L1 achieves with always-on cache summaries.

**Uniform demand shrinks the gap.** At `locality_factor=0`, all nodes share identical demand rankings, so neighbor cache summaries are redundant in principle; empirically the L1−L0 return gap is smallest there (~255 points vs ~457 at 0.3 in `results/data/week7/locality_summary.csv`). The headline benefit grows with heterogeneity and L1 peaks near locality 0.3 in that sweep.

**Other experiments are not headlined.** Scalability (5 to 50 nodes), traffic-pattern, and locality sweeps were run; summaries are in `results/data/week7/`. They are left out of the main results to keep the central claim focused on Experiment 1.

## 7. Setup and reproduction

```bash
git clone https://github.com/naoxcv/edge-cache-rl.git
cd edge-cache-rl
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -v           # 110 tests
```

Requires Python 3.9+. Tested on macOS ARM64.

**Pretrained models.** Training all four levels from scratch takes several hours. To skip to evaluation, download `edge-cache-rl-pretrained-exp1.zip` from [GitHub Releases](https://github.com/naoxcv/edge-cache-rl/releases) and unzip at the repo root so `pretrained_models/` sits beside `experiments/` and `configs/`. See `pretrained_models/README.md`.

**Reproduce the key result:**

```bash
python experiments/compare_comm_levels.py --config configs/default.yaml
```

Evaluates LRU, LFU, and the four shared-policy DQN checkpoints on 10 nodes, shifting traffic, locality 0.3, across seeds 42, 0, 7 (20 episodes each).

**SHAP explainability:**

```bash
python experiments/shap_analysis.py --config configs/default.yaml
```

Writes `results/data/week8/shap_l0_vs_l1.json` and figures under `results/figures/`.

**Train from scratch.** Final models use 400,000 timesteps, `learning_rate=1e-4`, `buffer_size=100,000`, `learning_starts=5000`, `batch_size=128`, `train_freq=4`, `gradient_steps=10`, epsilon annealed 1.0 to 0.01 over the first 30% of training, network `[128, 128, 64]`.

```bash
python experiments/train_multi.py train \
  --config configs/default.yaml \
  --comm-level 1 \
  --timesteps 400000 \
  --seed 42 \
  --run-name dqn_evict_level1_scratch_loc0.3
```

**Demo dashboard:**

```bash
cd dashboard/frontend && npm install && npm run build && cd ../..
python -m dashboard.server   # http://127.0.0.1:8000
```

Toggle between L0, L1, L3 DQN and the LFU heuristic; watch cumulative episode return, hit/forward/cloud rates, and cache state on the 10-node network.

## 8. Project structure

```
agents/       Shared-policy DQN, LRU/LFU baselines, IDQN ablation, training helpers
configs/      default.yaml: network, traffic, reward, and training hyperparameters
dashboard/    FastAPI + React live demo
env/          Edge nodes, network topology, request generator, Gymnasium envs
experiments/  Training, evaluation, SHAP, and sweep scripts
pretrained_models/  Exp1 checkpoints (unzip release zip here; *.zip gitignored)
results/      Published summaries, SHAP figures; local training runs under runs/ (gitignored)
tests/        pytest suite (110 tests)
run_paths.py  Resolve checkpoints from runs/ or pretrained_models/
```

## 9. Related work

This project extends work from Professor Genya Ishigaki's interconnect lab at SJSU:

- Jayaram, Jeelani, and Ishigaki, "Container Caching Optimization based on Explainable Deep Reinforcement Learning," IEEE Globecom 2023. Single-agent explainable DRL for container caching.
- Chen and Ishigaki, "Scaling Container Caching to Larger Networks with Multi-Agent Reinforcement Learning," IEEE ICCCN 2024. Extends caching to multi-node networks and raises the centralized-versus-decentralized question.
- Shah, Yanamandra, Ramesh, and Ishigaki, "An Intelligent Content Caching for NDN with Communication-Efficient State Sharing," IEEE ICCCN 2026. Introduces communication-efficient state sharing, the direct predecessor to the communication-level question studied here.

Related multi-agent RL: CommNet (Sukhbaatar et al., NeurIPS 2016), TarMAC (Das et al., ICML 2019).

## 10. License

MIT