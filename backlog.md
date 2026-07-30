# Project Context: Communication-Aware Multi-Agent RL for Edge Container Caching

## Purpose of This Document

This is a handoff document for an AI coding assistant. The user is a 4th-year CS undergrad at SJSU building this project over 9 weeks (summer 2026) to join Professor Genya Ishigaki's interconnect lab. The user has read about RL but has not implemented it. Guide them through implementation step by step.

---

## What the Project Is

An RL system where multiple edge computing nodes each run an independent DQN agent that decides which software containers to keep cached locally. The core research question: how does varying the degree of inter-agent communication affect caching performance? The user will test four communication levels (none, neighbor summary, full neighbor state, selective/uncertainty-triggered) and measure the tradeoff between coordination quality and communication overhead.

This directly extends prior work in Ishigaki's lab: Jayaram (Globecom 2023, single-agent explainable DRL for caching), Chen (ICCCN 2024, multi-agent scaling), Nian (AIoT 2025, Knative autoscaling with DRL), and Shah (ICCCN 2026, communication-efficient state sharing in NDN).

---

## Architecture

```
project/
├── env/
│   ├── container.py
│   ├── edge_node.py
│   ├── edge_network.py
│   ├── request_generator.py
│   ├── rewards.py           # Shared local/forward/cloud scoring
│   ├── caching_env.py       # Single-node Gymnasium env
│   ├── multi_agent_caching_env.py
│   └── wrappers.py          # RandomTrafficSeedWrapper
├── agents/
│   ├── baselines.py         # LRU / LFU (reactive)
│   ├── callbacks.py         # MultiSeedEvalCallback (single-agent)
│   ├── single_agent.py      # SB3 DQN
│   └── multi_agent.py       # Shared-policy SB3 DQN + warm-start
├── experiments/
│   ├── train_single.py
│   ├── compare_policies.py
│   ├── train_multi.py       # verify + train Level 0+
│   ├── compare_multi.py
│   ├── validate_week1.py
│   └── validate_traffic_patterns.py
│   # Week 7+: exp1–4 communication / scalability scripts (not yet)
├── configs/default.yaml     # Defaults: shifting traffic, forwarding ON
├── run_paths.py
├── results/runs/<name>/     # Canonical Level 0: dqn_multi_level0
├── tests/
├── backlog.md
└── README.md
```

---

## RL Formulation

**State per node:** binary vector of cached containers (length K=20), cache utilization (float 0-1), request frequency per container over last 50 timesteps (length K vector). For communication levels 1-3, neighbor state info is appended.

**Actions (discrete):** cache container k (evicting LRU if full), evict container k, or no-op. No-op exists because the current cache may already be optimal — forcing an action every step causes unnecessary churn.

**Reward:** +1.0 local cache hit, +0.5 forwarding hit (neighbor serves it), −1.0 cloud pull/cold start. Optional communication penalty: −λ · m_i(t).

**Episode:** 1,000 timesteps of request arrivals.

---

## Communication Levels (the independent variable)

- **Level 0:** No communication. Each agent sees only local state. Baseline.
- **Level 1:** Neighbor summary. Each agent receives a binary vector of what its immediate neighbors have cached. Low bandwidth.
- **Level 2:** Full neighbor state. Each agent receives neighbors' full state vectors (cache contents + utilization + request frequencies). Higher bandwidth.
- **Level 3:** Selective communication. Agents broadcast state only when their Q-value margin between best and second-best action falls below a threshold (high uncertainty). Most novel — agents learn *when* to communicate.

---

## Container Catalog

20 container types, each defined by `id` and `popularity_rank` only. No image size or startup latency — cache is slot-based (C=5 slots per node, one container per slot). This simplification was chosen to isolate the caching decision (which container to keep) from the bin-packing problem (how many fit). Request popularity follows Zipf distribution (α=1.0). Top 5 containers capture ~64% of requests under stationary traffic.

---

## Network Topology

Hierarchical clustering. Default: 3 clusters of 3-4 nodes each. No latency matrix — latency is a categorical lookup via `cluster_map` dict:
- Same cluster: intra-cluster cost (2ms)
- Different cluster: inter-cluster cost (20ms)
- Cloud fallback: 100ms

No NetworkX dependency. Topology is a `cluster_map` dict and an adjacency list. Neighbors are all nodes in the same cluster. `find_any_neighbor_with(container_id)` checks if any neighbor has a container cached — no shortest-path computation since intra-cluster latency is flat.

---

## Tech Stack

| Phase | Tools |
|-------|-------|
| Weeks 1-3 (single node) | Python 3.9, Gymnasium, NumPy, PyYAML, pytest, matplotlib |
| Week 2-3 (single agent RL) | Stable-Baselines3, PyTorch |
| Weeks 4-6 (multi-agent) | Stable-Baselines3 (shared-policy DQN; revisit RLlib only if Level 3 needs it) |
| Week 7 (experiment tracking) | Weights & Biases or TensorBoard |
| Week 8 (explainability) | SHAP |
| Week 8 (dashboard) | React, FastAPI |
| Week 9 (stretch: cloud deploy) | GKE, Knative |

---

## Progress So Far

**Week 1 — Environment (DONE)**
Built container catalog (Option A: id + popularity_rank only, slot-based cache), EdgeNode, EdgeNetwork (cluster_map, no latency matrix), Gymnasium wrapper, request generator with Zipf traffic. Implemented LRU/LFU baselines. LRU is consistently weak/negative under Zipf α=1.0 with K=20, C=5 — this is expected, not a bug. LRU's recency-only eviction can't hold the hot set against tail churn.

**Week 2 — Single-node DQN (DONE)**
Trained DQN via Stable-Baselines3. Under stationary traffic: DQN roughly matches LFU (mean DQN vs LFU = +2.7, std = 47.2 across 3 seeds). Expected — LFU is near-optimal for stationary Zipf by definition. Exploration rate was initially set too low (SB3 default 0.1); increasing to 1.0 with decay helped.

**Week 3 — Non-stationary traffic (DONE)**
Implemented shifting (popularity permutation at shift_interval=500) and bursty (burst_probability=0.05, burst_multiplier=10) traffic. Key result: shifting-trained DQN beats LFU under shifting traffic (mean +30.7 vs LFU). DQN adapts to popularity changes faster than LFU's frequency window. Bursty-trained DQN shows high variance across seeds (mean -7.3, std 44.9 vs LFU) — inconsistent but not failing. Single-node RL advantage is validated.

---

## Week 4 — Multi-Node Environment + Level 0 Multi-Agent

This is the transition from single-node to multi-node. The goal is a shared-policy SB3 DQN across a real multi-node network with forwarding, all under Level 0 (no communication). This is the baseline that weeks 5-6 improve upon.

**Note:** An RLlib Independent DQN attempt was abandoned — train reward looked OK but deterministic eval collapsed vs LRU/LFU. Week 4 continues with SB3 (same hyperparams / multi-seed eval / early stopping as single-agent).

### Backlog

**Environment changes:**
- [x] Modify / add multi-node env to step all N nodes per timestep (`env/multi_agent_caching_env.py`); keep single-node `CachingEnv` intact
- [x] Each node receives its own request from the request generator
- [x] Implement same-cluster-only forwarding via `find_any_neighbor_with(..., same_cluster_only=True)`
- [x] Decide: same-cluster-only for now (deeper forwarding policy deferred)
- [x] Return per-node observations/rewards/dones as dicts keyed by node_id
- [x] Verify: random actions on 3 and 10 nodes — forwarding nonzero (~33–39%)

**SB3 multi-agent (shared policy):**
- [x] Drop RLlib from requirements; rewrite `agents/multi_agent.py` for shared SB3 DQN
- [x] Custom train loop: 1 env step → N node transitions into one replay buffer
- [x] Multi-seed deterministic eval + early stopping + `best_model.zip` / `model.zip`
- [x] Smoke-train on 3 nodes / 1 cluster (verified)

**Training:**
- [x] Level 0 baseline = warm-start from single-node `dqn_shifting` (zero-shot transfer); from-scratch multi train failed; fine-tune did not beat zero-shot
- [x] Eval on 3 nodes / 1 cluster and 10 nodes / 3 clusters (shifting, 3 seeds)
- [ ] Optional later: longer multi-node fine-tune with low ε (not blocking Week 5)

**Baselines for multi-node:**
- [x] Extend LRU and LFU to run independently per node (`experiments/compare_multi.py`)
- [x] Run LRU and LFU on 10 nodes / shifting traffic / 3 seeds (full eval)
- [x] Record per-node and network-wide hit / forward / cloud rates

**Evaluation:**
- [x] Compare multi-agent DQN (Level 0) vs LRU vs LFU on 10 nodes / shifting / 3 seeds
  - 3-node: DQN vs LFU mean **−242 ± 128** (hit ~49–53%, fwd ~11–15%)
  - 10-node: DQN vs LFU mean **−427 ± 284** (seed 42 nearly ties LFU: −30); LRU still wins on network return via more forwarding
- [x] Key question: does forwarding help? (with vs without)
  - Yes, strongly. Same local hit rates; return jumps because neighbor hits replace cloud pulls (+0.5 vs −1.0).
  - 3-node mean Δ(ON−OFF): LRU +1192, LFU +626, DQN +595
  - 10-node mean Δ(ON−OFF): LRU +4203, LFU +2322, DQN +2361
  - LRU benefits most (higher cache diversity → more forwards)
- [x] Key question: cache redundancy across nodes — diversity printed; Level 0 still has overlapping popular items (motivates Week 5)
- [ ] Plot per-node reward distribution

**What success looks like for week 4:**
- [~] Multi-agent DQN (Level 0) roughly matches LFU on hit rate / is in the same ballpark on return (act-then-request vs reactive oracle gap remains)
- [x] Forwarding hits are nonzero
- [x] Observable redundancy / diversity gap vs LRU motivates communication levels
- Training from scratch on multi-node deferred; warm-start is the Level 0 baseline (`results/runs/dqn_multi_level0/`)

**What failure looks like and how to fix it:**
- SB3 shared policy won't learn: simplify to 3 nodes, single cluster, stationary traffic. Confirm deterministic eval (not train reward) is rising.
- Forwarding never happens: check that nodes in the same cluster actually have different request patterns. If all nodes see identical Zipf distributions, they'll all cache the same containers and forwarding is useless. Verify the request generator assigns different request streams per node.
- Training is too slow: reduce to 5 nodes, shorter episodes (500 steps), smaller replay buffer.
- Note: under multi-node + forwarding, LRU can beat LFU on *network* return because LRU churn increases cache diversity → more forward hits. Compare both hit rate and forward rate, not just ep return.

---

**Week 5 — Levels 1 and 2.**
Implement neighbor summary and full neighbor state communication. Append received info to each agent's observation vector. Compare against Level 0.

**Week 6 — Level 3 + Experiment 1.**
Implement selective communication (uncertainty-triggered). Run the communication-vs-performance tradeoff experiment at 10 nodes.

**Week 7 — Experiments 2-4.**
Scalability (5-50 nodes), traffic robustness (all levels × 3 traffic regimes), communication cost sensitivity (vary λ).

**Week 8 — SHAP analysis + dashboard.**
Explainability on best-performing config. Build React/FastAPI visualization.

**Week 9 — Cleanup, stretch goals.**
Code cleanup, documentation, final results. Optional GKE deployment.

---

## Key Implementation Guidance

- **Environment bugs are the #1 risk.** If RL results look wrong, check the environment first. Verify rewards compute correctly, observations update correctly, cache state is consistent.
- **Use Stable-Baselines3 DQN for single-agent and multi-agent (shared policy for now).** Don't write custom RL algorithms. The contribution is the environment and experimental analysis, not a novel algorithm.
- **SB3 default exploration is too low.** Set `exploration_initial_eps=1.0` and `exploration_fraction=0.1`. The default 0.1 start barely explores.
- **Train on the evaluation distribution.** A stationary-trained DQN underperforms on shifting traffic. Always retrain on the target traffic pattern.
- **LRU is a weak baseline under Zipf.** Expect negative returns. This is correct behavior — LRU can't hold the hot set with K=20, C=5, α=1.0. LFU is the real baseline to beat.
- **DQN matches LFU on stationary, beats it on shifting.** The RL advantage is adaptation speed after popularity changes. Don't expect big wins under stationary Zipf — LFU is near-optimal there by definition.
- **Evaluate across multiple seeds (minimum 3).** High variance across seeds is normal. Report mean and std. One bad seed doesn't invalidate the approach.
- **For multi-agent (SB3 Level 0):** shared-policy DQN — one network, all nodes' transitions in one buffer. Communication levels later expand the observation space, not the training algorithm. Revisit independent policies / RLlib only if Level 3 Q-margin needs it.
- **State representation matters more than algorithm choice.** If the agent can't learn, the observation probably doesn't contain enough information or contains too much noise. Debug by simplifying: fewer containers, smaller cache, static traffic.
- **Always compare against baselines.** Every result should be shown relative to LRU, LFU, and (once available) centralized DQN. Absolute numbers mean nothing without context.

---

## What This Project Is For

The user plans to email Professor Ishigaki in mid-August with results from this project and request a position in his lab. The deliverables that matter: a working codebase with reproducible results, a clear demonstration of the communication-performance tradeoff, and at minimum one novel finding (likely around Level 3 selective communication). This is scoped to seed a conference paper submission (IEEE ICCCN, Globecom, or ICC) during Fall 2026.