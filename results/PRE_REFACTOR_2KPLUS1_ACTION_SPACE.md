# Pre-refactor snapshot: 2K+1 action space

**Frozen:** 15 August 2026  
**Purpose:** Record environment, experiments, and findings **before** any action-space refactor. All numbers below were produced with `Discrete(2K+1)` per node.

Do not mix these tables with later runs that change the action encoding.

---

## 1. Environment that produced the results

Canonical config: `configs/default.yaml` unless a sweep overrides N, traffic, or `locality_factor`.

### 1.1 Network and catalog

| Parameter | Value |
|-----------|--------|
| Nodes | 10 (canonical); sweeps use 5 / 10 / 25 / 50 |
| Clusters | 3 (canonical); ~3–5 nodes per cluster (`5→2`, `25→5`, `50→10`) |
| Intra / inter / cloud latency | 2 / 20 / 100 ms (used for scoring path, not as a delay in the Gym step) |
| Container types **K** | 20 |
| Cache capacity **C** | 5 slots per node |
| Forwarding | On, **same-cluster only** |
| Bridge edges | Present in the graph for topology, **not** in L1–L3 observations and **not** used for forwarding |

### 1.2 Traffic

| Parameter | Value |
|-----------|--------|
| Demand | Zipf α = 1.0, one request per node per timestep |
| `locality_factor` | 0.3 canonical (0 = identical rankings, 1 = independent per node) |
| Cluster structure | Same-cluster nodes share more demand similarity than different clusters |
| Patterns | **stationary**, **shifting** (`shift_interval=500`), **bursty** (p=0.05, ×10) |
| Episode length | 1000 steps |
| Observation window | last 50 requests for frequency features |

### 1.3 MDP / step order (critical)

`MultiAgentCachingEnv.step` is **act-then-request** (blind):

1. All nodes choose actions from the **previous** observation (no current request ID in the obs).
2. Cache mutations apply **simultaneously**.
3. One request is generated per node.
4. Reward is scored against the **post-action** caches (local hit / same-cluster forward / cloud).
5. Request histories update; next observation is built.

Heuristics used for the fair tables are **blind delayed-reactive**: they act on the **previous** request, then `env.step` scores the current request — same information horizon as DQN. (Oracle LRU that scores-then-caches is isomorphic at scoring time; empirically LRU return is identical: 3717.3.)

### 1.4 Action space — 2K+1 (this snapshot)

Per node: `spaces.Discrete(2 * K + 1)` with **K = 20 → 41 actions**.

| Index | Meaning |
|-------|---------|
| `0 … K−1` | Cache container *i* (env evicts LRU slot `cache[0]` if full) |
| `K … 2K−1` | Evict container `i = a − K` |
| `2K` | No-op |

The agent does **not** see the current request when choosing. Most actions are irrelevant to the immediate reward; the policy must cache for **future** demand from history + (optional) neighbor cache bits.

### 1.5 Observation

Local (L0, always present), size **2K+1 = 41**:

- cache occupancy bits (K)
- utilization scalar (1)
- request frequencies over the window (K)

Communication extras use **same-cluster neighbors only**, padded to `max_neighbors` (3 on the 10-node / 3-cluster graph):

| Level | Extra features | Canonical obs dim |
|-------|----------------|-------------------|
| L0 | none | 41 |
| L1 | per-peer cache binary (K each) | 41 + 3×20 = **101** |
| L2 | per-peer full local state (2K+1 each) | 41 + 3×41 = **164** |
| L3 | same layout as L1; neighbor slots zeroed unless Q-margin says communicate | **101** |

L3: communicate when local-only Q-margin < `selective_comm_threshold` (0.01). Events counted even at `comm_penalty_lambda=0`.

### 1.6 Reward (per node, per request)

| Outcome | Reward |
|---------|--------|
| Local hit | +1.0 |
| Same-cluster forward hit | +0.5 |
| Cloud pull | −1.0 |

Network episode return ≈ `10000 × (hit + 0.5·fwd − cloud)` for 10 nodes × 1000 steps.  
Canonical tables use **overlap_penalty_weight = 0** and **comm_penalty_lambda = 0**. Overlap penalty exists as a one-shot cache-action nudge but was not used in the reported sweeps.

---

## 2. Training that produced the checkpoints

| Item | Protocol |
|------|----------|
| Algorithm | Shared-policy DQN (SB3): one network, all nodes’ transitions in one replay |
| Init | **From scratch** (no L0 warm-start; curriculum masking off unless pretrained) |
| Timesteps | 400k (10-node); 300k (5/25); 250k (50) |
| `gradient_steps` | `num_nodes` (restore single-agent update-to-data; 1 would under-update 10×) |
| Exploration | ε 1.0 → 0.01 over `exploration_fraction=0.3` |
| Net | MLP `[128, 128, 64]` |
| Train traffic | `randomize_episode_seeds=true` |
| Checkpoint | `best_model.zip` = mean eval over seeds `{42, 0, 7}` |
| L3 training | Explore with full neighbor features; Q-margin gate on greedy steps only |

L1–L3 Week-7 cells (non-canonical N / traffic / locality) were **retrained** after restricting observations to same-cluster peers (obs width change). Canonical `*_scratch_loc0.3` (10-node shifting) was already matched.

IDQN (one DQN per node) was trained separately at 10-node loc 0.3; it needs `idqn_gradient_steps = num_nodes` (default was 1 and failed).

---

## 3. Evaluation protocol (fair tables)

- 20 episodes × seeds `{42, 0, 7}`, mean ± std  
- Blind LRU / LFU via `experiments/eval_fair_suite.py`  
- Report **best** checkpoints; finals often collapse (see §5)  
- Artifacts: `results/data/fair_eval/` (`all_summary.csv`, `week7_retrained_eval.log`)

---

## 4. Findings

### 4.1 Absolute performance: heuristics dominate

**Canonical Exp1 — 10 nodes, shifting, locality 0.3** (best checkpoints)

| Policy | Return | Hit | Fwd | Cloud | vs LFU |
|--------|--------|-----|-----|-------|--------|
| **LRU (blind)** | **3717 ± 92** | 48.2% | 27.2% | **24.6%** | +158 |
| **LFU (blind)** | **3559 ± 106** | **55.1%** | 16.9% | 28.0% | — |
| Shared-L1 | 2744 ± 90 | 44.7% | 25.4% | 29.9% | −816 |
| Shared-L3 | 2690 ± 113 | 48.7% | 19.6% | 31.6% | −869 |
| Shared-L0 | 2536 ± 399 | 47.6% | 20.1% | 32.3% | −1024 |
| Shared-L2 | 2300 ± 232 | 44.6% | 22.5% | 32.9% | −1260 |
| IDQN-L0 best @120k | 3027 ± 362 | 41.3% | 31.7% | 26.9% | −532 |
| IDQN-L1 best @240k | 2633 ± 113 | 33.2% | 40.0% | 26.8% | −926 |

No trained 2K+1 agent beats LRU or LFU on this protocol. LRU wins on **lowest cloud**, not highest local hit.

The gap is **policy quality**, not “heuristics see the current request for this step’s reward.” Blind vs oracle LRU returns match.

### 4.2 Relative RL: communication still matters

Under the same MDP, among *learned* policies:

- **L1 − L0 = +208** (canonical). L1 trades local hit for more forwarding and less cloud.
- **L3 ≈ L1** (−53) at **687 comms/ep** vs L1 always-on 10 000 (**6.9%**).
- **L2 lags L1** — richer neighbor state, same budget; extra features look like noise for the cache decision.

This relative ranking is the communication result. It is **not** “RL solves edge caching.”

### 4.3 Traffic robustness (10 nodes, loc 0.3, best)

| Traffic | LRU | LFU | L0 | L1 | L2 | L3 | best RL vs LFU |
|---------|-----|-----|----|----|----|----|----------------|
| stationary | 3728 | 3612 | 2726 | **3038** | 2285 | 2951 | −574 |
| shifting | 3717 | 3559 | 2536 | **2744** | 2300 | 2690 | −816 |
| bursty | 3135 | 3180 | 2587 | **2691** | 2280 | 2624 | −489 |

L1 is the best RL row on every pattern. Gap to LFU is smallest under bursty.

### 4.4 Locality sweep (10 nodes, shifting, best)

| loc | LRU | LFU | L0 | L1 | L2 | L3 | best RL vs LFU |
|-----|-----|-----|----|----|----|----|----------------|
| 0.0 | 3846 | 3347 | 1692 | **2780** | 2096 | 2472 | −567 |
| 0.2 | 3783 | 3373 | 2456 | **2902** | 2118 | 2735 | −471 |
| 0.3 | 3717 | 3559 | 2536 | **2744** | 2300 | 2690 | −816 |
| 0.4 | 3663 | 3604 | 2895 | 2869 | 2320 | **3262** | −342 |
| 0.6 | 3578 | 3634 | 3060 | **3383** | 2596 | 3076 | **−251** |
| 0.8 | 3591 | 3711 | 3031 | 2972 | 2786 | **3214** | −497 |

- Uniform demand (loc 0.0): communication helps vs L0 but LRU is far ahead.  
- **Closest to LFU:** loc **0.6 L1** (3383 vs 3634, **−251**). Still not a win.  
- L3 leads RL at loc 0.4 and 0.8.

### 4.5 Scalability (shifting, loc 0.3, return / node, best)

| N | LRU | LFU | L0 | L1 | L2 | L3 | best RL vs LFU |
|---|-----|-----|----|----|----|----|----------------|
| 5 | 290 | 300 | 214 | 253 | 219 | **259** | −41 |
| 10 | 372 | 356 | 254 | **274** | 230 | 269 | −82 |
| 25 | 485 | 440 | 335 | 314 | 226 | **339** | −100 |
| 50 | 488 | 435 | 347 | 370 | 295 | **386** | −49 |

L3 is strongest RL at N=25/50; still below heuristics per node.

### 4.6 IDQN vs shared-policy

- `gradient_steps=1` IDQN is broken (under-update); L0 ~1564, L1 ~809.  
- With `idqn_gradient_steps=10`, IDQN-L0 **best 3027** beats Shared-L0 2536 but **loses to LRU by ~690**. Higher forwarding (31.7% vs 20.1%) is compensation for weaker local caching, not a coordination win.  
- IDQN-L1 does not beat Shared-L1.  
- IDQN peaks then drops: L0 3027→2453 (−19%), L1 2633→1942 (−26%). Not a deployable default.

### 4.7 SHAP (eviction Q, loc 0.3)

| Feature group | L0 share | L1 share |
|---------------|----------|----------|
| Local cache bits | **60.4%** | 25.6% |
| Local request freqs | 37.3% | 22.8% |
| Neighbor cache bits | — | **51.5%** |

L1 actually uses neighbor cache bits for evictions. That supports the communication story independently of beating LRU.

### 4.8 Training instability (report best vs final)

Canonical loc 0.3, 400k budget:

| Policy | Best | Final | Drop |
|--------|------|-------|------|
| Shared-L0 | 2536 | 2360 | −7% |
| Shared-L1 | 2744 | 1201 | **−56%** |
| Shared-L2 | 2300 | 795 | **−65%** |
| Shared-L3 | 2690 | 1952 | −27% |

L1/L2 **do not converge to a stable greedy policy**. Tables must say “best checkpoint.”

---

## 5. Pipeline lessons (not results, but they shaped the numbers)

1. Warm-start L1/L2 from L0 + best-model selection **blocked** learning neighbor features; train from scratch.  
2. Shared-policy `gradient_steps` must be `num_nodes`.  
3. Uniform Zipf across nodes makes communication useless; `locality_factor` is required.  
4. Overlap as a per-step tax swamped task reward; one-shot cache-action penalty only (unused in these tables).  
5. Observe only same-cluster peers — aligned with forwarding.  
6. Heuristics belong in every multi-node table.

---

## 6. How to cite this snapshot after a refactor

Valid claims from **2K+1, act-then-request, same-cluster obs/forward, blind eval**:

- Neighbor cache binaries (L1) improve **relative** RL return vs L0 under heterogeneous demand.  
- Selective L3 can approach L1 at ~7% always-on comms (canonical).  
- L2 > L1 does not hold at this budget.  
- SHAP: L1 eviction Q is driven by neighbor cache bits.  
- Blind LRU/LFU remain the **absolute** floor; RL did not beat them.

Invalid without a new eval:

- Any number from a different action space or from putting the current request in the observation.  
- “RL solves edge caching” / IDQN as production training.

Reproduce: `experiments/eval_fair_suite.py all --no-idqn --episodes 20 --seeds 42,0,7` on the 2K+1 checkpoints (`results/runs/dqn_multi_level*_scratch_*`).
