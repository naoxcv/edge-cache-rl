# Baseline Parity + IDQN Status (Aug 13, 2026)

## Decision-order fix (done)

Heuristics now default to **blind act-then-request**, matching `MultiAgentCachingEnv.step` / DQN:

1. Act from current obs + **previous** request (delayed reactive caching)
2. `env.step(actions)` applies caches, then scores the **current** request

Implemented as `blind_multi_baseline_step` in `agents/multi_agent.py`.  
`compare_comm_levels.py` / `compare_multi.py` use it by default; `--oracle-baselines` keeps the old reactive path for ablation.

## Important clarification

The old reactive path was **not** “see request → cache → score.” It was already
**score current, then cache for next time**. Blind delayed-LRU is therefore
**isomorphic** to oracle LRU at scoring time: inserting request \(t-1\) at the
start of step \(t\) yields the same cache as inserting it at the end of step
\(t-1\). Empirically LRU returns are **identical** (3717.3) oracle vs blind.

LFU can differ slightly (eviction uses frequencies at decision time); blind LFU
is 3559.3 vs oracle 3561.3.

So the heuristic gap was **never** primarily an unfair “chess with vision”
advantage on the current reward. Reactive LRU/LFU are simply stronger policies
than the trained agents on this multi-node task. DQN can in principle get a hit
on a first occurrence via proactive caching; heuristics never can under either
loop order.

## Fair comparison (blind heuristics, Week-7 protocol)

20 ep × seeds {42,0,7}, 10 nodes, locality 0.3, shifting, same-cluster forwarding.

| Policy | Return | Hit | Fwd | Cloud | vs blind LFU |
|--------|--------|-----|-----|-------|--------------|
| **LRU (blind)** | **3717 ± 92** | 48.2% | 27.2% | **24.6%** | +158 |
| **LFU (blind)** | **3559 ± 106** | **55.1%** | 16.9% | 28.0% | — |
| IDQN-L0 best | 3027 ± 362 | 41.3% | 31.7% | 26.9% | **−532** |
| Shared-L1 best | 2744 ± 90 | 44.7% | 25.4% | 29.9% | **−815** |
| Shared-L3 best | 2690 ± 113 | 48.7% | 19.6% | 31.6% | −869 |
| IDQN-L1 best | 2633 ± 113 | 33.2% | 40.0% | 26.8% | −926 |
| Shared-L0 best | 2536 ± 399 | 47.6% | 20.1% | 32.3% | −1023 |

Relative RL story unchanged: L1 > L0 (+208), L3 ≈ L1 at ~7% comms. Absolute
story unchanged: heuristics still win by 500–1000 return.

## Still required before publishing

1. Put blind LRU/LFU in every multi-node results table (Week 7 + IDQN).
2. Report best vs final for unstable runs (Shared-L1 / IDQN).
3. Frame communication findings as relative RL comparisons, not “RL solves caching.”

## Code / artifacts

- `agents/multi_agent.py`: `blind_multi_baseline_step`
- `tests/test_blind_baselines.py`
- `results/data/idqn/fair_blind_baselines.txt`
- `results/data/idqn/fair_blind_idqn.txt`
