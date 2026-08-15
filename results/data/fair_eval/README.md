# Fair Evaluation Results (post-retrain, blind LRU/LFU)

**Protocol:** blind act-then-request baselines (same step order as DQN).  
**Eval:** 20 episodes × seeds {42, 0, 7}.  
**Tool:** `experiments/eval_fair_suite.py all` after L1–L3 retrain under same-cluster obs.  
**Log:** `week7_retrained_eval.log` (finished).

All L1–L3 Week-7 cells now have matching obs widths (no skips).

---

## Exp1 — 10 nodes / shifting / locality 0.3

| Policy (best) | Return | Hit | Fwd | Cloud | vs LFU |
|---------------|--------|-----|-----|-------|--------|
| **LRU** | **3717 ± 92** | 48.2% | 27.2% | 24.6% | +158 |
| **LFU** | **3559 ± 106** | 55.1% | 16.9% | 28.0% | — |
| Shared-L1 | 2744 ± 90 | 44.7% | 25.4% | 29.9% | −816 |
| Shared-L3 | 2690 ± 113 | 48.7% | 19.6% | 31.6% | −869 |
| Shared-L0 | 2536 ± 399 | 47.6% | 20.1% | 32.3% | −1024 |
| Shared-L2 | 2300 ± 232 | 44.6% | 22.5% | 32.9% | −1260 |

**Relative RL:** L1 − L0 = **+208**; L3 − L1 = **−53**.

**Best → final drop @ loc 0.3:** L1 −56%, L2 −65%, L3 −27%, L0 −7%.

---

## Traffic (10 nodes, loc 0.3)

| Traffic | LRU | LFU | L0 | L1 | L2 | L3 | best RL vs LFU |
|---------|-----|-----|----|----|----|----|----------------|
| stationary | 3728 | 3612 | 2726 | **3038** | 2285 | 2951 | −574 |
| shifting | 3717 | 3559 | 2536 | **2744** | 2300 | 2690 | −816 |
| bursty | 3135 | 3180 | 2587 | **2691** | 2280 | 2624 | −489 |

L1 leads RL on every traffic pattern. Gap to LFU is smallest under bursty (−489).

---

## Locality sweep (10 nodes, shifting)

| loc | LRU | LFU | L0 | L1 | L2 | L3 | best RL vs LFU |
|-----|-----|-----|----|----|----|----|----------------|
| 0.0 | 3846 | 3347 | 1692 | **2780** | 2096 | 2472 | −567 |
| 0.2 | 3783 | 3373 | 2456 | **2902** | 2118 | 2735 | −471 |
| 0.3 | 3717 | 3559 | 2536 | **2744** | 2300 | 2690 | −816 |
| 0.4 | 3663 | 3604 | 2895 | 2869 | 2320 | **3262** | −342 |
| 0.6 | 3578 | 3634 | 3060 | **3383** | 2596 | 3076 | **−251** |
| 0.8 | 3591 | 3711 | 3031 | 2972 | 2786 | **3214** | −497 |

Closest RL gets to LFU: **loc 0.6 L1 (3383 vs 3634, −251)**. L3 wins at loc 0.4 and 0.8. Heuristics still win everywhere.

---

## Scalability (shifting, loc 0.3) — return / node

| N | LRU | LFU | L0 | L1 | L2 | L3 | best RL vs LFU |
|---|-----|-----|----|----|----|----|----------------|
| 5 | 290 | 300 | 214 | 253 | 219 | **259** | −41 |
| 10 | 372 | 356 | 254 | **274** | 230 | 269 | −82 |
| 25 | 485 | 440 | 335 | 314 | 226 | **339** | −100 |
| 50 | 488 | 435 | 347 | 370 | 295 | **386** | −49 |

At N=50, L3 is the strongest RL policy but still ~49/node (~2450 total) below LFU.

---

## Takeaways

1. **Heuristics still dominate** absolute return under the fair blind protocol.
2. **Relative RL story holds:** L1 > L0; L3 ≈ L1 (often with less comms at L3); L2 lags.
3. **Closest approach to LFU:** locality 0.6 Shared-L1 (−251). Still not a win.
4. **Report best checkpoints**; L1/L2 finals often collapse.
5. Retrain closed the obs-mismatch gap — full Week-7 tables are now valid.

## Files

| File | Contents |
|------|----------|
| `exp1_summary.csv` | Canonical 10/shifting/0.3 |
| `traffic_summary.csv` | stationary / shifting / bursty |
| `locality_summary.csv` | loc 0.0–0.8 |
| `scalability_summary.csv` | N ∈ {5,10,25,50} |
| `all_summary.csv` | Combined |
| `week7_retrained_eval.log` | Full run log |
