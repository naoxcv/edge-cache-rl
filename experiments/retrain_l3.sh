#!/usr/bin/env bash
# Retrain all eviction-only L3 runs after Q-margin threshold recalibration (0.01 → 0.1).
#   caffeinate -dims bash experiments/retrain_l3.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${ROOT}/venv/bin/python"
LOGDIR="${ROOT}/results/data/fair_eval_evict"
mkdir -p "${LOGDIR}"
LOG="${LOGDIR}/l3_retrain.log"
THRESH=0.1

{
  echo "============================================================"
  echo "L3 retrain threshold=${THRESH} start $(date)"
  echo "============================================================"
} | tee -a "${LOG}"

train_l3() {
  local timesteps="$1"
  local run_name="$2"
  shift 2
  echo "=== TRAIN ${run_name} steps=${timesteps} $(date) ===" | tee -a "${LOG}"
  "${PY}" experiments/train_multi.py train \
    --timesteps "${timesteps}" \
    --comm-level 3 \
    --comm-threshold "${THRESH}" \
    --run-name "${run_name}" \
    "$@" \
    2>&1 | tee -a "${LOG}"
}

# Canonical + traffic/locality (10 nodes, 400k)
train_l3 400000 dqn_evict_level3_scratch_loc0.3 \
  --traffic shifting --locality-factor 0.3 --nodes 10 --clusters 3

for loc in 0.0 0.2 0.4 0.6 0.8; do
  train_l3 400000 "dqn_evict_level3_scratch_n10_c3_shifting_loc${loc}" \
    --traffic shifting --locality-factor "${loc}" --nodes 10 --clusters 3
done

train_l3 400000 dqn_evict_level3_scratch_n10_c3_stationary_loc0.3 \
  --traffic stationary --locality-factor 0.3 --nodes 10 --clusters 3
train_l3 400000 dqn_evict_level3_scratch_n10_c3_bursty_loc0.3 \
  --traffic bursty --locality-factor 0.3 --nodes 10 --clusters 3

# Scale
train_l3 300000 dqn_evict_level3_scratch_n5_c2_shifting_loc0.3 \
  --traffic shifting --locality-factor 0.3 --nodes 5 --clusters 2
train_l3 300000 dqn_evict_level3_scratch_n25_c5_shifting_loc0.3 \
  --traffic shifting --locality-factor 0.3 --nodes 25 --clusters 5
train_l3 250000 dqn_evict_level3_scratch_n50_c10_shifting_loc0.3 \
  --traffic shifting --locality-factor 0.3 --nodes 50 --clusters 10

echo "=== FAIR EVAL exp1 $(date) ===" | tee -a "${LOG}"
"${PY}" experiments/eval_fair_suite.py exp1 \
  --no-idqn --episodes 20 --seeds 42,0,7 \
  2>&1 | tee -a "${LOG}"

echo "=== WEEK7 eval (no retrain L0–L2) $(date) ===" | tee -a "${LOG}"
"${PY}" experiments/week7_sweep.py all --no-train \
  --episodes 20 --seeds 42,0,7 \
  2>&1 | tee -a "${LOG}"

echo "=== L3 retrain done $(date) ===" | tee -a "${LOG}"
