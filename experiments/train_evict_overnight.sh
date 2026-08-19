#!/usr/bin/env bash
# Overnight eviction-only (C+1) training + eval.
# Run under caffeinate so the Mac does not sleep:
#   caffeinate -dims bash experiments/train_evict_overnight.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${ROOT}/venv/bin/python"
LOGDIR="${ROOT}/results/data/fair_eval_evict"
mkdir -p "${LOGDIR}"
LOG="${LOGDIR}/overnight.log"

{
  echo "============================================================"
  echo "eviction-only overnight start $(date)"
  echo "root=${ROOT}"
  echo "============================================================"
} | tee -a "${LOG}"

train_canonical() {
  local level="$1"
  echo "=== TRAIN shared L${level} $(date) ===" | tee -a "${LOG}"
  "${PY}" experiments/train_multi.py train \
    --timesteps 400000 \
    --comm-level "${level}" \
    --traffic shifting \
    --locality-factor 0.3 \
    --run-name "dqn_evict_level${level}_scratch_loc0.3" \
    2>&1 | tee -a "${LOG}"
}

for level in 0 1 2 3; do
  train_canonical "${level}"
done

echo "=== FAIR EVAL exp1 $(date) ===" | tee -a "${LOG}"
"${PY}" experiments/eval_fair_suite.py exp1 \
  --no-idqn --episodes 20 --seeds 42,0,7 \
  2>&1 | tee -a "${LOG}"

echo "=== WEEK7 remaining sweeps (skip existing canonical) $(date) ===" | tee -a "${LOG}"
"${PY}" experiments/week7_sweep.py all \
  --episodes 20 --seeds 42,0,7 \
  2>&1 | tee -a "${LOG}"

echo "=== eviction-only overnight done $(date) ===" | tee -a "${LOG}"
