#!/usr/bin/env bash
# Train one Week-7 wave: four comm levels in parallel for a fixed setting.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NODES="${1:?nodes}"
CLUSTERS="${2:?clusters}"
TRAFFIC="${3:?traffic}"
LOC="${4:?locality}"
STEPS="${5:?timesteps}"
EVAL_FREQ="${6:-25000}"

run_name() {
  local level="$1"
  if [[ "$NODES" == "10" && "$CLUSTERS" == "3" && "$TRAFFIC" == "shifting" && "$LOC" == "0.3" ]]; then
    echo "dqn_multi_level${level}_scratch_loc0.3"
  else
    echo "dqn_multi_level${level}_scratch_n${NODES}_c${CLUSTERS}_${TRAFFIC}_loc${LOC}"
  fi
}

pids=()
for LEVEL in 0 1 2 3; do
  NAME="$(run_name "$LEVEL")"
  if [[ -f "results/runs/${NAME}/best_model.zip" ]]; then
    echo "SKIP $NAME"
    continue
  fi
  echo "TRAIN $NAME"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ./venv/bin/python experiments/train_multi.py train \
    --nodes "$NODES" --clusters "$CLUSTERS" --traffic "$TRAFFIC" \
    --locality-factor "$LOC" --comm-level "$LEVEL" --overlap-penalty 0 \
    --comm-penalty 0 --timesteps "$STEPS" --eval-freq "$EVAL_FREQ" \
    --run-name "$NAME" &
  pids+=($!)
done

if [[ ${#pids[@]} -eq 0 ]]; then
  echo "Nothing to train for n=$NODES c=$CLUSTERS $TRAFFIC loc=$LOC"
  exit 0
fi

ec=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    ec=1
  fi
done
exit "$ec"
