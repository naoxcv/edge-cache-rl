#!/usr/bin/env bash
# Orchestrate Week-7 training waves then evaluate.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
WAVE=./experiments/week7_train_wave.sh
chmod +x "$WAVE"

echo "===== LOCALITY SWEEP (10 nodes / shifting) ====="
for LOC in 0.0 0.2 0.4 0.6 0.8; do
  echo ">>> locality=$LOC"
  "$WAVE" 10 3 shifting "$LOC" 400000 25000
done

echo "===== TRAFFIC SWEEP (10 nodes / loc 0.3) ====="
for TRAF in stationary bursty; do
  echo ">>> traffic=$TRAF"
  "$WAVE" 10 3 "$TRAF" 0.3 400000 25000
done

echo "===== SCALABILITY (shifting / loc 0.3) ====="
"$WAVE" 5 2 shifting 0.3 300000 20000
"$WAVE" 25 5 shifting 0.3 300000 20000
"$WAVE" 50 10 shifting 0.3 250000 20000

echo "===== EVALUATE ALL ====="
./venv/bin/python experiments/week7_sweep.py all --episodes 20 --seeds 42,0,7 --no-train
echo "WEEK7_DONE"
