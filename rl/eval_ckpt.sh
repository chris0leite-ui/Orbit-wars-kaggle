#!/bin/bash
# Evaluate an RL checkpoint against the local opponent panel.
# Usage: bash rl/eval_ckpt.sh /path/to/ckpt.pkl [max_seeds]
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="$1"
MAXSEEDS="${2:-32}"
NAME=$(basename "$CKPT" .pkl)
OUT="/tmp/rl_eval_${NAME}.py"

python -m rl.export_agent "$CKPT" "$OUT"

echo "=== smoke: 1 game vs v7_0 (seed 7) ==="
python fast.py play "$OUT" --vs submissions/v7_0_drop_one.py --seed 7 2>&1 | tail -5

for OPP in submissions/v7_0_drop_one.py agents/producer/main.py submissions/ledger_v1_4.py; do
    echo "=== eval vs $OPP (max_seeds=$MAXSEEDS) ==="
    python fast.py eval "$OUT" --vs "$OPP" --max-seeds "$MAXSEEDS" 2>&1 | tail -8
done
