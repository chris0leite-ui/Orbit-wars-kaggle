#!/usr/bin/env bash
# Post-data-gen MVP cycle runner: merge -> train -> embed -> bundle -> A/B.
#
# Assumes `data/value_head/pairing_*.npz` chunks exist (produced by
# `scripts/gen_value_training_data.py`). Stops at the A/B step per the
# Rule 1 + user-confirmed scope ("stop and observe").
#
# Env vars consumed:
#   AB_SEEDS  number of seeds for the A/B (default 16, Rule 45 triage)
#   AB_BUDGET BASELINE_WALLCLOCK_MS for A/B games (default 100)

set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR=data/value_head
WEIGHTS=$DATA_DIR/value_head_weights.npz
AB_SEEDS=${AB_SEEDS:-16}
AB_BUDGET=${AB_BUDGET:-100}

echo "=== 1/5 merge chunks ==="
python scripts/gen_value_training_data.py --merge "$DATA_DIR"

echo
echo "=== 2/5 train (local CPU) ==="
python scripts/kaggle_value_head_kernel/train.py

echo
echo "=== 3/5 embed weights into agents/baseline/value_learned.py ==="
python scripts/embed_value_head_weights.py --weights "$WEIGHTS"

echo
echo "=== 4/5 bundle baseline + derive A/B variants ==="
python scripts/bundle_agent.py agents/baseline --force --skip-parity-gate
python scripts/make_ab_bundle_variants.py

echo
echo "=== 5/5 A/B n=$AB_SEEDS (BASELINE_WALLCLOCK_MS=$AB_BUDGET) ==="
BASELINE_WALLCLOCK_MS=$AB_BUDGET python fast.py eval \
    submissions/baseline_learned.py \
    --vs submissions/baseline_favor.py \
    --max-seeds "$AB_SEEDS" \
    --workers 4 2>&1 | tee /tmp/ab_learned_vs_favor.log

echo
echo "=== A/B done. Result above. ==="
