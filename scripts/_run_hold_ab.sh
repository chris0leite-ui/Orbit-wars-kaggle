#!/usr/bin/env bash
# Lever 1 isolation A/B: focal (sync bundle, with hold code) vs the champion
# (no sync code → immune to the sync/hold env switches). Run TWICE on the same
# seeds, differing only in BASELINE_JOINT_SYNC_HOLD, so the win-rate delta
# isolates the size-to-hold contribution. Triage n=16 (seeds 8) per arm.
set -u
cd /home/user/Orbit-wars-kaggle

# Champion production config + sync ON (shared by both arms).
export BASELINE_VALUE_HEAD=hybrid BASELINE_CHOOSER=trajectory BASELINE_JOINT=1 \
  BASELINE_JOINT_AGGR=1 BASELINE_JOINT_TOP_K=5 BASELINE_JOINT_MAX_PAIRS=60 \
  BASELINE_REINFORCE_EMIT=1 BASELINE_REINFORCE_ANTICIPATE=1 \
  BASELINE_NEUTRAL_BONUS=2.0 BASELINE_NEUTRAL_EARLY_EXTRA=1.5 BASELINE_NEUTRAL_EARLY_HORIZON=50 \
  BASELINE_ORBITAL_SAFETY=1 BASELINE_PV_ETA=1 BASELINE_LAUNCH_RULES=1 BASELINE_CAPTURE_HORIZON_K=10 \
  BASELINE_JOINT_SYNC=1 BASELINE_JOINT_SYNC_SRC_K=3

FOCAL=submissions/baseline_joint_sync_hold_focal.py
OPP=submissions/baseline_launch_rules_universal.py
LOG=/tmp/hold_ab.log
SEEDS=${1:-8}
: > "$LOG"

echo "########## ARM A: sync-only (HOLD OFF) vs champion ##########" >> "$LOG"
unset BASELINE_JOINT_SYNC_HOLD
python scripts/clean_ab.py "$FOCAL" "$OPP" --seeds "$SEEDS" --workers 4 \
  2>/dev/null | grep -avE "INFO:|litellm|botocore|open_spiel" >> "$LOG"
echo "" >> "$LOG"

echo "########## ARM B: sync+hold (HOLD ON) vs champion ##########" >> "$LOG"
export BASELINE_JOINT_SYNC_HOLD=1
python scripts/clean_ab.py "$FOCAL" "$OPP" --seeds "$SEEDS" --workers 4 \
  2>/dev/null | grep -avE "INFO:|litellm|botocore|open_spiel" >> "$LOG"
echo "" >> "$LOG"
echo "########## HOLD A/B DONE ##########" >> "$LOG"
