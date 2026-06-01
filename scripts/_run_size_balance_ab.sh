#!/usr/bin/env bash
# SIZE_BALANCE (A+D) triage A/B (Rule 45 triage; n=16 default).
# Same focal (live branch baseline, which carries the default-OFF
# SIZE_BALANCE code) run TWICE vs the frozen champion bundle
# (baseline_launch_rules_universal — immune, its inlined proposer predates
# the flag). The two arms differ ONLY in BASELINE_SIZE_BALANCE, so the
# focal win-rate delta isolates the fix and cancels any focal-vs-champion
# branch drift. Mirrors scripts/_run_hold_ab.sh.
set -u
cd /home/user/Orbit-wars-kaggle

# Champion (baseline_launch_rules_universal) production config — shared by
# both arms, copied verbatim from the bundle's setdefault header.
export BASELINE_JOINT_AGGR=1 BASELINE_JOINT_TOP_K=5 BASELINE_JOINT_MAX_PAIRS=60 \
  BASELINE_REINFORCE_EMIT=1 BASELINE_REINFORCE_ANTICIPATE=1 \
  BASELINE_NEUTRAL_BONUS=2.0 BASELINE_NEUTRAL_EARLY_EXTRA=1.5 BASELINE_NEUTRAL_EARLY_HORIZON=50 \
  BASELINE_ORBITAL_SAFETY=1 BASELINE_PV_ETA=1 BASELINE_LAUNCH_RULES=1 BASELINE_CAPTURE_HORIZON_K=10 \
  BASELINE_VALUE_HEAD=hybrid BASELINE_CHOOSER=trajectory BASELINE_JOINT=1

FOCAL=agents/baseline/main.py
OPP=submissions/baseline_launch_rules_universal.py
LOG=/tmp/size_balance_ab.log
SEEDS=${1:-8}
: > "$LOG"

echo "########## ARM A: SIZE_BALANCE OFF (control, expect ~50%) vs champion ##########" >> "$LOG"
unset BASELINE_SIZE_BALANCE
python scripts/clean_ab.py "$FOCAL" "$OPP" --seeds "$SEEDS" --workers 4 \
  2>/dev/null | grep -avE "INFO:|litellm|botocore|open_spiel" >> "$LOG"
echo "" >> "$LOG"

echo "########## ARM B: SIZE_BALANCE ON (A+D fix) vs champion ##########" >> "$LOG"
export BASELINE_SIZE_BALANCE=1
python scripts/clean_ab.py "$FOCAL" "$OPP" --seeds "$SEEDS" --workers 4 \
  2>/dev/null | grep -avE "INFO:|litellm|botocore|open_spiel" >> "$LOG"
echo "" >> "$LOG"
echo "########## SIZE_BALANCE A/B DONE ##########" >> "$LOG"
