#!/usr/bin/env bash
# Step 3: does EXPLOITING the sync-coalitions (refine chooser) beat the champion?
# Parity control first (trajectory base must match the frozen bundle ~50%),
# then the refine winrate test. Opponent = frozen champion bundle (immune to
# BASELINE_CHOOSER), so focal=refine vs opp=champion has no env collision.
set -u
cd "$(dirname "$0")/.."

# Champion production config — set so the FROZEN BUNDLE plays at full champion
# strength (it reads these at runtime; unset would default several OFF).
export BASELINE_JOINT_AGGR=1 BASELINE_JOINT_TOP_K=5 BASELINE_JOINT_MAX_PAIRS=60 \
       BASELINE_REINFORCE_EMIT=1 BASELINE_REINFORCE_ANTICIPATE=1 \
       BASELINE_NEUTRAL_BONUS=2.0 BASELINE_NEUTRAL_EARLY_EXTRA=1.5 \
       BASELINE_NEUTRAL_EARLY_HORIZON=50 BASELINE_ORBITAL_SAFETY=1 \
       BASELINE_PV_ETA=1 BASELINE_LAUNCH_RULES=1 BASELINE_CAPTURE_HORIZON_K=10 \
       BASELINE_VALUE_HEAD=hybrid BASELINE_JOINT=1

FOCAL=agents/baseline/main.py
OPP=submissions/baseline.py

echo "########## PARITY CONTROL: trajectory vs champion bundle (expect ~50%) ##########"
BASELINE_CHOOSER=trajectory python scripts/clean_ab.py "$FOCAL" "$OPP" --seeds 4 --workers 4

echo ""
echo "########## STEP 3: refine vs champion bundle (n=16 seeds = 32 games) ##########"
BASELINE_CHOOSER=refine python scripts/clean_ab.py "$FOCAL" "$OPP" \
    --seeds 16 --workers 4 --save-replays /tmp/refine_step3_replays
echo "########## DONE ##########"
