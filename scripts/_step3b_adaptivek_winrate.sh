#!/usr/bin/env bash
# Step 3b (CORRECTED): teamwork (refine) vs the LIVE adaptive-K champion.
# The original Step 3 ran with adaptive-K OFF (fixed horizon 10) on both sides —
# a non-champion config. This re-bases on the real champion: BASELINE_ADAPTIVE_K=1
# (horizon 20 early -> floor 10 by step 30) for BOTH the focal agent and the
# frozen bundle opponent (the bundle reads BASELINE_ADAPTIVE_K from env, default
# off, so =1 turns it into champ_adaptiveK_on, mu~1170).
#
# Both choosers run on the SAME 16 seeds so refine and trajectory can be paired
# seed-by-seed (does refine win seeds trajectory loses? does it BREAK seeds
# trajectory wins?). Opponent is the immune bundle, so BASELINE_CHOOSER only
# affects the focal agent (no env collision).
set -u
cd "$(dirname "$0")/.."

export BASELINE_JOINT_AGGR=1 BASELINE_JOINT_TOP_K=5 BASELINE_JOINT_MAX_PAIRS=60 \
       BASELINE_REINFORCE_EMIT=1 BASELINE_REINFORCE_ANTICIPATE=1 \
       BASELINE_NEUTRAL_BONUS=2.0 BASELINE_NEUTRAL_EARLY_EXTRA=1.5 \
       BASELINE_NEUTRAL_EARLY_HORIZON=50 BASELINE_ORBITAL_SAFETY=1 \
       BASELINE_PV_ETA=1 BASELINE_LAUNCH_RULES=1 BASELINE_CAPTURE_HORIZON_K=10 \
       BASELINE_VALUE_HEAD=hybrid BASELINE_JOINT=1 \
       BASELINE_ADAPTIVE_K=1   # <-- the live champion's decaying horizon, ON

FOCAL=agents/baseline/main.py
OPP=submissions/baseline.py

echo "########## PARITY (adaptiveK): trajectory vs adaptiveK champion (expect ~50%) ##########"
BASELINE_CHOOSER=trajectory python scripts/clean_ab.py "$FOCAL" "$OPP" \
    --seeds 16 --workers 4

echo ""
echo "########## STEP 3b (adaptiveK): refine vs adaptiveK champion (n=16 seeds = 32 games) ##########"
BASELINE_CHOOSER=refine python scripts/clean_ab.py "$FOCAL" "$OPP" \
    --seeds 16 --workers 4 --save-replays /tmp/refine_adaptivek_replays
echo "########## DONE-3B ##########"
