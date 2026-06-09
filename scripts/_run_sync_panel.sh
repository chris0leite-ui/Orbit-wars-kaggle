#!/usr/bin/env bash
# Sync-coalition confirmation panel (Rule 43 breadth pass).
# Runs the FIXED sync focal vs each calibration-panel opponent via clean_ab
# (subprocess-per-game = clean env, no module-load pollution). Triage n=16
# (32 games/opp); extend passers to n=32 next session.
#
# Output: /tmp/sync_panel.log  (grep 'focal_wins' for the per-opponent verdict)
set -u
cd /home/user/Orbit-wars-kaggle

export BASELINE_VALUE_HEAD=hybrid BASELINE_CHOOSER=trajectory BASELINE_JOINT=1 \
  BASELINE_JOINT_AGGR=1 BASELINE_JOINT_TOP_K=5 BASELINE_JOINT_MAX_PAIRS=60 \
  BASELINE_REINFORCE_EMIT=1 BASELINE_REINFORCE_ANTICIPATE=1 \
  BASELINE_NEUTRAL_BONUS=2.0 BASELINE_NEUTRAL_EARLY_EXTRA=1.5 BASELINE_NEUTRAL_EARLY_HORIZON=50 \
  BASELINE_ORBITAL_SAFETY=1 BASELINE_PV_ETA=1 BASELINE_LAUNCH_RULES=1 BASELINE_CAPTURE_HORIZON_K=10 \
  BASELINE_JOINT_SYNC=1 BASELINE_JOINT_SYNC_SRC_K=3

FOCAL=submissions/baseline_joint_sync_focal.py
LOG=/tmp/sync_panel.log
: > "$LOG"

for OPP in submissions/v7_0_drop_one.py submissions/v4_planner.py submissions/v3.5.1.py; do
  echo "########## PANEL OPP=$OPP ##########" >> "$LOG"
  python scripts/clean_ab.py "$FOCAL" "$OPP" --seeds 16 --seed-start 0 --workers 8 \
    2>/dev/null | grep -avE "INFO:|litellm|botocore|open_spiel" >> "$LOG"
  echo "" >> "$LOG"
done
echo "########## PANEL DONE ##########" >> "$LOG"
