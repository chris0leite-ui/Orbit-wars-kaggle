#!/usr/bin/env bash
# Resume the secure-variants panel: finish geo_drift's two missing
# opponents, then run geo_all full panel. Detach-friendly: writes to
# audit/2026-05-14-secure-variants/panel.log, appends, terminates with
# an ALL PANELS DONE marker for easy completion detection.
set -u
cd /home/user/Orbit-wars-kaggle
LOG=audit/2026-05-14-secure-variants/panel.log

echo "=== RESUME $(date -u) ===" >> "$LOG"

echo "=== START geo_drift_resume $(date -u) ===" >> "$LOG"
python fast.py eval agents/geo_drift \
  --vs-panel "v4_planner,v3.5.1" \
  --max-seeds 32 --gate 0.50 --workers 4 >> "$LOG" 2>&1
echo "=== END geo_drift_resume $(date -u) ===" >> "$LOG"

echo "=== START geo_all $(date -u) ===" >> "$LOG"
python fast.py eval agents/geo_all \
  --vs-panel default \
  --max-seeds 32 --gate 0.50 --workers 4 >> "$LOG" 2>&1
echo "=== END geo_all $(date -u) ===" >> "$LOG"

echo "ALL PANELS DONE $(date -u)" >> "$LOG"
