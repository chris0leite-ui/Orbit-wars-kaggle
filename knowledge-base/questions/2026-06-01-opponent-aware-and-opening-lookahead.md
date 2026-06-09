# Open questions — 2026-06-01

- **Opponent-aware expansion shape.** A flat expansion credit is a non-gain
  (40% vs champion, live ~1086). What signal should gate aggression? Candidate:
  lean aggressive when the opponent's planet-growth rate exceeds ours, or when
  free neutrals remain near us early. How to estimate "opponent out-expanding
  us" cheaply at decision time?
- **Dynamic opening lookahead (PI direction).** Does a depth-adaptive lookahead
  (deeper early, gated by entity count / wallclock headroom) improve the opening
  measurably — evaluated vs AGGRESSIVE opponents, not the champion mirror? Build
  on or replace `lib/joint_solver/opening_planner.py` (`opening_plan`)?
- **Right eval instrument.** We keep A/B-ing vs the champion (a hoarder) and
  getting flat ~40% / INCONCLUSIVE. Which opponent set actually differentiates
  expansion/opening quality? The live istinetz/xdddd-type field is the truth;
  what local proxy correlates with it?
