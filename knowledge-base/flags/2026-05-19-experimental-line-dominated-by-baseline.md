# Flag — 2026-05-19 PM2 — experimental-line agents categorically dominated by Kaggle baseline

Across 8+ A/Bs this session, **6 of them ended 0/32 vs
`submissions/baseline.py`**. Five distinct architectures (Phase B veto,
hybrid, goal_planner ±validation, greedy_expand MVP). Combined with
the prior session's 5 trajectory_roi iterations (0-1/32) and the
historic v8_scavenge / v12 / v15 line, we now have 10+ data points
showing the same pattern: agents built from `_aim_and_eta` primitives
do not compete with baseline.py.

Only positive signal: `agents/baseline_veto/` wrapping the live
submission (12/32 = 37.5%, Wilson [0.229, 0.547] — INCONCLUSIVE but
notably better than 0/32).

**Why this is flag-worthy** (per standing duty `knowledge-base/flags/2026-05-06.md`):
this is a structural blocker, not a tuning gap. Continuing to iterate
choosers on our primitives is wasting compute. Future-session decisions
about agent design should be reviewed against this evidence — the
default move should NOT be "build new agent from our primitives."
