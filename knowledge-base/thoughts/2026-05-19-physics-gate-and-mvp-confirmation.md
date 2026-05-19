# 2026-05-19 PM2 — physics-validation gate + MVP confirmation

Continuation of the 5/19 PM v4-pivot session. Two structural findings:

## 1. Our experimental line ignored our own physics layer

`lib.trajectory.predict_fleet_fate` exists and is battle-tested by
baseline.py via `lib/mechanism.py:593,686,775` (`sun_avoid`,
`path_clears_other_planets`, `oob_guard`). Our entire experimental
agent line — `trajectory_roi`, `cluster_solver`, all Phase B variants,
`goal_planner` v1 — never imported it. Replay-probe shows ~6.8% of
trajectory_roi emits are physically wasted (sun + OOB).

PI's question was the unlock: "what about our whole physics and
trajectory modeling, do we use it?" The answer was: only baseline.py,
the live submission we were trying to beat from scratch.

The fix is mechanical (`lib/goal_planner/validate.py` + late-with-
fallback in sequencer/defense). The diagnosis is the lesson: every
PRIMITIVE in our line should be ground-truth-verified against the
env's actual behaviour, not just self-consistent on constructed
scenarios. Promoted as Rule 41 + Rule 42 candidates (improvements.md).

## 2. The MVP confirmed the chooser layer is strategically neutral

Built `agents/greedy_expand/` (60 LOC, no predicate, no portfolio, no
sequencer, no defense). Just ROI-sorted greedy expansion + physics
validation. A/B vs `goal_planner` (~500 LOC, full architecture stack):
14/32 (43.8%, Wilson [0.282, 0.607]) — **statistical tie**.

A 60-LOC agent matched a 500-LOC architected planner. The entire
architectural overlay added no measurable value at the primitive
layer we're operating.

Both agents lose 0/32 to Kaggle baseline. The chooser doesn't matter
because the FOUNDATION matters more, and our foundation is missing
what baseline has (`composite_capture_value`, hold-feasibility filter,
defender accumulation, joint candidate evaluation, opp counter
modeling, ~100KB of accumulated mechanism/world_model/mission code).

## What this implies for next session

Three things in tension:

- **Promoted rules 41+42 want us to verify primitives before
  chooser work.** That points to a "verify lib.trajectory primitives
  on N real replays" deliverable before anything new.
- **The only signal across 8+ A/Bs was wrapping baseline.py**
  (`baseline_veto` 12/32 = 37.5%). Wrap-vs-replace asymmetry is
  strong; PI did NOT ratify it as a rule but it remains the only
  path with evidence of value.
- **The 5/19 PM scenario-suite-first plan** is at
  `/root/.claude/plans/no-go-forward-test-fluttering-token.md` —
  built around objective-first scenarios. Likely still the right
  direction but now needs the physics-gate addendum.

The honest read: building from our primitives has been falsified
twice (PM + PM2, 10 total iterations 0-1/32). The cluster-tablebase
work and goal-directed-planner are intellectually clean but live-
ladder-dominated. Next session probably needs to combine: (a)
verify each primitive end-to-end vs env, (b) build the next agent
either by wrapping baseline OR by adopting baseline's primitives
(`composite_capture_value`, hold-feasibility, joint evaluation) into
our line.

## Per-A/B record this session (for calibration ladder)

| focal | opp | wins/n | Wlo | verdict |
|---|---|---|---|---|
| tablebase_veto | trajectory_roi | 15/32 (46.9%) | 0.309 | INCONCLUSIVE |
| tablebase_veto | submissions/baseline | 0/32 | 0.000 | FAIL |
| tablebase_hybrid | submissions/baseline | 0/32 | 0.000 | FAIL (max 1564ms over cap) |
| baseline_veto | submissions/baseline | 12/32 (37.5%) | 0.229 | INCONCLUSIVE |
| goal_planner (no validate) | submissions/baseline | 0/32 | 0.000 | FAIL |
| goal_planner (with validate) | submissions/baseline | 0/32 | 0.000 | FAIL |
| greedy_expand | goal_planner | 14/32 (43.8%) | 0.282 | INCONCLUSIVE |
| greedy_expand | submissions/baseline | 0/32 | 0.000 | FAIL |
| greedy_expand | nearest (smoke ×3) | 3/3 | n/a | PASS |
