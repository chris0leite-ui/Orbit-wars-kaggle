# 2026-05-22 — baseline_kt substrate wire-up (Option 1)

**Branch:** `claude/extract-physics-trajectory-Vjaz9`
**Commit:** `923852e` — "feat(baseline): env-gated kinematic-table priming + baseline_kt wrapper"
**Hypothesis (Option 1 of the planning thinking-pass):** Wire the
per-turn kinematic table into the baseline agent. Freed
`predict_relative` cycles flow into the chooser's wallclock-adaptive
candidate loop → modest lift OR parity, never regression.

## What landed

- `agents/baseline/main.py:861-867` — env-gated
  `kinematic_table.begin_turn(world)` after `World.from_obs`.
  Default OFF. `KINEMATIC_TABLE_ENABLED=1` activates.
- `agents/baseline_kt/main.py` — wrapper, same brain as
  `baseline_full` plus `KINEMATIC_TABLE_ENABLED=1`. Local-eval only
  (bundler refuses cross-agent wrappers).
- `scripts/bundle_agent.py:DEFAULT_LIB_ORDER` — inserted
  `"kinematic_table"` after `"orbit"` for future bundled submissions.

## Verification

### Rule 38 — speed reproduction (env-OFF control vs env-ON treatment)

3 games vs `v7_0`, single-threaded:

| Metric        | env-OFF (control) | env-ON (treatment) | Δ                  |
|---------------|------------------:|-------------------:|--------------------|
| p50           |             665 ms |              654 ms | −11 ms             |
| **p95**       |         **859 ms** |          **741 ms** | **−118 ms (−14%)** |
| p99           |             904 ms |              777 ms | −127 ms            |
| max           |             991 ms |              815 ms | −176 ms            |
| over_1000 ms  |                  0 |                   0 | 0                  |
| **Verdict**   |        **WATCH**   |          **PASS**   | gate boundary      |

env-OFF hits WATCH because p95 ≥ 800 ms. env-ON passes the PASS gate
(p95 < 800 ms AND zero turns ≥ 1000 ms). Speed reduction is real.

Behavioural note: seed 0 ran 380 turns env-ON vs 301 turns env-OFF
(both p0_win). Same brain, same wins, more candidates evaluated per
turn → different — but equally-winning — plays emerge. The chooser's
wallclock loop consumes the freed budget, as designed.

### Rule 45 — A/B parity gate (n=32 seeds × 2 swaps = 64 games)

`python fast.py eval agents/baseline_kt --vs agents/baseline_full --max-seeds 32 --full-panel`

- **n=64, wins=31/64, point=48.4%, Wilson [0.366, 0.604]**
- Verdict: **INCONCLUSIVE** (Wilson-lo 0.366 < 0.50, but 0.5 inside CI).
- Wallclock: 3509 s (~58 min total over 8 workers).

## Reading

**Net:** Speed gain real (~14% p95 reduction unlocks budget gate),
but the additional candidates evaluated do not shift the chooser's
top-scoring move. Point estimate is parity. Wilson-lo fails the
strict 0.50 gate; CI spans 50% so we cannot conclude regression.

This matches the existing finding that "the K=10 rollout already
implicitly encodes the signal" — giving the same brain more
candidates inside the same `WALLCLOCK_BUDGET_MS=600` window doesn't
move the leaderboard. The chooser was not actually candidate-starved.

## What's left on the table (next options)

The substrate is wired and parity-clean. To turn this into a real
ladder lift, one of:

1. **Use freed cycles for DEPTH, not breadth.** Bump
   `MIN_HORIZON`/`MAX_HORIZON` in `agents/baseline/proposer.py:29-30`
   (currently 25/40). With the table the per-step cost is lower; we
   can roll out further without hitting the 1s Kaggle gate. This is
   a true substrate-driven mechanism change, not a brain change.

2. **Phase 2 substrate swaps** — wire the table into the smaller
   call sites (`world_model.time_to_enemy_threat`,
   `mechanism.lead_aim_v2`, `aim.aim_orbiting + search_safe_intercept`,
   `proposer.aim_and_eta` wait_N, `joint_solver/opp_projection`).
   Cumulative ~500–800 additional calls/turn saved. May tip the
   per-turn budget headroom enough to unlock another K-bump.

3. **Option 3 from the planning thinking-pass** — value-upstream
   defensive mechanism. Multiply candidate capture value by
   P(fleet survives flight), computed by walking the planned path
   step-by-step through the table. Addresses the H44 finding (65%
   of capture failures are fleet-destroyed-in-flight) via
   modeling-correctness (Rule 40) rather than restriction-tuning.

## Rollback

Work is fully additive and gated. To revert:

1. `git revert 923852e` removes both the `agents/baseline/main.py`
   priming + the wrapper + the bundler change.
2. `KINEMATIC_TABLE_ENABLED=` unset (default) → slow path for every
   agent.
3. `baseline_full` remains the stable rolling-pair floor reference.

## Submission decision

**Not yet.** Parity-with-known-floor is not worth a submission slot
when the rolling pair currently holds μ=1078/1117.9 (3 days into a
31-days-remaining window). Wait for one of the three follow-on
options to produce real lift, then submit.
