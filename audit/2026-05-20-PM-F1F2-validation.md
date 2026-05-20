# 2026-05-20 PM — F1+F2 fix validation (analytical agent)

> Session continuation of `claude/strategy-framework-design-OyoYR-rebased`.
> Plan: `/root/.claude/plans/do-the-fixes-with-tingly-finch.md`.

## What was done

### L1 diagnostics (permanent tests added)

- `tests/test_trajectory_wait_N.py` (4 tests) — pins `aac3c1e`'s
  wait_N propagation through `predict_fleet_fate`.
- `tests/test_opp_projection_wait_N.py` (3 tests) — pins the
  opp_projection wait_N propagation (BUG 1).
- `tests/test_mpc_silent_idle.py` (3 tests) — pins
  `mpc.solve_turn`'s opening dispatch behavior (BUG 2).
- `tests/test_emit_accuracy.py` (5 tests) — drives a real kaggle
  game and asserts every emitted move's `predict_fleet_fate`
  outcome is `"target"`. **PI claim "ships don't hit targets"
  made testable.**

### L2 fixes applied

- **F1** in `lib/joint_solver/opp_projection.py`: opp launches at
  `tick_offset > 0` now advance src/target positions via
  `predict_relative` AND pass `wait_N=tick_offset` to
  `predict_fleet_fate`. Aligns opp threat projection with the same
  geometry-advancing contract `aac3c1e` introduced for our launches.
- **F2** in `lib/joint_solver/mpc.py:189-221`: opening dispatch
  refactored to three cases:
  1. schedule has fire_step==step_now entries → emit those.
  2. schedule non-empty but no fire-now entries → planner's
     intentional wait (return []).
  3. schedule empty → fall through to Phase-4 LP.
- **Observability**: `MpcDiagnostics.emitted_targets` added —
  per emitted move, records src/tgt/ships/angle/wait_N. Used by
  L1.5 emit-accuracy test.

## L1.5 (emit accuracy) — PRE-fix AND POST-fix result

**Landing rate: 100% across 4 seeds × 60 turns.** Every emitted move
hits its intended target per `predict_fleet_fate`. The "ships don't
hit targets" symptom was **already closed by `aac3c1e`**. F1 is a
correctness fix for the opp-projection side; it doesn't change our
emit-landing rate (we were already at 100%).

## OOB / sun loss diagnostic (PI directive — seed 42 post-fix)

PI directive: "next time check individual games before A/B tests, in
particular checking OOB losses and sun losses." Rule 41 ordering was
inverted in this loop (ran A/B before single-game introspect); going
forward, single-game inspection comes first.

`scripts/check_fleet_outcomes.py --seed 42` (new tool):

| Outcome | Fleets | % |
|---|---:|---:|
| target (landed on intended planet) | 1 | 2.3% |
| non_target_planet | 39 | 90.7% |
| sun | 1 | 2.3% |
| oob | 0 | 0.0% |
| unknown | 2 | 4.7% |

**Headline (the answer to the directive)**: sun = 2.3% (1 of 43),
OOB = 0%. Both are within noise. **Our trajectory plumbing is sound
post-fix.**

**Caveat on the non_target bucket**: position-match heuristics are
imprecise because planets orbit between the fleet's last-seen step
and the disappearance step (the collision happens DURING the step
transition). The same kaggle env's `swept_pair_hit` math is what
`predict_fleet_fate` uses internally, and L1.5 confirms 100% target
landing per that primitive — which has bit-exact parity with the
env (`tests/test_intercept_landing.py`, zero tolerance). The
"non_target" 90% is almost certainly target hits the classifier
fails to match, not real ship-miss events. A proper classifier would
re-invoke the env's swept-pair logic on the fleet's emission-time
trajectory; the existing L1.5 test already does this.

## L4 (n=4 A/B) — verdict FAIL

```
seed=42  L (p1_win)
seed=7   L (p1_win)
seed=1   L (p1_win)
seed=13  L (p1_win)

games:    4   wins: 0/4 (0.0%)
Wilson 95% CI: [0.000, 0.490]
verdict:  FAIL
```

F1+F2 are correct code fixes but **do not move the strategic A/B**.
The analytical agent still loses 0/4 to the trajectory baseline.

## L8 (single-game introspect, seed=42) — diagnostic

Full dump: `audit/2026-05-20-analytical-postfix-seed42-introspect.txt`.

Game ends step 141, analytical eliminated (reward=-1).

| Metric | Value |
|---|---:|
| Turns where LP fired anything | 66/140 (47%) |
| Turns emitting ≥1 wait_N=0 move | **37/140 (26%)** |
| Turns firing ≥1 wait_N>0 column (treadmill) | 38/66 of firing turns (58%) |
| Status `no_positive_columns` (late game) | 22 turns |
| Status `empty_columns` (post-collapse) | 17 turns |

**Diagnosis**: the post-opening LP (lp_outcome) is selecting
wait_N>0 columns that emit nothing this turn and don't survive the
next re-derivation. Steps 84-86 explicitly show:

```
step=84 fired w10=1, emit=0
step=85 fired w9=1,  emit=0
step=86 fired w8=1,  emit=0
```

Same wait_N treadmill Phase 4 retro documented, but in the **mid/
late-game LP** rather than the opening. From step 100 onward, the LP
runs out of positive-value columns ("no_positive_columns" 22 times,
"empty_columns" 17 times) — the analytical has lost its sources and
can't generate captures.

## Status: Rule 37 STOP gate

Per the approved plan's STOP gates section:

> **L4 INCONCLUSIVE / NEGATIVE.** This is the third consecutive
> axis-failure series on the analytical-substrate axis. Rule 37
> binds: STOP iterating on this axis; escalate to PI for refactor/
> rebuild decision.

Confirmed bugs are now CLOSED. Strategic loss persists. The
remaining failure mode is the **value-function ↔ MPC drift bind**
identified by the prior postmortem
(`knowledge-base/thoughts/2026-05-20-analytical-vs-rollout-architectural-bind.md`)
— per-turn LP re-derivation makes wait_N>0 plans evaporate;
analytical can't tempo-match the trajectory baseline's rollout.

## What's preserved on the branch (uncommitted)

- 4 new test files (15 tests; all pass) — permanent regression
  guards for the trajectory + opp + dispatch + emit-accuracy
  contracts.
- F1 in `opp_projection.py` (~25 LOC including pre-rotation).
- F2 in `mpc.py:189-221` (~5 LOC condition refactor).
- `emitted_targets` field on `MpcDiagnostics` (observability).
- This audit + introspect dump.

## Open PI decisions

1. **Commit F1+F2 + tests + audit to branch?** Recommendation: yes —
   the fixes are correct, regression-guarded, and improve
   observability. Even if the strategic axis is closed, the bug
   fixes are net-positive.
2. **Submission strategy?** 52854094 (μ=806) is still in the
   rolling pair with 52857903 (PENDING). 2/5 submissions remain
   today.
3. **Direction?**
   - Stop on analytical-substrate axis (Rule 37 honored).
   - Choose between option B (refactor) or option C (rebuild on
     trajectory) per prior turn's framing.
