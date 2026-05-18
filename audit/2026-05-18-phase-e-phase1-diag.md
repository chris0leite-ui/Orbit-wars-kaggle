# Phase E Phase 1 single-state diagnostic — same-source pseudo-joint bug

**Date**: 2026-05-18
**Branch**: `claude/ml-competition-strategy-PFhzM`
**Trigger**: Phase 1 A/B at n=8 regressed bundle vs v7_0 from 13/16 (Wlo 0.57)
to 9/16 (Wlo 0.33) and bundle vs baseline from 2/16 to 0/16. PI ratified
single-state diagnostic before tuning.

## TL;DR

`BundleEvaluator._detect_joint_captures` was firing on SAME-SOURCE
pseudo-joints. Two launches from `src=0` at identical angles count as a
"joint capture" against the same hit-planet, earning the `joint_bonus`
even though splitting one launch into two smaller fleets is strictly
worse on the fleet_speed curve. The companion
`BundleSearch._enumerate_joint_seeds` was already correctly requiring
distinct sources (`srcs_by_dist[i]` vs `[j]` with `j > i`); only the
scorer-side detector was permissive.

Fix shipped in `9cbcc8f`: detector now requires `len(distinct_sources)
>= 2` in the same-target launch group before tagging a joint.

## Diagnostic procedure

`scripts/diag_single_turn.py` with `--reuse-obs` re-scores the same
pinned obs under different env-var configurations. Apples-to-apples
comparison; the only variable is the joint coefficient.

Initial probe at turn 20 (the inflection point used in cands=5
diagnostic) showed **no behavioral change** between joint-OFF and
joint-ON — at that mid-game state with balanced ship counts, no source
pair was eligible for a joint seed (every source has avail > defender
for nearby targets). The regression must originate at OTHER turns.

Probe at turn 5 (early-game, max launch opportunity per source) revealed
the bug.

## The smoking gun (seed=42, turn=5, bundle vs baseline)

State at turn 5:
- bundle (P0): 4 planets, ~52 ships total, ~13 ships/planet
- baseline (P1): 4 planets
- 12 neutrals

### Joint OFF (`BUNDLE_JOINT_BONUS=0` `BUNDLE_JOINT_SEEDS=0`)

```
Bundle's top-10 candidates (by score):
   1. score= 229.00  EMPTY
   ...
   9. score= 229.00  src0->turn0(3sh), src0->turn0(3sh)    ← pseudo-joint
  10. score= 229.00  src0->turn0(3sh), src0->turn0(2sh)
Bundle CHOSE this turn (0 actions)
```

Many candidates tie at 229. The 2-launch same-source candidate at rank 9
scores **vanilla 229** — no bonus, no preference.

### Joint ON (`BUNDLE_JOINT_BONUS=0.5` `BUNDLE_JOINT_SEEDS=10`)
**Before fix:**

```
Bundle CHOSE this turn (2 actions):
  src=0 angle=-0.300 ships=3
  src=0 angle=-0.300 ships=3
Bundle's chosen score: 236.50  (empty was 229)
```

Score jumped 229 → 236.50 = vanilla 229 + **7.5 joint bonus**. Bundle
emitted **two 3-ship launches from the same source**. Strategically
worse than a single 6-ship launch (fleet_speed scales with ship count),
but the bonus pushed the chooser to prefer it.

### Joint ON, **after fix** (distinct-sources required)

```
Bundle CHOSE this turn (0 actions)
```

The same 2-launch same-source candidate still appears at rank 9, but
now with vanilla 229 score — no bonus, no preference over empty. Bundle
correctly does nothing at turn 5.

## Why the search emitted same-source candidates

`BundleSearch._enumerate_candidates` emits per-(src, target, launch_turn,
ship_ratio) tuples. The `ship_ratios=(0.5, 1.0)` grid plus the
`capture_min = tgt.ships + 1` variant produces 2-3 distinct ship counts
per (src, tgt). During the depth-2 search's ADD iterations, the beam
can extend `{(src0, 3sh)}` with another `(src0, 2sh)` or `(src0, 3sh)`
(same source, different ship counts after the small-usable-budget
collapses both ratios to identical integers near the boundary). This
isn't a bug in enumeration — it's a legitimate exploration of the
action space. But the scorer's joint detector misinterpreted it.

## Why turn 20 looked clean

At turn 20 (n=4 own planets, ~13 ships/planet, balanced state), the
search's frontier converges on EMPTY because every individual candidate
scores at or near empty's 153. The 2-launch same-source candidates either
don't get into the top-10 OR they tie at 153 (no incremental gain from
the pseudo-joint bonus at this state — production×remaining drops to
near-zero at long arrival turns under horizon=15).

So the regression accumulated in EARLY-game turns (5-15) where ship
budgets are small, neutrals are close, and same-source pseudo-joints
emerged from the ADD iterations on tight budgets.

## Validation properties

After the fix:
- J1+J2 oracles pass (both use distinct sources in their fixtures).
- 8 bundle oracles + all 31 trajectory_layer event_driven_opp tests
  unaffected.
- Turn-5 diag now produces IDENTICAL behavior joint-OFF vs joint-ON
  (because no real joint opportunity exists at this state).

The fix is necessary AND likely sufficient for the same-source class of
false positives. Whether REAL joints (distinct sources, sum > def,
each indiv < def) now move the A/B needle is the next measurement.

## Artifacts

- `audit/diag_obs_baseline_seed42_t5.pkl` — pinned obs (reusable)
- Diag log (this file) — reproduction commands + before/after
- `9cbcc8f` — fix commit
