# 2026-05-24 — Hold-aware-opening falsified; latent LP-pending bug surfaced

## What I bet on

After Phase ζ.v2 LP-side hold-aware shipped at Gate 5 = 4/16 vs
orbitfix (commit `5db11f2`), I extended the mechanism to the
`opening_planner` closed-form value math. Reasoning: orbitfix wins
on opening tempo (144-ship lead by step 60); the LP doesn't run
until step 30; therefore the lift needs to land in the opening too.

Mechanism: replace the binary Stage-1 gate in `_expected_hold_duration`
with a continuous `hold_from_counter`, so bigger `capture_residual`
(= bigger fires) → bigger hold → bigger MILP value → MILP picks
bigger fleets.

## What happened

Gate 5 vs orbitfix:
- Unscoped opening hold-aware: **1/16** (vs 4/16 baseline; −3 wins).
- Scoped opening hold-aware (`_predict_opp_counter` restricted to
  `OPP_RESPONSE_LAG=4` window): **2/16** (still −2 wins).

Both below baseline. Per Rule 37 + the plan's kill criterion, the
opening hold-aware axis is **falsified**. Reverted commit `9a19306`;
closed track added to `state/MULTI_BRANCH.md`.

## The surprise — latent bug exposed

The 1/16 wasn't just "mechanism doesn't help." Per-turn introspect on
seed 0/P1 (a prior win that became a loss) showed an explosive
divergence at step 12: the agent emitted **12 duplicate fires from
src=15, ships=21 each = 252 ships from a planet that had 1 ship**.

Trace:
1. The opening hold-aware change made `_predict_opp_counter`
   over-pessimistic (considered ALL opp planets, including
   too-distant-to-respond ones). `_build_candidates` rejected more
   targets → opening_planner returned empty schedules during the
   opening phase.
2. Empty opening schedule → pipeline fell through to
   `decision_outcome_aware_milp` at step < 30. **The LP runs without
   pending-aware budget by default** (`LP_PENDING_AWARE_BUDGET=0`
   in the bundle).
3. The LP at each of steps 1..11 sees src=15 with full ships and
   commits a wait_N>0 fire targeting fire_step=12. `commit_persistent`
   appends without de-duplication; the pending list grows to 11
   identical `ScheduledFires`.
4. At step 12, all 11 decant plus the LP's own wait_N=0 fire = 12
   emissions. Source drained. Game lost.

The scoped fix (closing the over-rejection) eliminated the
12-duplicate bug. But the mechanism's continuous-hold modeling
**didn't move MILP picks beneficially** — `_predict_opp_counter`
under the `OPP_RESPONSE_LAG` scope often returns the same opp source
the legacy gate already considered, just with closest-by-ETA rather
than max-by-ships. The two predicates produced byte-identical seed-0
behavior through step 14.

## Why the mechanism is structurally too weak

`production × hold_dur × γ^t` isn't expressive enough to
discriminate ship counts strongly. Bigger residual marginally
increases hold_dur (bounded by Stage-2 eta-delta), but the
multiplicative factor between fires of size 3 vs size 11 is
maybe 1.3× — not enough to overcome `SHIP_COST=1.0` × 8 extra
ships = ~8 cost units.

To make hold-aware-opening actually work, the value formula would
need a real per-tick simulator (the LP-side already has this via
`outcome_table._simulate_one`). That's a substantial rewrite of
opening_planner. Likely worth re-attempting as Phase η (search-based
opening) — but as a different SOLVER, not as a coefficient bump on
the existing closed-form.

## Durable findings

1. **Latent LP-pending bug** (`lp-pending-not-deducted-during-opening-fallthrough`)
   is now in `audit/friction.md`. It's a real defect: any path that
   causes opening_planner to return empty during opening will
   re-trigger 11-duplicate-fire behavior. Fix candidates: (a) enable
   `LP_PENDING_AWARE_BUDGET=1` by default (band-aid), or (b)
   de-duplicate in `commit_persistent.commit()` (root fix). Both
   need their own A/B.

2. **Promotion candidate rule**: when adding a gate to a front
   pipeline stage that can return empty, instrument the
   fallthrough rate before/after the change. Latent downstream
   bugs are masked by the front stage's normal output.

3. **Don't trust short smoke when both variants look identical.**
   The seed-42 60-step smoke showed both `hold_aware_on` and
   `_off` firing 6 ships at step 1. Should have been a
   "mechanism not activating, inspect" signal, not a "proceed to
   A/B" signal.

## Open questions

- Is the LP_PENDING_AWARE_BUDGET=1 default flip safe? Test file
  `tests/test_lp_pending_aware_budget.py` has positive tests but
  no A/B has confirmed it doesn't regress on the existing
  ladder-evidenced opponents. Future session.
- Is `commit_persistent.commit()` de-dup the right semantic, or
  could it mask legitimate intentional double-commits? Worth
  thinking through before implementing.
- Does the opening's value formula limitation generalize to the
  Phase η search-based opening idea? Phase η would replace the
  formula entirely with a per-tick rollout — addresses this at
  the root.
