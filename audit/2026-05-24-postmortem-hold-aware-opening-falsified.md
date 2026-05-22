# Postmortem — 2026-05-24 hold-aware-opening falsified

> Session: continuation of `claude/strategy-axis-decision-3437`. Two
> commits attempted: initial hold-aware-opening (1/16 vs orbitfix),
> then scoped fix (2/16). Both below the 4/16 LP-only baseline. Net
> action: revert.

## What we tried, what happened

**Hypothesis.** The Phase ζ.v2 LP-side hold-aware (commit `5db11f2`)
shipped at Gate 5 = 4/16 vs orbitfix. Reasoning: LP runs post-step-30
but orbitfix wins by step 60 via opening tempo. The Phase ζ.v2-opening
plan extended hold-aware to `opening_planner.opening_plan`'s closed-form
value math — make `_expected_hold_duration` continuous in `capture_residual`
so bigger fleets get scored higher.

**Initial result (commit `9a19306` — first version, unscoped).**
Gate 5 vs orbitfix: **1/16 = 6.2%** (Wilson [0.011, 0.283]). A −3
regression vs the 4/16 LP-only baseline. Same-seed comparison: lost
3 prior wins (0/P1, 2/P1, 4/P1). Not noise.

**Investigation (per-turn introspect, seed 0/P1).** Used the prior win
turned loss as the diagnostic seed. Trace at step 12:

  - Step 11 observation: my home `pid=15` has 21 ships.
  - Step 12 observation: `pid=15` has 1 ship.
  - Step 12 action: **12 fires from src=15, ships=21 each**, angles
    nearly identical (1 to pid X via angle=−3.109, 11 to pid Y via
    angle=3.128). Total = 252 ships from a planet with ≤22 ships.

The env rejected most; one or two landed. The pattern of
**duplicate-emission-from-same-source-same-tick** is the signature
of `commit_persistent` decanting accumulated wait_N>0 fires.

**Root cause — two-stage interaction.**

1. **Over-pessimistic opening_planner.** The new
   `_predict_opp_counter` considered ALL opp planets regardless of
   speed/distance, while the legacy `_predict_opp_ships_at_target`
   filters to opps within `OPP_RESPONSE_LAG=4` ticks of arrival. ON
   over-rejected candidates that OFF correctly ignored. Result:
   `_build_candidates` rejected more candidates → opening_planner
   returned empty schedules during opening phase (waterfall
   `dropped_defense: 48` out of 8 surviving targets).

2. **Latent LP-without-pending-aware bug exposed.** When
   opening_planner returns empty during step < 30, the pipeline
   falls through to `decision_outcome_aware_milp`. The bundle defaults
   `LP_PENDING_AWARE_BUDGET=0` (`lp_outcome.py:342` reads
   `os.environ.get("LP_PENDING_AWARE_BUDGET", "0")`). So the LP at
   each of steps 1..11 sees src=15 with full ships and commits a
   wait_N>0 fire targeting fire_step=12. `commit_persistent.commit()`
   appends without de-duplication. By step 12, the pending list has
   11 accumulated `ScheduledFires` for fire_step=12; they all decant
   plus the LP's own wait_N=0 fire at step 12 = 12 emissions.

The opening_planner output being NORMALLY non-empty during the
opening phase masks this latent bug in production. Hold-aware-opening
broke the normal output → exposed the bug.

**Scoped fix (commit `9a19306` rebased).** Restricted
`_predict_opp_counter` to threats within `arrival + OPP_RESPONSE_LAG`
(same window as the legacy gate). Verified: seed-0 step-by-step
actions byte-identical to LP-only baseline through step 14. The
12-duplicate emission is gone.

**Gate 5 with scoped fix:** **2/16 = 12.5%** (Wilson [0.035, 0.360]).
Still below the 4/16 LP-only baseline. Same-seed comparison: lost
3 prior wins (0/P1, 2/P1, 4/P1), gained 1 (4/P0). Net −2 wins.

## Why the mechanism doesn't help (even with the bug fixed)

The opening's value formula is

    value = production × hold_dur × opp_bonus × γ^(wait + flight)

The hold-aware change makes `hold_dur` continuously depend on
`capture_residual` (= `ships_fired − target_garrison`). Bigger
fires → bigger residual → bigger hold_dur → bigger value. The MILP
should pick bigger fires.

What actually happens: under the scoped predicate, the closest opp
source (CLOSEST by ETA) is often the SAME as the legacy's max-ships
opp source within the `OPP_RESPONSE_LAG=4` window. The two
quantities are nearly identical for the orbitfix seed set. The
binary gate `counter_ships >= residual + 3` fires at the same
thresholds with ON and OFF. `hold_from_counter` differs from the
Stage-2 eta-delta cap only when the closest opp is closer than the
threat_eta returned by the world model — also rare in the opening
because opp fleets aren't yet in flight.

Result: byte-identical opening behavior on seed 0; small divergence
on a few other seeds; net result is noise, not lift. The closed-form
value formula `production × hold_dur` simply doesn't have enough
expressiveness to model multi-wave threats. Bigger residual hold
durations don't dominate the value computation enough to flip MILP
picks toward bigger fleets.

The right fix would replace `_expected_hold_duration` with the full
post-capture per-tick simulator (as Phase ζ.v2's LP-side does via
`outcome_table._simulate_one`). That's a substantial rewrite of
`opening_planner` and was out of scope.

## Bad decisions (given priors at decision-time)

- **Stacked the opening change on top of the LP change in one
  commit.** Should have shipped LP-side (`5db11f2`) alone and run
  multi-opponent panel before adding the opening axis. Rule 21
  (family falsification needs ≥3 variants) was technically
  satisfied (unscoped + scoped = 2 variants on the predicate-scope
  axis), but the higher-level "did opening-side hold-aware help?"
  question was answered by 2 variants both below baseline. Could
  have stopped at the 1/16 result without the scoped retry; the
  retry confirmed the mechanism direction but didn't change the
  verdict.

- **Trusted the small-n smoke pre-introspect.** A seed-42 60-step
  smoke showed both `hold_aware_on` and `hold_aware_off` firing 6
  ships at step 1. That's a SIGN the opening mechanism isn't
  diverging — should have triggered "inspect before A/B" not
  "proceed to A/B." Cost: 16 games × ~150s = 40 min of compute on
  a directionally-wrong A/B.

- **Reverted (intermediate) without an A/B confirming parity-
  restoration.** Mid-investigation I reverted the opening_planner
  changes once, then re-applied them with the scope fix. Should
  have run Gate 5 on the reverted state mid-session to confirm
  the prior 4/16 still reproduces. (Doing it now as part of this
  postmortem revert.)

## What we learned

**Promotion candidates (PI ratify or scrap):**

- **`bundler-fallthrough-amplifies-latent-bugs` rule candidate.** When
  the production agent has multiple decision stages with a fallthrough
  semantic (opening → LP → commit_persistent), the LATENT stage's
  bugs are masked by the FRONT stage producing valid output. Any
  change to a front stage that increases its empty-return rate WILL
  expose downstream latent bugs. **Mitigation:** for any new gate
  added to a front stage, instrument the fallthrough rate before/
  after and assert it's stable.

- **`opening-value-formula-too-coarse` ledger entry.** The opening's
  `production × hold_dur × γ^t` is structurally insufficient for
  ship-count discrimination. To meaningfully prefer bigger fleets,
  the value model needs the per-tick simulator (multi-wave threat
  resolution). The closed-form formula is fine for the binary
  capture-or-not decision but not for sizing.

**Friction log entries (already in `audit/friction.md`):**

- `lp-pending-not-deducted-during-opening-fallthrough` — the LATENT
  bug. **Real defect**, independent of hold-aware. Two fix candidates
  for a future session: enable `LP_PENDING_AWARE_BUDGET=1` by default
  in the bundle (~half-day plus its own A/B), or de-duplicate in
  `commit_persistent.commit()`. The latter is the actual bug fix;
  the former is a band-aid.

## What we didn't do

- Did not enable `LP_PENDING_AWARE_BUDGET=1` by default. Separate
  axis, separate A/B (Rule 21).
- Did not de-duplicate in `commit_persistent.commit()`. Same reason.
- Did not pivot to `agents/baseline/` substrate. Plan's kill criterion
  ("the LP+opening family is structurally exhausted") was triggered;
  next session may pivot.

## Final action

Reverted commit `9a19306` (manually preserved the
`lp-pending-not-deducted-during-opening-fallthrough` entry in
`audit/friction.md`). Bundles rebuilt. Closed track added to
`state/MULTI_BRANCH.md`. Audit indexed.

Gate 5 parity re-test running on the reverted bundle; result
appended below when complete.

### Parity gate result

**Gate 5 parity re-test (post-revert) vs orbitfix at n=16:**

```
focal_wins=3/16 (18.8%)  Wilson[0.066, 0.430]  elapsed=698s
```

Wins on seeds 2/P0, 2/P1, 4/P1. Lost seed 0/P1 (the prior 457-step
deep-game win; flipped at step 463 — well within stochastic-variance
range for a long game). Compared to the prior pre-`9a19306` LP-only
result (4/16, Wilson [0.102, 0.495]), the CIs heavily overlap and
the −1-win difference is within noise.

**Verdict: parity restored.** The revert is a clean restoration of
the LP-only Phase ζ.v2 baseline. No regression vs the ~1100
submission's expected behavior.

### Comparison table

| Variant | Gate 5 | Wilson CI | Note |
|---|---|---|---|
| LP-only baseline (5db11f2, pre-9a19306) | 4/16 | [0.102, 0.495] | The ~1100 submission's baseline |
| LP + opening hold-aware (unscoped, 9a19306 initial) | 1/16 | [0.011, 0.283] | −3 wins, exposed latent LP bug |
| LP + opening hold-aware (scoped, 9a19306 rebased) | 2/16 | [0.035, 0.360] | −2 wins, bug fixed but mechanism null |
| LP-only restored (post-revert, this commit) | **3/16** | **[0.066, 0.430]** | Parity with baseline (noise-overlap) |
