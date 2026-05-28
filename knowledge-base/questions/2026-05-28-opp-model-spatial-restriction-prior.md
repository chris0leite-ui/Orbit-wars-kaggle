# Question — does `lite_greedy_policy` systematically over-estimate opp counter-attacks?

**Opened:** 2026-05-28 PM3
**Owner:** next session
**Status:** open

## The question

Inside `chooser_trajectory.choose_trajectory`, when scoring a candidate
launch, `lite_greedy_policy` simulates opp counter-attacks as part of
the rollout's opp side. Currently it considers ALL opp planets as
potential counter-attack sources, regardless of `eta(opp_planet →
our_target)`. PI's diagnosis (PM2 handover): the agent "expects
opponents from everywhere," causing fleet sizes to be under-counted.

**Does `lite_greedy_policy` predict more opp ships at our target than
actually arrive in real games? By how much?**

## Why it matters

If predictor over-estimates by >30%, then:
- Every candidate is scored against an inflated opp counter-pool.
- Fleet sizing comes out too large (we strand ships) OR captures get
  rejected (we don't launch at all → "silent turns" pathology).
- Spatial-restricted opp model (Item 3 from handover) is the modeling
  fix per Rule 40.

If predictor is within 10%, then opp-model is well-calibrated and
Item 3 won't lift — pivot to Item 4 (commit-to-hold sizing) or
proposer-side work instead.

## How to answer (cheap diagnostic)

~1-2 hours of work:

1. In `agents/baseline/chooser_trajectory.py`, instrument the rollout
   to log per-candidate: `(target_id, eta, predicted_opp_ships_at_arrival)`.
2. Run 10 self-play games at seeds {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} with
   `BASELINE_WALLCLOCK_MS=800`. Log the per-candidate predictions to
   a JSONL file.
3. Post-game, for each prediction, find what actually happened: did
   that target get hit by opp ships within `eta + 5` turns? If so,
   how many?
4. Compute the per-candidate prediction error: `actual - predicted`,
   normalised by `predicted + 1`. Plot the distribution; report mean
   and 90th percentile of relative over-estimate.

## Decision rule

- **Mean over-estimate > 30%**: Item 3 (spatial restriction) is the
  next build. High prior.
- **Mean over-estimate 10-30%**: borderline; consider Item 3 OR a
  lighter-touch fix (decay the counter-attack contribution by
  `1 / (1 + eta_normalised)`).
- **Mean over-estimate < 10%**: opp-model is well-calibrated; pivot
  to Item 4 (commit-to-hold) or proposer-side work.

## Linked

- `audit/2026-05-28-postmortem-pm3-macro-layer-null-result.md` —
  next-steps section.
- `knowledge-base/thoughts/2026-05-28-pm3-macro-null-opp-model-pivot.md`
- `HANDOVER.md` PM2 next-action item #3 (opp-model spatial restriction).
