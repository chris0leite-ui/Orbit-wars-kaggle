# Opp-axis has a compute/quality Pareto wall at 1000ms turn cap

Date: 2026-05-30 (claude/kaggle-submission-review-gZsCu).
Inputs: paired-seed asymmetric A/Bs of nearest, top_tier_mirror, v7_0,
and no-launch as the chooser's internal rollout opp model, all baked
into anchor-clone bundles to avoid the env-var symmetry trap.

## The finding

Among rollout opp-policy variants we can test inside the 1000ms turn
cap, `lite_greedy_policy` sits on the cost/quality Pareto frontier.
Three data points stake the curve:

| Opp model | per-call cost | per-turn evals | result vs lite_greedy |
|---|---|---|---|
| lite_greedy   | 0.01 ms                       | ~1,100 | baseline |
| nearest       | ~0.01 ms (same loop)          | ~1,100 | 3-2 lite_greedy (n=5, preview-null) |
| no-launch     | ~0 µs                         | ~1,100 | 3-1 lite_greedy (n=4) |
| top_tier_mirror | ~5-10 ms (5-10× lite_greedy) | ~80    | 5-0 lite_greedy (n=5, compute-bound) |
| v7_0 full agent | 101.6 ms (10,000× lite_greedy) | ~15  | not run (bench-falsified) |

The chooser's `affordable_validate_cap` (`agents/baseline/chooser.py:140`)
sizes per-turn candidate evaluation against the per-leaf cost. When
the opp policy is heavy, the cap collapses to its 8-candidate floor;
the chooser's search starves before policy quality can matter.

## Why this matters

PM3 named the opp model "the most under-explored lever." Today's
work falsifies that framing under the *current compute budget*. The
lever exists — no-launch beat lite_greedy 1 of 4 seeds, meaning the
chooser's belief about opp does materially change emit decisions —
but the budget closes off the heavy variants. Two structural fixes
are conceivable:

1. **Fast opp policy** — port v7_0/v3.5.1-class logic to a sub-0.5 ms
   representation (tabular lookahead, compiled C, distilled neural
   net). This is a multi-day engineering project, not an A/B session.
2. **Budget pad** — set BASELINE_WALLCLOCK_MS to 5000+ for testing
   purposes only, run mirror, isolate strategy claim. Cannot ship to
   ladder (1000ms env cap), but tells us whether the engineering
   project is worth it.

Both of these are open-ended; the session-budget A/B sweep cannot
resolve them.

## What this lets us close

- The "is lite_greedy the right opp model?" question is *budget-
  conditional* now answered: among feasible alternatives, yes.
- The "expects opponents from everywhere" PM3 diagnosis is not
  testable cheaply — spatial-restricted lite_greedy is the cheapest
  remaining variant in that direction, hasn't been A/B'd, would
  cost the same as the no-launch run (~5 min n=4) and probably
  produces the same precision result.
- The nearest result (3-2) is direction-only; the no-launch result
  (3-1) is a cleaner control. Neither tells us anything about the
  *direction* of lite_greedy's mismodeling — only that there IS one.

## Methodology note (load-bearing for any future opp-axis run)

Three bake-asymmetric A/Bs today; each required:
- Copy `submissions/baseline_pv_eta_anchor_1163.py` to a new file.
- Patch `_select_opp_policy()` to return the variant.
- Optionally inject the variant function inline (nearest, no-launch)
  or import dynamically (v7_0).
- Pre-bench per-turn invocation count vs lite_greedy. Ratio >2× is
  a compute confound; ratio >5× is mechanically determined.
- Run paired-seed A/B (n=4 wave-aligned with 4 workers, or n=5 if
  willing to pay a 2-wave wallclock).

This pattern works for any chooser-internal-policy swap (opp model,
value head, proposer mechanism). The env-var route (BASELINE_OPP_MODEL,
BASELINE_VALUE_HEAD) is fine when only one bundle reads it, but for
anchor-vs-anchor isolation, baking is required.

## What I'd want to revisit

- Seed 3493 in the no-launch A/B is anomalous: lite_greedy LOST to
  no-belief there, while winning on the other 3 seeds and on the
  same seed against nearest and mirror. A trace through this seed
  would tell us *how* lite_greedy over-predicts threat in that
  specific board geometry — and would point at the modeling fix.
  Filed as a question.

- Spatial-restricted lite_greedy (eta ≤ 15 cap on candidate targets)
  is the cheapest remaining variant directly tied to PM3's
  diagnosis. ~5 min n=4. Would either firm up "opp-axis is Pareto-
  closed" or surface a new lift candidate. Not chased this session
  because the no-launch result already told us opp-axis is alive
  but lite_greedy is hard to beat.
