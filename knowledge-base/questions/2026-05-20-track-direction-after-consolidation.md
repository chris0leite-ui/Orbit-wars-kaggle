# 2026-05-20 — open questions for PI after the consolidation pass

These were surfaced in HANDOVER.md "Open PI questions" but warrant
permanent KB filing per Rule 36.

## 1. Track A (Analytical chooser) — park or pivot?

OyoYR-rebased knowledge-base entry 2026-05-20: *"analytical-chooser
axis closed (10 slices, 0 lift)."* Both live pushes regressed (806.5
+ 829.1 are the current rolling pair). Architectural finding from
the day's postmortem: stacking analytical layers on a rollout chooser
is noise — the rollout's implicit 25-30 tick planning horizon
dominates any single-turn analytical layer.

Options:
- (a) Park the track entirely; preserve `chooser_roi.py` as opt-in
  research code via `BASELINE_CHOOSER=roi`.
- (b) Pivot to multi-turn analytical glue (DP / rolling LP / receding
  horizon) — analytical with horizon matching the rollout's.
- (c) Pivot to analytical-leaf-inside-rollout — analytical as
  acceleration of rollout's leaf evaluation, not replacement of the
  rollout itself.

(c) is the highest-EV repurpose of the substrate built so far; (b) is
the most ambitious but lowest-confidence.

## 2. Track C (Verify-first + goal-directed) — wrap-baseline-as-veto or substrate-only?

Day-19 PM2 verdict: *"chooser axis confirmed neutral"* — greedy_expand
(60 LOC) tied goal_planner (500 LOC). The only positive signal across
10+ Track-C iterations: wrap-baseline 12/32 = 37.5%.

Open question: does the wrap-baseline asymmetry indicate a viable
"augment baseline with a portfolio veto layer" design, or is Track C
contributing substrate-only (lib/trajectory_layer.py, the analytics
verification suite) and the chooser-level work should be abandoned?

## 3. Recovery submission lineage — which to rebundle?

Three strong evicted lineages with documented A/B + live μ:
- composite_a2_hybrid (#52744856, μ 1149.2) — team peak; bundler had
  silent import bug at sibling #52744234 ERROR.
- trajectory v4 + wait_N + wallclock (#52754310, μ 1143.7).
- hold-feasibility solo (#52811320, μ 1135.1) — solo-validation A/B
  still pending on btjeK Phase B.

PI must choose before the next push (Rule 42 claim board + Rule 43
panel + h2h gate).

## 4. Priority sequencing for the next session

Two priorities compete:
- Code consolidation pass (start small with substrate primitives:
  `lib/trajectory_layer.py` + `agents/precision/`).
- SessionStart hook implementation (improvements.md TOP PRIORITY,
  needs code).

The hook gives every future session a fresh baseline; the consolidation
pass unblocks Track-B and Track-C work. PI sequencing call open.
