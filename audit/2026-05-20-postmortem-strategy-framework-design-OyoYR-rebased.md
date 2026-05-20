# Postmortem — 2026-05-20 strategy-framework-design-OyoYR-rebased

> Session wrap-up postmortem. Decision-quality framing per
> `knowledge-base/concepts/decision-quality-vs-outcome-quality.md`.
> Multi-session arc (started 2026-05-19) closes today with the
> analytical-chooser axis exhausted.

## What went wrong

### Bad decisions (would not retake given same priors)

1. **Slice 8c wait_N filter — rushed without full introspect read.**
   When Slice 8 showed 0.75 emits/turn, I hypothesised "wait_N>0
   candidates dominate the top of per-source rankings, lock the
   source, emit nothing." Built a one-line filter, ran small A/B,
   verdict 3/16 (vs Slice 8's 6/16) — strictly worse. The introspect
   trace I'd already collected showed **all 7 positive-Δ candidates
   from src=8** at step 0, with no positive candidates from other
   sources. That's a per-source-allocation observation, not a
   wait_N one. Filtering wait_N didn't change the per-source
   dedup that was the actual bottleneck. Reconstructable from the
   priors; decision quality: poor.

2. **Suggested "relax the Δ > 0 emit gate" as a fix.** When Slice 8c
   regressed, I proposed loosening the gate to allow small-positive
   moves. PI override (mid-session): "Δ=+0.0 is a feature, not a
   bug — the candidate space is incomplete." This was a real
   conflation: "the analytical chooser does nothing this turn"
   ≠ "the analytical chooser is broken." Loosening would have
   injected noise back into a clean substrate. Promoted to
   improvements.md as candidate Rule 42.

### Architectural pattern (cross-session, ≥7 slices)

3. **Stacking analytical commits on a rollout chooser is noise.**
   Demonstrated across Slices 4 (predicates-as-priors backstop),
   5 (bounded-interval dominance), 6 (LP commit-as-hint), 8c
   (wait_N filter on differential), 9 (migration solver), 10
   (joint LP chooser). Every "add analytical layer on top of
   the existing chooser" attempt either matched baseline or
   regressed. The rollout was doing implicit planning via its
   leaf-favor evaluation; analytical commits override its
   decisions, not augment them. The architectural diagnosis
   arrived late (mid-Slice-10) but is now durable. Promoted as
   candidate Rule 43.

### PI overrides this session

1. "Δ=+0.0 is a feature, not a bug" — corrected my "broken
   chooser" framing.
2. "Don't throw them away if they're statistically
   indistinguishable" — guided the preservation strategy
   (Wlo ≥ 0.30 keep cutoff applied across all slices).
3. "We must not fail to succeed because of some bug" — drove
   Slice 10's 38-test load-bearing test coverage.
4. (Earlier) Rule 41 promotion: "inspect first, small A/B
   second" — added to CLAUDE.md after a 45-min n=32 was
   started before single-game inspect.

### Rule-bypass failures

None identified. Rule 41 (inspect first, small A/B) was
followed consistently this session. The wait_N filter rush
(decision 1 above) was a failure to USE the introspect
output deeply enough, not a rule bypass — Rule 41 says
"inspect first," and I did inspect; I just didn't read
the implication carefully before acting.

### Rule-gap failures

- "Don't loosen analytical math when it correctly says no
  action" — implicit in Rule 40 (modeling > restriction-tuning)
  but specifically applied to the chooser INPUT (proposer
  filters). Not previously applied to chooser OUTPUT (emit
  gates). Promoted to candidate Rule 42.

- "Analytical work either replaces substrate or stays in input
  layer — never stacks on top of a rollout's decisions."
  Not previously articulated. Promoted to candidate Rule 43.

## Frictions logged this session

See `audit/friction.md` § 2026-05-20:

- `tag: analytical-zero-not-bug`
- `tag: stack-not-replace-analytical-on-rollout`
- `tag: per-source-distribution-vs-class-filter-misread`
- `tag: introspect-script-stale-after-architecture-change`

## Promotion candidates (PI ratified: YES for both)

1. **Candidate Rule 42** — "Closed-form zero is honest, not
   broken; investigate the candidate space, not the gate."
   Promoted to `.claude/skills/kaggle-comp/improvements.md`
   pending CLAUDE.md insertion.

2. **Candidate Rule 43** — "Analytical work either REPLACES the
   chooser substrate or stays in the heuristic input layer;
   don't stack it on top of a rollout's decisions." Promoted
   to `improvements.md` pending CLAUDE.md insertion.

Both candidates are pending — they live in `improvements.md`
under the "Pending" section until a future audit-pass session
promotes them into CLAUDE.md (matching the pattern of how Rules
38 and 41 reached CLAUDE.md).

## PI additions (from step 4)

PI asked to "promote the two candidates" — no additional
frictions, no rule modifications. Postmortem stands as drafted.

## What's preserved on the dev branch (for next session)

All analytical work stays in-tree as opt-in research:

- `agents/baseline/predicates.py` (W1/W2/L1/L2 + value bounds + dominance)
- `agents/baseline/migration_solver.py` (capture-EV + ship migration)
- `agents/baseline/strategic_lp.py` (Hungarian assignment infrastructure)
- `agents/baseline/chooser_differential.py` (closed-form leaf eval)
- `agents/baseline/chooser_lp.py` (joint-LP chooser)
- `agents/baseline/chooser_layered.py` (composition adapter)
- `scripts/layer0_introspect.py`, `scripts/differential_introspect.py`
- 6 audit docs covering Slices 4-10
- 130+ unit + property tests across all modules

Production unchanged: `BASELINE_CHOOSER=trajectory` remains the
ladder agent. Rolling-pair floor μ=1118.8 preserved.

## What might be next session (PI to decide)

Three plausible directions, NOT a commitment:

1. **Continue analytical work on a different axis** — multi-turn
   LP, value calibration audit, opp-counter-projection. Each is
   a multi-slice undertaking.
2. **Pivot to trajectory-side improvements** — better opp model
   for the rollout, leaf-eval caching, opp-policy-class
   selection.
3. **Park analytical experiments; focus on submission
   strategy** — given 33 days to deadline (2026-06-23), the
   marginal hour might be better spent on ladder strategy,
   submission timing, opponent-class fingerprinting.

## Framework version at session-end

- Branch: `claude/strategy-framework-design-OyoYR-rebased`
- Commit SHA (pre-wrap): `b5c8bbe`
- Active CLAUDE.md rules: 1..41 (Rule 41 added this session-line
  on 2026-05-19; Rules 42 and 43 candidates pending in
  improvements.md).
- Loaded skills this session: `kaggle-comp`, `postmortem`.
- Days to deadline: 33 (2026-06-23 23:59 UTC).
