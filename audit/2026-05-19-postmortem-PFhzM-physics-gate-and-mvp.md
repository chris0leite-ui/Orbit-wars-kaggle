# Postmortem — 2026-05-19 PM2 (claude/ml-competition-strategy-PFhzM)

Session continued from the 5/19 PM v4 pivot. Built Phase B cluster
work + goal_planner Phase 1-4 + physics-validation gate +
greedy_expand MVP. NO Kaggle submission. **Eight A/Bs run, six 0/32
vs the live Kaggle baseline.**

## What went wrong

- **Five distinct architectures all 0/32 vs Kaggle baseline.** Phase B
  veto-on-trajectory_roi, Phase B hybrid, goal_planner (no validation),
  goal_planner (with validation), greedy_expand MVP. Different chooser
  layers; same outcome. Prior session shipped 5 trajectory_roi
  iterations with similar pattern (0-1/32). Building-from-our-primitives
  axis is dominated regardless of how the chooser is structured.
- **Physics-validation gap discovered at session-end, not start.**
  Four A/Bs were burned against agents leaking ~7% of launches into
  the sun. `lib.trajectory.predict_fleet_fate` exists in the repo
  (baseline.py uses it via `lib/mechanism.py:593,686,775`) but our
  entire experimental line (`trajectory_roi`, `cluster_solver`, all
  Phase B variants, `goal_planner` v1) never imported it. Discovered
  when PI asked: "what about our whole physics and trajectory modeling,
  do we use it?" Trace evidence: replay-probe (`scripts/probe_emits_via_fate.py`)
  shows trajectory_roi has ~6.8% physics-wasted launches (1.0% sun,
  5.8% OOB); goal_planner after the validate gate has 0%.
- **Synthetic-scenario tests gave false confidence.** All 17
  goal_planner unit tests passed on constructed geometries — agent
  still produced physically invalid launches at runtime. The
  constructor (me) chose tight-east geometries where the sun was
  never in the trajectory. Same pattern would have caught us in cluster
  solver tests too (`tests/test_cluster_solver.py:35-77` all use
  clear-line geometry).
- **Duplicate-emit bug in hybrid agent.** `find_solvable_clusters`
  returned 19 overlapping clusters containing planet 12 at one observed
  turn (planet 12 was hub-positioned near 6 neighbors → C(6,1)+C(6,2)
  = 21 combinations). Hybrid concatenated each cluster's solver action
  → emitted 5× identical 15-ship launch from a 15-ship planet (75 ships
  requested from a 15-ship budget). Tests didn't cover hub-position
  geometry. Caught only via single-game inspection script.
- **MVP confirmed the chooser layer was strategically neutral.**
  `greedy_expand` (60 LOC, no predicate/portfolio/sequencer/defense)
  tied `goal_planner` (~500 LOC, full stack) at 14/32 vs each other
  (Wilson [0.282, 0.607]). The entire architectural overlay added no
  measurable value at this primitive layer.

## PI overrides this session

- **"3 games before any A/B"** (greedy_expand smoke). Process discipline
  preventing yet another premature n=32 burn. The 3-game smoke showed
  greedy_expand survives 344 turns vs baseline (vs goal_planner's
  elimination at ~123) — informative without paying for n=32 upfront.
- **"What about our whole physics and trajectory modeling — do we use
  it?"** Turn-around moment. Surfaced the foundation gap. Without
  this question I would have continued patching goal_planner instead
  of asking why a 50-LOC greedy might match it.
- **"Something here is really off. How can it be that we have this fast
  analytic machine and we cannot get anything out of it?"** Drove the
  MVP framing. Made me strip until ONE testable claim of value emerged
  rather than keep adding layers.

## Frictions logged this session

See `audit/friction.md` under `## 2026-05-19 PM2` heading. New tags:
- `physics-primitives-not-used-by-our-line` — predict_fleet_fate
  exists, only baseline imports it.
- `synthetic-scenarios-miss-constructor-blind-spots` — 17/17 tests
  green; agent broken at runtime.
- `detector-overlapping-clusters-overcommit` — hub planet → C(6,2)
  clusters → 5× duplicate emit.
- `chooser-architecture-strategically-neutral` — 60 LOC ≈ 500 LOC
  at the level we're operating.
- `building-from-scratch-dominates-0-of-32` — five architectures,
  same outcome.

## Promotion candidates (PI ratified 1, 2, 3 ✓; 4 not promoted)

1. **Rule 41 candidate (carried from 5/19 PM session, RE-CONFIRMED):**
   "verify analytics before iterating on chooser." Five more iterations
   this session on a chooser stack sitting on physically broken
   primitives. **Ratified for promotion.**

2. **New rule: physics-validation gate is mandatory.** Every emit must
   round-trip through `predict_fleet_fate` (or equivalent ground-truth
   check) before reaching the env. Mirror baseline.py's `lib/mechanism.py`
   gating pattern. Origin: this session, ~6.8% sun/OOB waste in
   experimental line. **Ratified for promotion.**

3. **Amend "scenarios are the gate" principle.** Synthetic scenarios
   ALONE are insufficient. Need synthetic + replay-position oracle
   (every primitive's output checked against `predict_fleet_fate` or
   equivalent on a sample of real replay positions). Origin: this
   session, 17/17 tests green while agent shipped broken. **Ratified
   for promotion.**

4. ~~**"Wrap > replace when a strong live base exists."**~~ Not
   promoted. Worth keeping in this postmortem as a session observation;
   may be re-evaluated next session if pattern recurs.

## PI additions

> (none — PI ratified 1/2/3 directly without further additions)

## Framework version at session-end

- Commit SHA: `ed4cd8e` (greedy_expand MVP)
- Branch: `claude/ml-competition-strategy-PFhzM` (ahead 118 / behind 21 vs origin/main)
- Active rules: 1-40 (CLAUDE.md Top-level rules); promoting #41 (verify
  analytics before chooser iteration) per ratification above.
- Loaded skills this session: postmortem, kaggle-comp (implicit).
- This session's commits (newest first): ed4cd8e, bdb8fea, f91da00,
  d497dd6, 8537fdc, 29af281, 98c6f2c, c1f9428, 03b2708, c90b69a,
  2568ffa, 0e82657, 52a68d1.
