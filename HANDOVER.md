# HANDOVER.md — next-session brief

> Last written: 2026-05-17 by `claude/kaggle-baseline-strategy-lO4mm`
> (clean modular re-baseline of v15).
> Prior session: `claude/recover-main-foundations-MV0e2` and
> `claude/merge-2026-05-16-knowledge` (the v9 → v15 → v20 chooser line).

## Where we are

- **Comp:** Orbit Wars. Deadline 2026-06-23 23:59 UTC.
- **Team peak agent:** v15_banded (multi-wait-grid + banded
  (src, tgt, wait_band) dedup), shipped 5/16 as submission #52710995.
  **Do NOT hardcode the live μ here** — it drifts as the rolling
  rating settles. Query Kaggle at session start:
  `kaggle competitions submissions orbit-wars`.
- **Rolling-last-2** (Kaggle auto-keeps these two for final eval; the
  third push auto-evicts):
  - v20_dogpile (5/16 22:00 UTC, sub #52721807) — most recent
  - v15_banded  (5/16 14:00 UTC, sub #52710995) — current champion
- **Daily submission budget:** 5/day; 5/17 starts at 0/5 used.
- **Calibration WARNING** (still active): multiple recent submissions
  over-predicted live by 20–30 pp. Every new push needs a 3-opponent
  local panel (`fast.py eval --vs-panel`) PLUS h2h vs the current
  rolling champion (not just a fixed baseline) — closes the
  `panel-pass-without-h2h-vs-current` and `local-overpredict-2x`
  frictions.
- **v15 source** is NOT in the working tree — the "Bootstrap: nuke
  historical strategy code" reset wiped it. It lives at
  `f315dc7:agents/v15/main.py` (787 LOC) in git history. The clean
  modular re-implementation at `agents/baseline/` is the working
  foundation for everything going forward.

## What just landed (2026-05-17, this branch)

`agents/baseline/` — clean modular re-baseline of v15 in 577 LOC.

```
agents/baseline/
├── value.py    (60 LOC)   F1 + F2 favor leaf with pv_horizon discount
├── proposer.py (262 LOC)  multi-wait extra_surplus grid + banded dedup
├── chooser.py  (132 LOC)  reactive-opp idle baseline + per-cand Δ + emit
└── main.py     (123 LOC)  entry + env-var knobs + pipeline glue
```

Backed by the same proven primitives v15 used: `lib/fast_sim.py`
(0.12 ms/step), `lib/opp_model.lite_greedy_policy` (reactive),
`lib/scoring.pv_horizon`, `lib/world_model.WorldModel`. None of `lib/`
was modified. Env-var knobs: `BASELINE_GAMMA` (default 0.99),
`BASELINE_WALLCLOCK_MS` (default 600), `ORBIT_WARS_PARITY_WALLCLOCK_MS`
(bundle-parity override).

`tests/test_baseline_*.py` — 5 files, 26 test cases:
- `test_baseline_value.py` — F1+F2 monotonicity + PV-discount + 4P sum-of-opps
- `test_baseline_proposer.py` — wait-grid + banded dedup + capture_size + sizing
- `test_baseline_chooser.py` — reactive baseline length + score_action + emit shape
- `test_baseline_smoke.py` — vs random both seats + per-turn budget (skip if no env)
- `test_baseline_h2h.py` — gated on `BASELINE_RUN_H2H=1` (n=16 vs v7_0_drop_one)

Local validation (run again at session start to refresh):
- unit tests (23 cases): green in ~3 s
- `fast.py bench baseline` (3 games / 557 turns): p50/p95/max within v15's published envelope
- `fast.py eval baseline` (n=64 vs v7_0_drop_one): PASS (Wilson lo > 0.55)
- `fast.py eval baseline --vs /tmp/v15_resurrect/main.py` (n=64): INCONCLUSIVE — CI brackets 0.50 = **functional parity with v15**

**Not submitted.** Submission is single-shot per Rule 1 and needs PI approval.

## Next-session first-action (ranked by EV / cost)

0. **PI-designated next-session task: per-geometry-class priority prior.**
   Read `knowledge-base/concepts/per-class-priority-prior.md` end to
   end before anything else. The 2026-05-19 audit
   (`audit/2026-05-19-archetype-per-planet-class.md`) showed top-10
   over-allocates +10 pp of fleet share to `low_prod_rotating_inner`
   and we over-allocate to `high_prod_static_*`. The design doc
   contains the alpha table, opponent-posterior calculation,
   combining formula, injection point (`proposer.py::cheap_marginal_value`),
   default coefficients (`lambda_alpha=3`, `lambda_gap=2`), a six-question
   preflight, a validation plan, and a next-session checklist. The
   PI's framing: this is **prior weights that get updated by what
   the opponent is doing**, giving the agent orientation on which
   planets matter most. v1 ships from the closed-form prior — no IL,
   no training compute. Tasks 1–3 below are the previous handover's
   leftover candidates, lower priority than task 0.

1. **Architectural pivot on top of baseline** (~1 day). The v9–v15
   chooser axis is structurally saturated (Rule 37 cap hit at v16–v20).
   The clean modular split lets you swap ONE of value / proposer /
   chooser / opp_model independently. Highest-EV candidates:
   - **Learned value head** replacing `agents/baseline/value.favor`:
     `lib/value_heads.composite_capture_value` already exists; train
     a small head on replay corpus or use the existing logistic
     regression weights (Mine 2 hit 0.77 AUC).
   - **Portfolio search** in `chooser.py`: enumerate 3-5 named
     portfolios (incumbent / conservative / aggressive / no-op /
     drop-weakest) and score each — different action-space topology
     from drop-one.
   - **IL warm-start** from top-10 replays — `data/shot_validator/`
     already has 37k labeled examples (24-dim); the MLP head is
     deferred but the pipeline is ready.
   Pre-flight: Rule 16 6-question check; Rule 19 issue-tree claim.
2. **Map-type-conditional opening book** (H40, ~4 h). 4 board
   archetypes identified earlier; tier-1 experiment = override
   proposer's first 30 turns with a cluster-specific template. Gate:
   ≥55% Wilson on 3-agent panel + h2h vs v15 baseline.
3. **Submit the clean baseline as a calibration probe** (~20 min) —
   PI-approved single-shot. Expected outcome: functional parity with
   v15, but a clean live data point against the live-drift WARNING.
   Costs: evicts v20 from rolling-last-2 (v15 stays — it's the
   second-most-recent of `[baseline, v15]`).

## Pointers

- `agents/baseline/` — clean modular re-baseline of v15 (this branch).
- `tests/test_baseline_*.py` — 26 baseline tests.
- `lib/fast_sim.py`, `lib/game/interpreter.py` — the bit-exact forward
  simulator + game-rule engine; do NOT rewrite.
- `lib/opp_model.lite_greedy_policy` — the reactive opp model used in
  both `agents/baseline/chooser.build_idle_baseline` and `score_action`.
- `state/current.md` — submitted-agent state (no μ values; query Kaggle).
- `state/mechanism-ledger.md` — every agent family tried.
- `state/hypothesis-board.md` — open ideas (H40, H42) + killed list.
- `audit/2026-05-16-v15-final-results.md` — v15's panel + h2h.
- `audit/2026-05-16-v16-v20-asymmetric-compounding-postmortem.md` —
  the v15→v20 chooser saturation iteration; Rule 37 application.
- `knowledge-base/thoughts/2026-05-17-baseline-functional-parity-with-v15.md` —
  this session's wrap-up.
- `knowledge-base/concepts/per-class-priority-prior.md` — next-session
  design doc (PI-designated task 0).
- `audit/2026-05-19-archetype-per-planet-class.md` — per-archetype ×
  per-class rollup that drives the alpha table in the design doc.
- `lib/per_planet_class.py` — 8-class binning helpers
  (`classify_planet`, `compute_board_medians`, `ALL_CLASS_LABELS`).
- `fast.py` — single-file iteration entry: smoke / bench / eval / play.
- `scripts/bundle_agent.py` — bundler for submissions.

## Rule reminders

- Rule 1: submissions are single-shot, PI-approved. No retry loops.
- Rule 12: rolling-last-2 — Kaggle auto-keeps last 2 submits for final
  eval. Never push a speculative variant after a known-good submit
  unless you're willing to lose the good one's ladder spot.
- Rule 27 analogue: h2h vs the **current submitted agent** (not just
  a fixed baseline) is the FIRST gate, not the LAST. Panel pass alone
  is insufficient (v17, v18 lost h2h vs v15 despite panel pass).
- Rule 32: session-start `kaggle competitions submissions orbit-wars`
  is the source of truth for μ. State files do NOT record μ.
- Rule 37: 3-variant axis cap. The v9–v15 chooser axis hit it; future
  work must pivot to a different axis.
- Rule 40: prefer modeling-correctness over restriction-tuning
  (no MAX_WAIT / MAX_HORIZON / MIN_FLEET_SIZE bumps to fix symptoms).
