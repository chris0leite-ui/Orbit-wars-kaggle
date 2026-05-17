# HANDOVER.md — next-session brief

> Last written: 2026-05-17 by `claude/kaggle-baseline-strategy-lO4mm`
> (clean modular re-baseline of v15).
> Day-N PM addendum: `claude/improve-fleet-efficiency-cQXg4` —
> 7 variants across 2 axes (chooser-filter + opening-overlay) all
> falsified at n=32 vs v15. See `## Day-N PM improve-fleet-efficiency-cQXg4`
> section below.
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

## Day-N PM improve-fleet-efficiency-cQXg4

Session goal (from PI): minimise wasted fleets — fleets that fail
their missions (defending, capturing, sabotage). Investigation +
sustainable fix on evaluation and choice among candidate actions.

**Diagnostic landed:** mined 16 v15/v20 live replays (8 each).
Empirical waste rates:
- **15% of launches target comets; 100% MISS** (~20 ships/game).
- **60–70% of CAPTUREs lost back within 50 turns**; median hold = 8.
- **43–53% of lost-backs are UNDEFENSIBLE** (outnumbered locally
  at recapture in R=30 neighborhood).

**Iterations tried (all FAILED at n=32 vs v15):**

| Variant | Axis | Approach | n=32 winrate |
|---|---|---|---:|
| v21 | chooser filter (3-layer) | A (joint-emit) + E1 (cheap target-quality prefilter) + E2 (rollout hold-check) | 31.2% Wlo=0.18 |
| v21_a/_ae/_solo | chooser filter | ablations of v21 | 43.8% n=16 |
| v22 | rollout opp | counter-recapture in lite_greedy at every rollout step | 25.0% Wlo=0.13 |
| v23 (w=15) | opening overlay | propose_opening_missions for turns 0..15 of 2P games | 15.6% Wlo=0.07 |
| v23 (w=10) | opening overlay | retry with smaller window per plan's falsification path | 25.0% Wlo=0.13 |

**Conclusion:** v15 is structurally the local optimum for this
codebase. Surface modifications across the chooser-filter and
opening-overlay axes lose by 25–35 pp. Patterns identified:
- `pattern-overlay-on-tuned-baseline-doesnt-lift` (3× recurrence)
- `launch-rate-is-symptom-not-cause`
- `explicit-rewrite-of-implicit-behavior` (2× recurrence — promoted)
- `n16-falsely-shows-parity` (Wilson CI width 0.45 hides 20pp regressions)

**Artifacts on the branch:**
- `agents/v{21,21_a,21_ae,21_solo,22,23}/` + dependency variants
- `agents/v23` short-circuit pattern + `lib/missions/opening.py`
  `window` parameter (backward-compatible)
- `audit/2026-05-17-v21-pivot.md` (v21/v22 postmortem)
- `audit/2026-05-17-v23-postmortem.md` (axis-pivot postmortem)
- `scripts/instrument_v21.py`, `scripts/diag_v21_vs_v15.py` (diagnostics)
- 5 new friction entries in `audit/friction.md`

**Live ladder unchanged through session:** v15 floor μ≈1112; v20
rolling μ≈1094; v9_scavenge ceiling μ≈1120 unbreached.

**Next-session recommendation (after PI direction):** stop iterating
on v15. Wholesale architectural pivot — three candidates:
1. **Portfolio search across multiple value heads** (smallest lift)
2. **Imitation learning from top-10 replays** (biggest upside, multi-day)
3. **4P-specific chooser** (~36% of ladder games, orthogonal axis)

No submissions burned this session.
