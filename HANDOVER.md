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

## What just landed (2026-05-17, A2 follow-up)

After PR #27 merged, public-notebook research (Rule 22 plateau scan)
reordered priorities. Key findings:

- `romantamrazov/orbit-star-wars-lb-max-1224` (peak μ=1224, +109 above
  our ceiling) — pure heuristic mission portfolio, ZERO ML. Decomposes
  into 6+ named missions + 4P weakness exploitation + inter-enemy
  dynamics + indirect wealth map + adaptive modes.
- `konbu17/orbit-wars-rule-base-ml-shot-validator-hybrid` — 24→64→32→1
  MLP, BCE loss, base64 NPZ embed. +19 pp local lift on 8.8k examples;
  we have 37k examples (but `labels.parquet` is gitignored — need to
  regenerate from a replay corpus that's also absent).
- `aidensong123/lb-highest-1000-search-learned-value-function` — GBC
  value head, caps at LB 1000+ (*lower* than us → PARKED).

**A2 (4P weakness exploitation) landed in `agents/baseline/value.py`:**

  ELIMINATION_BONUS=55, WEAK_ENEMY_THRESHOLD=110
  WEAKEST_ENEMY_MULT_4P=1.5, ELIMINATION_GATE_RATIO=0.9

4P only. In 4P, opps aggregate by WEIGHTED-sum (weakest 1.5x). Elim
bonus +55 when weakest_strength ≤ 110 AND my_strength ≥ 0.9× theirs.
2P path is UNCHANGED from the original baseline (after the 2P uniform
bias attempt was rolled back).

**Failed attempt: 2P uniform 1.25x bias.** Tested as part of A2; h2h
vs v15 (n=64) showed 25 wins (39.1%, Wlo=0.281, Whi=0.513, INCONCLUSIVE
verdict per the plan's Wlo>0.50 hard gate). The bias made the chooser
over-aggressive against v15's well-tuned strategy. Rolled back.

**Process change:** CLAUDE.md Rule 27a codifies h2h-vs-rolling-champion
as the FIRST submission gate (n≥64, Wlo>0.50). Closes the
panel-misleads-h2h friction (4 prior recurrences).

**Validation:**
- unit tests (28 cases): green
- bench (3 games / 505 turns): p50=91 p95=261 max=347ms (in envelope)
- smoke vs random both seats + vs nearest: PASS
- h2h vs v7_0_drop_one (baseline gate): not re-run (A2 4P-only, no
  expected change in 2P)
- h2h vs v15 (n=64): INCONCLUSIVE (39.1%) — but expected since A2 is
  now 4P-only and fast.py harness is 2P-only

**Not submitted.** A2's 4P lift cannot be validated via fast.py's 2P
harness. Need 4P FFA panel or self-play 4P games to gate it.

## Next-session first-action (ranked by EV / cost)

1. **Validate A2 in 4P FFA panel** (~1 h). `scripts/ffa_panel.py`
   --focals baseline,v15_bundle --background v7_0,v4_planner,v3.5.1
   --seeds 32. If A2 has clear higher first-place rate than v15, A2's
   4P weakness mult is the lift it was designed to be. If FFA is also
   parity/regression, A2 doesn't lift on its own and the strategy is
   structurally bound up in romantamrazov's larger mission portfolio.

2. **Stage 4 — mission portfolio subset port** (~2-4 days). Port
   romantamrazov's `build_elimination_missions` + `build_gang_up_missions`
   onto baseline's chooser. ~800-1000 LOC. Higher cost than initial
   400 LOC estimate but ceiling is the public-notebook +109 μ pattern.
   Gate: h2h vs v15 n=64 Wlo>0.50 (2P) + 4P FFA panel.

3. **B3 — train MLP shot validator** (~6-12 h end-to-end, blocked on
   data gen). Need to either pull top-LB replays via Kaggle replay API
   (per-episode rate-limited) or generate via self-play with
   `scripts/generate_selfplay_replays.py` (self-mimicking labels —
   weaker signal than konbu17's gold labels). Then train the
   24→64→32→1 MLP, integrate as `agents/baseline/validator.py`
   post-filter at threshold 0.4.

4. **Resurrect v9_scavenge as h2h gate baseline.** μ=1119.9 (>v15's
   1112.8) but evicted from rolling-last-2. Bundle from git history
   and use as additional h2h gate — our actual team peak.

5. **B2 (PARKED) — opp_traj + CRN principled refactor.** Still good
   engineering; the cross-game audit (84% v8 losses = mid_economy)
   argues for it. But public-notebook evidence says action-space
   architecture is the dominant gap; B2 is downstream of that.

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
  earlier session wrap-up (baseline parity).
- `knowledge-base/thoughts/2026-05-17-public-notebook-research-and-A2.md` —
  public-notebook research findings + A2 implementation + 2P-bias rollback.
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
