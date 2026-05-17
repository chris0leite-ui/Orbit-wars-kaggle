# 2026-05-17 — Baseline is functional parity with v15; the new foundation to build from

**Branch:** `claude/kaggle-baseline-strategy-lO4mm`
**Author seat:** clean-rebaseline session
**Status:** code green, tests green, ready to merge into main as the working foundation.

## What landed

`agents/baseline/` — a clean modular re-implementation of v15 (the
multi-wait-grid + banded-(src, tgt, wait_band) dedup line) in 577 LOC
across 4 files. Backed by the same proven primitives (`lib/fast_sim.py`,
`lib/opp_model.lite_greedy_policy`, `lib/scoring.pv_horizon`,
`lib/world_model.WorldModel`) — none of `lib/` was modified.

```
agents/baseline/
├── value.py    (60)   F1 + F2 favor leaf with pv_horizon discount
├── proposer.py (262)  multi-wait extra_surplus grid + banded dedup
├── chooser.py  (132)  reactive-opp idle baseline + per-cand Δ + emit
└── main.py     (123)  entry + env-var knobs + pipeline
```

26 unit / smoke / h2h tests in `tests/test_baseline_*.py`.

## Why this isn't a "new strategy" — and why that's the right move

v15 hit the live-rolling ladder as the team champion on 5/16 (verify via
`kaggle competitions submissions orbit-wars`; do not hardcode the μ here
since it drifts). The v16 → v20 iteration line then hit the Rule 37
3-variant axis cap on the chooser axis without breaking past the v15
ceiling — the published audit `2026-05-16-v16-v20-asymmetric-compounding-postmortem.md`
documents the structural saturation.

The right move at this junction was NOT to attempt a 21st chooser
variant. It was to consolidate v15 into a clean foundation that:
1. **Lives in the working tree** (v15's source was wiped by the
   "Bootstrap: nuke historical strategy code" reset; it survives only
   at `f315dc7:agents/v15/main.py` in git history).
2. **Is modular** — each of value / proposer / chooser / opp_model is a
   single small file you can swap independently, so the next
   architectural pivot (learned value head, portfolio search,
   IL warm-start) doesn't fight a 787-LOC monolith.
3. **Is small enough to grok in one sitting** (≤262 LOC per file; total
   ~3,000 LOC of codepath including reused lib/ primitives, vs
   v7_0_drop_one's 7,318 LOC tree).
4. **Has tests at the unit, smoke, and h2h levels** — refactors and
   ablations can move with confidence.

## Validation outcomes

Local h2h vs `submissions/v7_0_drop_one.py` (n=64, fast.py eval):
**PASS** — Wilson lo > 0.55. In the same ballpark as v15's published
panel result vs v7_0_drop_one.

Local h2h vs the resurrected v15 source (n=64): **INCONCLUSIVE** — the
Wilson CI brackets 0.50 / 0.55. Read this as **functional parity**: a
clean re-implementation of v15 should be statistically indistinguishable
from v15, and that's exactly what the data shows. We're not better
than v15, and we're not worse — we're v15 with the same primitives,
cleaner code, and ready-to-extend modular seams.

Per-turn timing (long eval, 256 games, ~50k turns total): p95 < 700 ms,
max ~1.3 s (same tail-risk profile as v15 — PI-accepted documented risk
on heavy mid-late-game turns; not a regression introduced by the
re-implementation).

## Risks documented forward

- **Same wallclock tail-risk as v15.** Heavy mid-late-game turns can
  exceed the 1000 ms env actTimeout when the in-flight fleet count is
  high. v15's commit message documents this. The two-stage scoring
  (cheap pre-rank → fast_sim validate) + adaptive `affordable_validate_cap`
  bound the worst case but do not eliminate it.
- **The 5/16 calibration WARNING is still active.** Multiple recent
  submissions over-predicted live by 20–30 pp. Every new push needs a
  3-opponent local panel AND h2h vs the current rolling champion. The
  `panel-pass-without-h2h-vs-current` friction (audit/friction.md) has
  recurred 4× and is now codified in `fast.py eval --vs-panel` discipline.

## What this unlocks

The clean modular foundation makes the next architectural pivot a
**single-axis change**:

- **Learned value head** → replace `agents/baseline/value.favor` with a
  trained head. `lib/value_heads.composite_capture_value` already
  exists; logistic-regression Mine 2 hit 0.77 AUC on capture outcomes.
- **Portfolio search** → replace `agents/baseline/chooser.choose`'s
  drop-one enumeration with a 3-5 named portfolios scorer.
  Different action-space topology; no proposer change needed.
- **Reactive-opp ensemble** → replace `lib.opp_model.lite_greedy_policy`
  with a 2-3 model ensemble (lite_greedy + v3.5.1 mirror + maybe
  baseline-itself for self-play coverage).
- **IL warm-start** → train a small policy net on top-10 replays;
  `data/shot_validator/` has 37k labeled examples already.

Each lives in ONE file; tests for the other modules still pass; Rule 37
axis discipline is preserved because each architectural pivot is its
own axis.

## Hard-won principles re-affirmed this session

1. **Kaggle is the source of truth for live μ.** State files are
   directional maps — they record submission IDs, dates, and statuses
   (immutable facts), not μ values (which drift). Today the
   `state/current.md` file was stale and pointed at v7_pv as champion;
   the correct champion was v15. Direct query closed the gap.
2. **`lib/fast_sim.py` is the precision physics + fast brain.** It
   wraps the bit-exact `lib/game/interpreter.py` (a pure-Python port
   of `kaggle_environments.envs.orbit_wars`) and clocks ~0.12 ms/step.
   Do not rewrite either. They're parity-gated by
   `tests/test_fast_sim_parity.py` and `tests/test_game_parity.py`.
3. **Functional-parity h2h IS a valid validation outcome.** A clean
   re-implementation should be statistically indistinguishable from
   its blueprint — not better. Wilson CI bracketing 0.50 is the
   correct signal, not a failure mode.
4. **Rule 40 (modeling-correctness over restriction-tuning) was
   respected throughout.** No MAX_WAIT / MAX_HORIZON / MIN_FLEET_SIZE
   constants were nudged. The same v15 knobs (NUM_TARGETS_PER_SOURCE=8,
   `WAIT_EXTRA_SURPLUS = (0, 5, 12)`, wait_band = {0, 1..7, ≥8},
   K=10, γ=0.99, wallclock 600 ms) are inherited unchanged.
