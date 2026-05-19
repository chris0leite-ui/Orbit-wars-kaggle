# Archive: 2026-05-17 HANDOVER content (clean baseline + first-action recommendations)

Archived 2026-05-19 by `claude/ml-competition-strategy-PFhzM` per
WRAPUP step 5 (HANDOVER.md > 150 lines). Superseded by the Day-19 PM
section pivot to clean ROI + scenario gate; the 5/17 next-session
first-action recommendations (architectural pivot on baseline,
opening book, calibration-probe submit) are no longer current.

The 5/17 architectural work (`agents/baseline/` clean modular
re-baseline of v15) IS still load-bearing — `agents/baseline/` is the
current live-champion source and `tests/test_baseline_*.py` remain
green. What's archived is the NARRATIVE around that work, not the
work itself.

---

## What just landed (2026-05-17, claude/kaggle-baseline-strategy-lO4mm)

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

Local validation (5/17 results):
- unit tests (23 cases): green in ~3 s
- `fast.py bench baseline` (3 games / 557 turns): p50/p95/max within v15's published envelope
- `fast.py eval baseline` (n=64 vs v7_0_drop_one): PASS (Wilson lo > 0.55)
- `fast.py eval baseline --vs /tmp/v15_resurrect/main.py` (n=64): INCONCLUSIVE — CI brackets 0.50 = **functional parity with v15**

## Stale next-session first-action recommendations (5/17 vintage)

These three recommendations were SUPERSEDED on 2026-05-19 by the ROI
pivot. Preserved here for traceability.

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
2. **Map-type-conditional opening book** (H40, ~4 h). 4 board
   archetypes identified earlier; tier-1 experiment = override
   proposer's first 30 turns with a cluster-specific template. Gate:
   ≥55% Wilson on 3-agent panel + h2h vs v15 baseline.
3. **Submit the clean baseline as a calibration probe** (~20 min) —
   PI-approved single-shot. Expected outcome: functional parity with
   v15, but a clean live data point against the live-drift WARNING.
   Costs: evicts v20 from rolling-last-2 (v15 stays).

Note (5/19): item (3) effectively happened when composite+A2 hybrid
was bundled and pushed (sub 52744856); live μ TBD as of archive time.
Items (1) and (2) were displaced by the ROI/scenario-gate pivot —
the chooser/value-head axis on `agents/baseline/` is back on the
table after the ROI experiment if ROI underperforms.
