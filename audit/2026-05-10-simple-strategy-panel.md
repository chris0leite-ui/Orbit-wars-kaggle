# 2026-05-10 — simple-strategy panel (target-selection ablations)

Branch: `claude/simple-trading-strategies-QS0xV`
Plan: `/root/.claude/plans/read-the-handover-next-imperative-whisper.md`
Tournament JSON: `audit/tournaments/20260510T123059Z.json`

## What this exercise does

Five strategies under `agents/simple/` share v1.1's mechanism stack
(`[validate, arrival_size, lead_aim]`). The only thing that differs
across the five is the **target-selection score function** — so any
winrate gap inside the panel is attributable to the targeting axis
alone (mechanism stack is held fixed; one launch per owned planet
per turn is held fixed).

Goal: (a) learn which targeting axis matters before investing in
v2's heavy arrival-ledger work; (b) seed a richer hold-out opponent
panel for D.4; (c) exercise the Strategy abstraction with ≥3
concrete instances.

## How to reproduce

```bash
# Quick iter (≤10 min CPU on this box):
python -m scripts.strategy_panel --seeds 8

# Confidence (32 seeds; ≤45 min CPU):
python -m scripts.strategy_panel --seeds 32

# Filter the panel:
python -m scripts.strategy_panel --strategies nearest production roi
```

Default panel = the five simple strategies + the comp-shipped
`baseline` (`data/main.py`) + the live `v1_orbitfix`. Round-robin,
both seats, self-play included so A.6 P0/P1 asymmetry stays visible.

## 8-seed smoke result (2026-05-10)

7 agents × 7 agents × 8 seeds = 392 games; total wallclock ~9 min.

| Strategy      | Hypothesis                                              | Mean panel WR | vs v1_orbitfix (both sides) | Verdict (8-seed) |
| ------------- | ------------------------------------------------------- | ------------- | --------------------------- | ---------------- |
| `roi`         | argmax `target.production / dist` is the right ROI      | **96.9%**     | **100% (16/16)**            | ✅ strong        |
| `production`  | argmax `target.production`, tiebreak distance           | 75.0%         | 69% (11/16)                 | ✅ confirmed     |
| `nearest`     | (control) reproduces v1's distance-greedy               | 56.2%         | 19% (3/16) — within noise   | ≈ tied with v1   |
| `enemy_first` | enemy planets first, then nearest (pressure on opp)     | 32.3%         | 12% (2/16)                  | ❌ refuted       |
| `weakest`     | argmin `target.ships` (cheap snipes)                    | 15.6%         |  0% (0/16)                  | ❌ refuted       |

Other readings:
- All five simple strategies + v1 beat the shipped baseline in their
  rows.
- `weakest` and `enemy_first` lose to almost everything **except** the
  shipped baseline — they're worse than the panel as a whole but
  remain useful diversity for D.4 (a hold-out opponent that's
  consistently worse-but-different is exactly what the panel needs to
  surface RL-style overfit).
- p95 turn wallclock is 0.3-0.4 ms across all strategies — three
  orders of magnitude under the 1-second budget. The mechanism stack
  is not a wallclock bottleneck at this scale.
- Self-play P0/P1 split (the A.6 sanity check) is well-balanced for
  every strategy: nearest 2/1/5, production 2/3/3, roi 1/0/7 (most
  draws — ROI's symmetry collapses many seeds to ties), weakest 5/3/0,
  enemy_first 0/0/8 (every seed a draw — same target ordering on both
  sides). Tie-break RNG is doing its job per the v1 fix.

## What we learned (axis-level)

1. **Distance-only targeting (`nearest`) is mediocre.** It's the
   control and it ties v1_orbitfix as expected, but it loses to
   `production` 31/69 and to `roi` 0/100. Distance-greedy is leaving
   structural value on the table.
2. **Production-aware targeting is a clear win.** `production` (75%
   panel) beats `nearest` (56%) and v1 (56%) decisively. The lift
   does not require a heavy world-model — argmax over a single field
   on the obs is enough.
3. **Travel-adjusted production (ROI) is the structural answer.**
   `roi` dominates every other agent in the panel, including v1, by
   a margin large enough to consider a v1.2 submission **once
   confirmed at 32 seeds AND v1.1's live μ has settled** (rolling-last-2
   means we can't push speculatively without evicting v1.1).
4. **Pure-weakness targeting (`weakest`) is actively bad.** Sniping
   the smallest garrison ignores production entirely — capturing a
   1-ship rock on the far side of the board is worse than capturing
   a closer 50-ship planet that produces 5 ships/turn.
5. **Pressure-on-opponent (`enemy_first`) is also bad.** Skipping
   neutrals to attack enemies trades fast economy growth for slow
   trench warfare; in a 500-step game the economy compounds faster
   than the trench-warfare value.

## Open follow-ups (queued for next session)

- **H-roi-32 and H-production-32**: rerun the panel with `--seeds 32`
  to pull Wilson CIs from ±9pp to ±5pp before any submission decision.
- **PI-deferred axis follow-up batch.** PI flagged a 6th axis (sizing /
  coordination / defence — not yet specified). After we read the
  32-seed result, PI names the axis and we add the next batch under
  `agents/simple/`.
- **Hedge-ladder construction (Rule R2 / hypothesis-board).** Once
  `roi` and `production` confirm at 32 seeds, populate the hedge ladder
  in `state/hypothesis-board.md` so the rolling-last-2 final pair is
  not wasted on accidentally-correlated submits.

## Files added/changed

New:
- `agents/simple/__init__.py`
- `agents/simple/{nearest,production,roi,weakest,enemy_first}.py`
- `scripts/strategy_panel.py`
- `tests/test_simple_strategies.py` (25 tests, all green)
- `audit/2026-05-10-simple-strategy-panel.md` (this file)
- `audit/tournaments/20260510T123059Z.json` (the 392-game JSON)

Edited:
- `state/hypothesis-board.md` — five new hypotheses + verdicts
- `state/mechanism-ledger.md` — `simple-greedy-target-selection-variants` row
- `state/calibration-ladder.md` — local-only rows for `roi` + `production`
- `ISSUES.md` — claimed B.1.1 leaf
