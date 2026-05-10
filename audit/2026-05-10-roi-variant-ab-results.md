# 2026-05-10 — ROI variant A/B test results (Day 1 PM)

> Branch: `claude/improve-strategy-ab-testing-jYA2R`
> Plan: `/root/.claude/plans/yes-ask-clarifying-questions-compressed-peach.md`

## TL;DR

PI flagged five issues with the live `roi` agent (μ=1104.9). I built 12
variants across four design axes and ran a 13-agent × 8-seed exploratory
panel + a 3-agent × 64-seed confidence panel.

**Single-axis winner: `roi_enemy2x`** (one-line change: multiply
production-ROI numerator by `m`, where `m=1` for neutral targets and
`m=2` for enemy targets — i.e. enemy captures count double because they
shift the zero-sum margin both ways).

| | vs `roi` (head-to-head) | Wilson 95% lo | Status |
|---|---|---|---|
| `roi_enemy2x` | 80/128 (62.5%) | **0.538** | Just under 0.55 advance gate |
| `roi_combo` (enemy2x + sun-aware pivot) | 79/128 (61.7%) | 0.530 | Slightly worse than enemy2x alone |

**Recommendation:** do NOT submit on this evidence. Lift is
directionally clear at every seed count (8: 88%, 32: 61%, 64: 62%) but
Wilson 95% lower bound straddles 0.50–0.55, so we can't rule out the
"nothing happened" hypothesis. Per PI's "defend the anchor; push for
top-1% only on strong local lift" posture, hold the slot.

## Design axes tested

1. **Arrival projection** (V1) — score on projected garrison instead of
   current snapshot. `lib/scoring.py` provides
   `eta_proxy / projected_garrison / s_needed / horizon /
   margin_multiplier` reused across variants.
2. **ETA discount** (V2 — geom γ=0.92, hyperbolic λ=0.05, hard cap 30).
3. **Dominance gate** (V3 — α ∈ {1.20, 1.50, 2.00}; only fire if
   `mine.ships ≥ α · S_needed`, else hold).
4. **Sun-aware pivot** (V4) — strategy walks ranked targets, skips
   sun-blocked, plus `sun_avoid` mechanism as final guard.
5. **Margin-EV scoring** (V5 — enemy2x bonus, horizon weighting,
   combined m·P·H − S_needed, enemy-only denial).

## 8-seed panel results (13 agents)

| strategy | mean panel WR | vs `roi` | verdict |
|---|---|---|---|
| roi_enemy2x | **98.4%** | 88% (14/16) | clear winner |
| roi (control) | 85.9% | — | anchor |
| roi_safe | 74.0% | 31% | sun-pivot net negative vs ROI |
| roi_horizon | 72.4% | 12% | bad |
| roi_arrival | 69.3% | 19% | bad — projection alone hurts |
| roi_discounted_cap | 67.2% | 19% | mediocre |
| roi_discounted_geom | 47.9% | 0% | bad |
| roi_margin | 33.9% | 0% | bad |
| roi_discounted_hyper | 33.9% | 0% | bad |
| roi_denial | 23.4% | 0% | bad |
| roi_dominance_200 | 14.1% | 0% | terrible |
| roi_dominance_120 | 13.5% | 0% | terrible |
| roi_dominance_150 | 13.0% | 0% | terrible |

JSON: `audit/tournaments/20260510T182640Z.json`.

## 32-seed panel (5 agents — top contenders)

| | vs roi | vs enemy2x | vs combo | vs safe |
|---|---|---|---|---|
| roi_enemy2x | 39/64 (61%) | — | 25/64 (39%) | 64/64 (100%) |
| roi_combo | 38/64 (59%) | 19/64 (30%) | — | 64/64 (100%) |
| roi (anchor) | — | 25/64 (39%) | 26/64 (41%) | 49/64 (77%) |

`roi_safe` collapsed (0/64 vs both winners). Sun-avoid is net negative
even with the target-pivot fallback at 32 seeds.

JSON: `audit/tournaments/20260510T183225Z.json`.

## 64-seed confidence panel (3 agents)

| | vs roi | vs enemy2x |
|---|---|---|
| roi_enemy2x | 80/128 (62.5%) — Wilson [0.538, 0.703] | — |
| roi_combo | 79/128 (61.7%) — Wilson [0.530, 0.696] | 42/128 (33%) |

JSON: `audit/tournaments/20260510T183754Z.json`.

## Key takeaways for the next session

1. **Margin-aware scoring is real, but only the simplest form works.**
   The 1-line `m * production` numerator beats every more elaborate
   variant. The full margin EV `m·P·H − S_needed` (`roi_margin`) loses
   heavily — likely because subtracting cost overcommits to high-cost
   far targets that get reinforced by arrival.
2. **Sun-avoidance is not the bottleneck PI thought it was.** Even the
   target-pivoting `roi_safe` loses to `roi`. Sun-eaten fleets are not
   a measurable lift opportunity at this strategy level.
3. **Dominance gating is universally bad in this regime.** Holding
   garrison while waiting for "clear win" lets the opponent expand
   freely. ROI's "just send 1 over" sizing — combined with
   `arrival_size`'s production-aware bump — already wins enough
   captures.
4. **ETA discounting is bad.** Geometric and hyperbolic both lose;
   only the hard cap is mediocre. The arrival-projection in
   `lib.scoring` already accounts for far targets via the
   `projected_garrison` term.
5. **Pure denial loses badly.** Neutrals are too valuable to skip
   early-game.

## What to test next (deferred)

- **Variants of m**: try m=3, m=1.5 for enemy. The 2x is theoretically
  motivated (zero-sum) but the empirical optimum could differ.
- **Combine enemy2x with arrival-projection** more carefully —
  `roi_enemy2x` reuses `projected_garrison` so this is partly already
  in. Try tightening to m·P / G_now (no projection) as a control.
- **Replay analysis** of `roi vs roi_enemy2x` 64-seed games to see
  WHERE the 12% lift comes from (turn-of-decision delta between
  the two scores). PI explicitly deferred this; surface for the next
  session.
- **A second submission slot tomorrow** — if the answer is to submit
  `roi_enemy2x`, push it after midnight UTC for the fresh quota.

## Infra delivered

- `scripts/tournament.py` — `workers` param for parallel runs
  (mp.Pool, fork). 4× speedup on this 4-core box. `_THIS_MODULE` pin
  defends against pytest's repeated `spec_from_file_location` reloads.
- `scripts/strategy_panel.py` — `--workers N` flag; SEEDS_32 extended
  to 64 entries for confidence runs; `_resolve_agent_path` auto-finds
  any `agents/simple/<name>.py`.
- `lib/scoring.py` — `eta_proxy / projected_garrison / s_needed /
  horizon / margin_multiplier`. 7-test unit module.
- `tests/test_fixture_smoke.py` — added 2 parallel-runner tests
  (determinism gate + callable-rejection guard). Full suite: 160 green.
- 12 strategy variants under `agents/simple/roi_*.py` + `roi_combo.py`.
