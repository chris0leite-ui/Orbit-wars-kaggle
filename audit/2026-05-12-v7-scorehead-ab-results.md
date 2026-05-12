# 2026-05-12 — v7_minimax scoring-head A/B (PRODUCTION_WEIGHT=50)

## TL;DR

**Verdict: NEUTRAL. Variant does NOT beat v7 baseline at the 55% gate.**

| Stage | n | Variant wins | Draws | Losses | W-D rate | Wilson 95% W-D |
|-------|---|-------------:|------:|-------:|---------:|----------------|
| Smoke | 8  | 6 | 0 | 2  | 75.0% | (small n — uninformative) |
| Full  | 64 | 32 | 4 | 28 | 56.3% | **[0.441, 0.677]** |

Smoke and full disagree by 19 pp — the 75% smoke was 4-seed noise. The
clean 64-game result places the variant at parity with the v7 baseline:
Wilson lo 0.44 sits below the 0.50 indifference threshold, so we cannot
even reject "variant equals baseline." Promotion gate not cleared.

**Not submitting v7_planetscore.**

## Hypothesis tested

v7_minimax's maximin overlay scores each (our, opp) cell with
`our_ships − opp_ships` at K=3 turns (lib/lookahead.py:_ship_total_by_owner).
At K=3 turns post-launch, in-flight ships dominate the scalar (the
fleets haven't arrived yet), so the maximin systematically prefers
candidates that launch *more*, regardless of where those ships are
going. Hypothesis: replacing the scoring head with a planet-production
composite would align the search objective with the comp metric
(planets at end → TrueSkill ranking) and produce better picks.

Variant change (a single edit in `lib/lookahead.py`):

```python
_PRODUCTION_WEIGHT = 50.0

def _ship_total_by_owner(observation) -> dict[int, float]:
    totals = {}
    for p in observation.get("planets", []):
        owner = int(p[1])
        if owner >= 0:
            totals[owner] += ships + _PRODUCTION_WEIGHT * production
    for f in observation.get("fleets", []):
        owner = int(f[1])
        if owner >= 0:
            totals[owner] += float(f[6])
    return totals
```

PRODUCTION_WEIGHT=50 means one production unit counts as 50 ships of
value (intuition: ~50 turns of remaining game × 1 ship/turn produced).

## Setup

- Worktree off `origin/claude/game-theory-strategy-analysis-0oH4N` HEAD
  (commit `d9b25d6`, post-deterministic-seed-fix).
- **Baseline bundle:** `submissions/v7_baseline.py` 81768 B,
  sha256 `1393d32b1f4e691d`. **Bit-exact match to live submission
  #52568317** — we are A/B-ing against exactly the agent that scored
  μ=1063.0 on the ladder.
- **Variant bundle:** `submissions/v7_planetscore.py` 82399 B,
  sha256 `26a4b1d0a6791be0`. Identical to baseline except the
  `_ship_total_by_owner` body above.
- A/B harness: `scripts/v7_scorehead_ab.py` (smoke=4, full=32 seeds,
  both sides each). Workers=4. Wilson CI from
  `scripts.tournament.PairStat`.
- Seeds-32 bag: `[42, 1, 7, 13, 21, 17, 100, 200, 3, 19, 23, 29, 31,
  37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
  107, 109, 113]`.

## Smoke (n=8, seeds {42, 1, 7, 13})

```
[1/8] baseline (P0) vs variant (P1) seed=7:  p1_win  steps=189  dShips=-1890
[2/8] baseline (P0) vs variant (P1) seed=1:  p0_win  steps=403  dShips=+2126
[3/8] baseline (P0) vs variant (P1) seed=13: p1_win  steps=182  dShips=-3608
[4/8] baseline (P0) vs variant (P1) seed=42: p1_win  steps=500  dShips=-50
[5/8] variant  (P0) vs baseline (P1) seed=7: p0_win  steps=192  dShips=+2038
[6/8] variant  (P0) vs baseline (P1) seed=1: p1_win  steps=403  dShips=-2126
[7/8] variant  (P0) vs baseline (P1) seed=42:p0_win  steps=500  dShips=+60
[8/8] variant  (P0) vs baseline (P1) seed=13:p0_win  steps=199  dShips=+4730
```

- baseline P0 vs variant P1: variant 3W/0D/1L of 4 (variant P1 75%)
- variant P0 vs baseline P1: variant 3W/0D/1L of 4 (variant P0 75%)
- **Combined: variant 6W/0D/2L of 8 = 75%** — balanced across seats,
  not seat-biased.
- p95 turn ms 645/644 — under 750ms hard bail.

This looked decisive. We promoted to 32-seed.

## Full (n=64, 32-seed bag, both sides)

```
baseline (P0) vs variant (P1): P0 15W/2D/15L of 32; Wilson 0.309..0.636
                               mean ΔShips (P0-P1) -185.1
                               p95 turn ms P0=826 P1=828
variant  (P0) vs baseline (P1): P0 17W/2D/13L of 32; Wilson 0.364..0.691
                                mean ΔShips (P0-P1) +560.7
                                p95 turn ms P0=833 P1=821
```

Combined (variant POV):

- **Pure win rate:** 32/64 = 50.0% Wilson 95% **[0.381, 0.619]**.
- **Win+Draw rate:** 36/64 = 56.3% Wilson 95% **[0.441, 0.677]**.
- Mean ΔShips: variant slightly positive in both seats (+560 as P0,
  +185 as P1 from variant's view) → variant *tends* to leave slightly
  more ships on the board even when game outcomes are even.
- p95 turn ms 826/828 — over the 750ms hard bail in this 4-worker
  multiprocess run. The variant DOES occasionally trigger the
  budget-bail (collapsing to v3 row 0); production single-core latency
  would likely be lower, but we cannot confirm without a single-worker
  rerun.

## Why the smoke and full disagree

- **Seed leverage.** The smoke bag {42, 1, 7, 13} happens to favor
  the variant in 3 of 4 seeds (only seed=1 has baseline-wins-both-sides).
  Extending to 32 seeds dilutes the leverage of any individual seed.
- **N=8 Wilson CI is wide.** 6/8 has Wilson 95% [0.35, 0.90] — easily
  consistent with the true rate being 50%.
- **Lesson confirms project Rule 2** — smoke + 1-fold time-probe; do
  not commit on smoke evidence. Adding to friction record.

## Why the variant doesn't lift v7

Three live hypotheses; we did not falsify between them in this A/B:

1. **PRODUCTION_WEIGHT=50 is the wrong magnitude.** With K=3 lookahead,
   a 10-planet board, and per-planet production ~0.5-3.0, the
   variant's planet term contributes 50 × 1.5 × 10 ≈ 750 "ship-equivalent"
   to the score — likely larger than ship totals at K=3 (~100-300
   ships per side). The maximin is now driven almost entirely by
   planet ownership, ignoring ship movements. Probably overshoots.
   A weight of **5-15** would balance ships and planet count rather
   than swamp ships entirely.
2. **K=3 horizon is too short for production-based scoring to fire.**
   At K=3 the planet-ownership delta over 3 turns is at most ±1
   capture flip. The production-weighted scoring barely differs from
   a planet-count scoring at K=3.
3. **σ-equiv + tie-break-to-row-0 dominates the maximin layer.**
   The v7 audit notes the maximin overlay differentiates on only ~5%
   of turns. If those 5% are all in σ-equiv-indifferent regions where
   the new scoring still ties, the change has no effect on play.

## Calibration honesty

Predicted at session start: "variant should beat baseline ≥55% W-D
because the scoring head is the largest single-lever miscalibration."
Outcome: 56% W-D at n=64, Wilson lo 0.44 — at parity. **Prediction
was directionally right (variant non-negative) but magnitude was
wrong.** Adding to calibration ledger as another "Claude overshoots
its prior on local A/B lift, n=64 reality is smaller than n=8
suggests."

## What to do next

Ranked by EV/cost (Rule 6 — heuristics before heavy compute):

1. **Sweep PRODUCTION_WEIGHT** — try 5, 10, 25. Same harness; ~25 min
   per weight at 32 seeds. Cheap. EV: locates the right magnitude if
   #1 above is the bottleneck.
2. **Increase K** — try K=5 with current scoring; K=5 with
   PRODUCTION_WEIGHT=10. Same harness; ~50 min per config. EV: tests
   #2 above. Risk: turn time blows past 1s budget (already at 826ms
   p95 with K=3 symmetric).
3. **Recapture mission (independent of v7 overlay)** — addresses the
   "more ships, still lost" recovery-deficit pattern directly. Lives
   in lib/missions/recapture.py, doesn't touch v7. ~4-6h. Highest EV
   for the user's specific complaint per yesterday's analysis.
4. **Calibration probe** — test whether the local 32-seed A/B
   predicts ladder direction at all. v3.5.1 passed 32-seed at 68.8%
   and regressed -112μ on the ladder. Need to verify the predictor
   before any more local-gate-only submissions.

## Artifacts

- `audit/tournaments/v7-scorehead-smoke-4-20260512T140421Z.json` (n=8)
- `audit/tournaments/v7-scorehead-full-32-20260512T145136Z.json` (n=64)
- Variant source: `/tmp/v7-ab/lib/lookahead.py` (worktree, not on this
  branch — patch is reproducible from the writeup above).
- Variant bundle: `/tmp/v7-ab/submissions/v7_planetscore.py` 82399 B
  sha256 `26a4b1d0a6791be0` (not committed — variant did not pass
  gate, so the bundle is not promotion-ready).

## Rolling-last-2 not touched

We did **not** submit. Current rolling-last-2 remains
`[v3.5.1 (943.1), v7_minimax (1063.0)]`. Next experimental push evicts
v3.5.1 (good); the one after evicts v7 (bad). This A/B preserved that
budget for a real candidate.
