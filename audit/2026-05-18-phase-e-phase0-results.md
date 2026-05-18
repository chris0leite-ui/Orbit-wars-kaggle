# Phase E Phase 0 — Bundle-vs-baseline failure-mode breakdown

**Date**: 2026-05-18
**Branch**: `claude/ml-competition-strategy-PFhzM`
**Config**: post lead-aim revert (commit `4bd2eec`); bundle = cands=5 default,
lead-aim OFF, ME_FOLLOWUP=off.
**Run**: `scripts/diag_bundle_baseline_failures.py --seeds 8 --workers 4`
(8 seeds × 2 sides = 16 games, ~9 min wallclock).
**Detail dump**: `audit/2026-05-18-phase-e-phase0-failures.json`.

## Aggregate (16 games, 1578 fleets, 65,939 ships launched)

| Bundle wins | 2/16 = 12.5% |  | |
|---|---|---|---|
| **Bucket** | **% ships** | **fleets** | **role** |
| captured | 50.3% | 691 | productive |
| reinforced_self | 26.9% | 470 | productive |
| **bounced_enemy** | **21.3%** | **368** | **load-bearing waste** |
| bounced_neutral | 0.9% | 26 | trivial |
| arrived_but_lost | 0.6% | 20 | trivial (recapture) |
| hit_planet_unknown_flip | 0.1% | 2 | noise |
| comet_collision | 0.0% | 1 | noise |
| sun / oob / vanished | 0.0% | 0 | none |
| alive_at_end | 0.0% | 0 | none |

| Roll-up | % ships |
|---|---|
| PRODUCTIVE (captured + reinforced) | 77.2% |
| WASTED (bounce + recapture + transit) | 22.8% |
| IN-FLIGHT at end | 0.0% |

## PI gate ratification

Literal gate (Phase E plan):
- ≥30% wasted → PASS (efficiency axis)
- <15% wasted → FAIL (target-selection axis)
- 15-30% → GRAY (PI ratifies)

Result lands in the gray band at **22.8%**. PI ratified PROCEED to Phase 1
based on the breakdown: **21.3% of all bundle ships bounce off enemy
planets** — a single directly-addressable bucket. A 30% spread across
many buckets would be less actionable than 21.3% concentrated in one.

## Implications for the Phase E plan

1. **Phase 1 (joint coordination bonus) is exactly on target.** The 368
   bounced_enemy fleets are bundle's would-be partners — the ones that
   should have ganged up into successful joint captures. Phase 1's
   explicit joint enumeration + scorer bonus addresses this directly.

2. **Phase 2 (bounce penalty) addresses the residual.** Of the 368
   bounces, some won't have a partner available at all (single
   over-committed launch). Bounce-penalty prevents these by making
   solo bounces score below empty.

3. **Phase 3 (compound-ROI weighting) remains relevant.** The 50.3%
   captured + 26.9% reinforced are productive but undifferentiated by
   value. Compound weighting helps the chooser prefer high-prod
   capture-and-hold over equally-large but low-prod or short-hold
   captures.

4. **Phase 4 (recapture-risk discount) is DEFERRED.** At 0.6%
   arrived_but_lost, recapture is below the n=16 measurement noise
   floor. PI decision: keep Phase 4 listed-but-deferred; revisit
   AFTER Phase 1-3 A/Bs in case captures grow more aggressive and
   recapture rate rises.

## Verification properties for Phase 1 design

Phase 1's expected effect, observable on the same diagnostic:
- bounced_enemy % drops (some bounces convert to joint-captured)
- captured % rises (the converts)
- bundle_wins increases (the goal)

If Phase 1 lifts captured % significantly but bundle_wins barely
moves, the captures are landing on low-strategic-value targets —
that's the signal that Phase 3 (compound ROI) is needed.

If bounced_enemy drops sharply but captured doesn't rise (i.e.,
bundle simply launches less), Phase 1's joint enumeration is
under-reaching — Phase 2 bounce penalty may be needed first.

## Per-game results

| seed | seat | reward | n_steps | fleets |
|---|---|---|---|---|
| 1 | 0 | -1 | 281 | 59 |
| 7 | 0 | -1 | 127 | 94 |
| 42 | 0 | -1 | 178 | 92 |
| 31 | 0 | -1 | 217 | 80 |
| 100 | 0 | -1 | 210 | 52 |
| 13 | 0 | -1 | 356 | 128 |
| 23 | 0 | -1 | 265 | 44 |
| 17 | 0 | **+1** | 293 | 169 |
| 1 | 1 | -1 | 154 | 42 |
| 42 | 1 | -1 | 180 | 121 |
| 7 | 1 | -1 | 174 | 85 |
| 100 | 1 | -1 | 191 | 67 |
| 31 | 1 | **+1** | 303 | 173 |
| 13 | 1 | -1 | 356 | 132 |
| 17 | 1 | -1 | 500 | 192 |
| 23 | 1 | -1 | 315 | 48 |

The 2 wins (seeds 17 P0, 31 P1) used higher fleet counts (169, 173)
than the median (~90), suggesting more launches per turn. Not pursued
here — out of scope for Phase 0.

## Artifacts

- `audit/2026-05-18-phase-e-phase0-failures.json` — per-game data,
  per-fleet outcomes, aggregate buckets.
- `scripts/diag_bundle_baseline_failures.py` — reusable for
  before/after measurement after each Phase 1-3 A/B.
