# simple/production — economy-first targeting

> File: `agents/simple/production.py`.
> Role: production-greedy variant in the simple-strategy panel.
> Status: local-only. Strong 8-seed signal (75% mean panel winrate);
> awaits 32-seed confirmation before submission consideration.

## One-liner

For each owned planet, pick the **highest-production non-owned
planet** as the target. When several targets tie on production
(production is integer, range [1, 5], so ties are common), break the
tie by distance — closer wins.

## Mechanism

Per turn:

1. Build planet list and split into `my_planets` / `targets` exactly
   as in `simple/nearest.py`.
2. Per-turn RNG seeded by `step ^ (player + 1) * 1009` (the v1 A.6
   anti-mirror seed).
3. For each owned planet, score every target by the tuple
   `(-target.production, distance)`. Sorting ascending gives:
   - Highest production wins (ascending sort over the negated value).
   - Among tied productions, nearest wins (the second tuple element).
   - Among full ties, the per-turn RNG breaks the deadlock.
4. Emit `Intent(src_id, target_id, ships=target.ships + 1)`.
5. `realize(...)` with `DEFAULT_MECHANISMS` adds orbit-aware aim
   and bumps fleet size for production growth on enemy targets.

## Why it works (or doesn't)

In Orbit Wars, production is the only renewable resource on the
board. Captured ships are spent in flight; planets keep generating
ships every turn for the rest of the 500-step game. Capturing a
production-5 planet on step 50 generates 2,250 free ships by step
500; capturing a production-1 planet on step 50 generates 450.

Distance-greedy ignores this gradient entirely: it ranks targets by
travel time, not by long-run yield. Production-greedy inverts that
priority — distance becomes the tiebreaker, not the primary signal.

**Where production-greedy can lose:**

- **Distance pulls hard at the extreme.** If the highest-production
  planet is on the far side of the board, your fleet spends 30+
  turns travelling — during which the enemy may be out-expanding
  you locally. The fixed-point lead helps with aim but doesn't
  change the wallclock cost of the trip.
- **Garrison strength ignored.** A production-5 planet with 99 ships
  is a 100-ship investment for a 5-ship/turn return; the same fleet
  could flip a production-3 planet with 10 ships and start farming
  three turns earlier.

`roi` (production / distance) addresses the first failure mode;
neither simple strategy addresses the second (that's a v2 arrival-
ledger / mission-classifier job).

## Gotchas

- **Ties are common (production is small-integer).** The
  RNG-tiebreaker carries real weight here. If you see the
  per-turn-RNG fix (the A.6 seed) get pulled, expect the
  `production` strategy to start mirroring across self-play and
  introduce P0/P1 winrate asymmetry — the symptom would look like
  "production beats production 8/0 on one seat and 0/8 on the
  other."
- **Production = 0 isn't possible (range [1, 5] per spec).** Don't
  add a `if target.production > 0` guard; it's pure dead code.

## Evidence

8-seed smoke (2026-05-10):
`audit/tournaments/20260510T123059Z.json`.

| vs            | aggregated winrate |
| ------------- | ------------------ |
| `nearest`     | 69% (11/16)        |
| `roi`         | 19% (3/16)         |
| `weakest`     | 100% (16/16)       |
| `enemy_first` | 94% (15/16)        |
| `baseline`    | 100% (16/16)       |
| `v1_orbitfix` | 69% (11/16)        |
| self-play     | 2 P0 / 3 P1 / 3 draws — within ±15% of 50/50 |

Mean panel winrate: **75.0%**. p95 turn ~0.3 ms.

The 69% beat over `v1_orbitfix` is the load-bearing local result.
v1_orbitfix on the live ladder sits at μ=508; if `production` holds
the 69% gap at 32 seeds, the calibration table predicts a Δμ of
roughly +50 to +250 vs v1 on the live ladder — comfortably submission-
worthy, subject to the rolling-last-2 economy (do **not** evict v1.1
before v1.1's μ has settled).

## What it does NOT do

- **No travel-time adjustment.** A production-5 planet 80 units away
  is preferred over a production-4 planet 8 units away, even though
  the latter starts paying off ~10 turns sooner. `simple-roi.md`
  fixes this.
- **No combat budgeting.** `production` does not weigh fleet cost
  against the time-discounted production return — it's pure
  myopic-greedy on the production field.
- **No defence, no fleet coordination, no comet-path lead, no
  sun-avoidance.** Inherited gaps from the simple-strategy panel's
  shared shape; resolved only at v2/v3.
