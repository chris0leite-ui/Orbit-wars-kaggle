# simple/roi — travel-adjusted production targeting

> File: `agents/simple/roi.py`.
> Role: ROI-greedy variant in the simple-strategy panel; the 8-seed
> standout.
> Status: local-only. **Strongest 8-seed signal in the panel:**
> 96.9% mean panel winrate, 100% (16/16) vs `v1_orbitfix`.
> Awaits 32-seed confirmation before submission consideration —
> rolling-last-2 economy means we cannot evict the live `v1.1`
> submission lightly.

## One-liner

For each owned planet, pick the target with the **highest
production-per-distance** ratio. Production rewards captures that
keep paying off; distance penalises captures that take forever to
arrive. Their ratio is the cheapest hand-coded approximation of
"expected return on this fleet."

## Mechanism

Per turn:

1. Build planet list, split into `my_planets` / `targets` (same
   shape as the rest of the panel).
2. Per-turn RNG seeded by `step ^ (player + 1) * 1009`.
3. For each owned planet, score every target by:
   ```
   roi   = target.production / (distance + 1.0)
   score = (-roi, distance)
   ```
   Sorting ascending: highest ROI wins, with closer-target as the
   tiebreaker. The `+ 1.0` in the denominator is defensive against
   coincident planets (distance = 0); it shifts the ROI curve
   imperceptibly for the typical 5-90 unit distances on the board.
4. Emit `Intent(src_id, target_id, ships=target.ships + 1)`.
5. `realize(...)` with `DEFAULT_MECHANISMS` finishes the work.

## Why it works (or doesn't)

ROI = production / distance is the simplest formula that
**simultaneously** values long-run yield (production) and time-to-
first-payment (distance). Distance-greedy and production-greedy each
optimise one of those axes alone; ROI optimises both at once.

The 8-seed panel result is dramatic: `roi` beats every other
strategy in the panel — `nearest` 100/0, `production` 81/19,
`weakest` 100/0, `enemy_first` 100/0, `baseline` 100/0,
`v1_orbitfix` 100/0. Self-play converges to 7 draws + 1 P0 win out
of 8 seeds; the score function is symmetric enough that two ROI
agents on the same map race to the same captures and stalemate.

**Why ROI is so strong against v1_orbitfix specifically:**
v1_orbitfix already has the orbit-aware lead and the production-
aware sizing (via `arrival_size`). The only remaining lever is
**which target to pick first**. ROI's preference order shifts the
early-game expansion toward high-yield-per-step planets; once those
production engines are spinning, the ship count compounds away from
v1's distance-greedy pace. v1 cannot recover the gap because the
production gradient doesn't reverse.

**Where ROI can lose:**

- **Garrison cost is still ignored.** A ROI-5 planet with 99 ships
  is a poor first capture even though its long-run yield is
  excellent — by the time you've paid 100 ships and waited 10+ turns
  to start collecting, the opponent has captured three production-3
  planets with 10 ships each. `roi` does not see this; it sees only
  the production / distance ratio.
- **Comets exploit the same formula.** Comets have production-1 and
  finite lifetime; ROI's denominator-only model treats them like any
  short-lived neutral. A comet captured for the last 10 steps of its
  trajectory is wasted ships. The mechanism layer's `comet_aim` is
  off by default (loses 22.5% in ablation; documented in
  `lib/mechanism.py`); ROI does not currently adjust for that.
- **Mirror-symmetric stalemate in self-play.** 7 of 8 self-play seeds
  end as draws. If two `roi` agents face off on a symmetric map with
  the same RNG draws, they converge to the same expansion plan and
  fight to a tie. Not a problem on the live ladder (opponent diverges)
  but does mean the local self-play P0/P1 sanity check shows a high
  draw rate, not a balanced win/loss split.

## Gotchas

- **The `+ 1.0` in the distance denominator is load-bearing for
  numerical stability**, not a tuned hyperparameter. Don't tune it
  without a clear reason.
- **`target.production` for comets is always 1** per the comp spec.
  ROI evaluates them as low-priority correctly, but it does not
  factor in the comet's remaining lifetime — a long-lived comet
  could outscore a static planet of the same production field.
  This is a v3-class concern, not a v2 concern.
- **Self-play draws are normal.** A ROI-vs-ROI panel cell that
  shows a near-50/50 W/L split (rather than mostly draws) is more
  alarming than the high-draw split — it would suggest the strategy
  is no longer symmetric across player IDs.

## Evidence

8-seed smoke (2026-05-10):
`audit/tournaments/20260510T123059Z.json`.

| vs            | aggregated winrate         |
| ------------- | -------------------------- |
| `nearest`     | 100% (16/16)               |
| `production`  | 81% (13/16)                |
| `weakest`     | 100% (16/16)               |
| `enemy_first` | 100% (16/16)               |
| `baseline`    | 100% (16/16)               |
| `v1_orbitfix` | **100% (16/16)**           |
| self-play     | 1 P0 / 0 P1 / 7 draws      |

Mean panel winrate: **96.9%**. p95 turn ~0.4 ms.

A 16/16 result has Wilson 95% CI [0.79, 1.00] — already
distinguishable from the 60% submission gate but well within
"could be a 12/16 result on a different seed bag." The 32-seed
follow-up will tighten this to roughly Wilson [0.85, 1.00] if it
stays at 100%, or surface the true rate if seed-42-vs-friends
happened to flatter ROI specifically.

## What it does NOT do

- **No combat budgeting.** Garrison cost not considered when scoring.
- **No comet lifetime correction.** Treats short-lived comets and
  long-lived planets identically for the production field.
- **No same-turn ledger.** If two of our owned planets share the
  same top-ROI target, both launch independently and waste the
  surplus.
- **No defence.** Garrison stays put; an incoming enemy fleet still
  walks into our home undefended.
- **No mission classification.** ROI is one global score function,
  not a portfolio of per-mission scoring rules.

These are the same gaps as every other simple-panel strategy. The
panel is a target-selection ablation; structural mechanisms land at
v2 (`docs/strategies/roadmap.md`).
