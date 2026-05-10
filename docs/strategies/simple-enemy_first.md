# simple/enemy_first — pressure-on-opponent targeting

> File: `agents/simple/enemy_first.py`.
> Role: enemy-prioritising variant in the simple-strategy panel.
> Status: local-only. **Falsified at 8 seeds** (32.3% mean panel
> winrate). Kept in **Open** until 32-seed confirmation, then
> demoted to opponent-panel diversity per the same rationale as
> `simple-weakest`.

## One-liner

For each owned planet, prefer **enemy-owned** targets over neutrals.
Within each ownership group, pick the nearest target. Falls back to
plain nearest-greedy until contact (i.e. when no enemy planet exists
in the target list).

## Mechanism

Per turn:

1. Build planet list, split into `my_planets` / `targets` as elsewhere.
   (`targets` includes everything not owned by `player` — both
   neutrals with `owner == -1` and enemies with `owner ∈ {0..3} \ {player}`.)
2. Per-turn RNG seeded by `step ^ (player + 1) * 1009`.
3. For each owned planet, score every target by the tuple
   `(is_neutral, distance)`:
   - `is_neutral = 0` for enemy-owned planets, `1` for neutrals.
   - Sort ascending: enemies sort first, neutrals second; within each
     group, nearest wins.
4. Emit `Intent(src_id, target_id, ships=target.ships + 1)` and run
   through the standard mechanism stack.

Until contact (no enemy planets exist), `enemy_first` collapses to
plain `nearest`-greedy. So early-game it behaves identically to the
control; the divergence only kicks in once both players hold ground.

## Why it works (or doesn't)

The hypothesis was: every turn we leave the enemy's planet alone,
their production compounds against us. Capturing one of theirs is a
**two-sided swing** — we gain the production, they lose it — so a
single enemy capture is worth roughly 2x a neutral capture in
relative-production terms. A strategy that prefers enemies should
out-pressure one that ignores them.

The hypothesis is **wrong** in our setup:

- **Enemy planets are usually well-defended.** An enemy planet has
  been accumulating ships every turn since the player captured it.
  By step 50, an enemy production-3 planet starts with maybe 10 home
  ships and gets 150 production ticks → ~160 ships before our fleet
  can plausibly reach it. The +1 sizing rule (`target.ships + 1`)
  combined with `arrival_size`'s production-aware bump is correct,
  but the absolute fleet cost dwarfs anything we'd spend on a
  neutral.
- **Neutrals are still expanding.** Picking only enemy targets means
  every cheap unowned production-5 planet on the map sits there
  unclaimed. The opponent collects those neutrals first because they
  *don't* have an enemy-first preference — they pick by distance/
  production/ROI and grab the closest payoff. Within 100 turns the
  opponent is ahead on production and ahead on garrisons.
- **Two-sided swing only matters when fleets are comparable.**
  Capturing an enemy production-5 planet is worth +5/-5 = 10
  production/turn relative to neutral. But if it cost 200 ships
  versus 20 ships for a neutral production-3, the opportunity cost
  per ship invested is much higher on the enemy capture.

The 8-seed panel confirms: `enemy_first` is a midcard strategy that
beats `weakest` (88% / 14-of-16) and the shipped baseline (75% /
12-of-16) but loses to `nearest`, `production`, `roi`, and
`v1_orbitfix`. Self-play converges to 8 draws (every game ends in a
tie) — when both sides leave neutrals untouched, the games stalemate
hard.

## Gotchas

- **The 8-of-8 self-play draws are diagnostic.** When two
  enemy-first agents meet, neither expands; both starve the other
  of progress. Result: pure draws. This is the single clearest signal
  in the whole panel that "ignore neutrals" is structurally bad.
- **`enemy_first` *can* still be useful as a strategic component.**
  At v2, the hypothesis "always pressure enemy" can be generalised
  into a *mission* — `recapture` (an arrival-ledger-aware version of
  enemy targeting) and `gang_up` (multi-source on a single enemy
  flip) are both Roman-1224 mission classes. The enemy_first strategy
  here is the "what if enemy-pressure was the *only* policy" baseline
  for those missions to beat at v3.
- **Until-contact behaviour is identical to `nearest`.** A panel run
  that shows `enemy_first` decisively diverging from `nearest` in the
  first 20-30 steps before contact would indicate a bug, not a
  strategy effect.

## Evidence

8-seed smoke (2026-05-10):
`audit/tournaments/20260510T123059Z.json`.

| vs            | aggregated winrate |
| ------------- | ------------------ |
| `nearest`     | 12% (2/16)         |
| `production`  | 6% (1/16)          |
| `roi`         | 0% (0/16)          |
| `weakest`     | 88% (14/16)        |
| `baseline`    | 75% (12/16)        |
| `v1_orbitfix` | 12% (2/16)         |
| self-play     | 0 P0 / 0 P1 / 8 draws |

Mean panel winrate: **32.3%**. p95 turn ~0.3 ms.

The 0% vs `roi` is an outlier in the same direction as `weakest`'s
0%-rows — `roi`'s production / distance signal beats both
"weakest-first" and "enemy-first" decisively because it sees yield
where they see only one axis.

## Why we keep it (after losing)

Same rationale as `simple-weakest`:

1. **Hold-out opponent diversity for future agents.** v2 and beyond
   should not be tested only against agents that prioritise the same
   axes; an enemy-first opponent stresses the arrival-ledger
   forecaster differently than ROI does (it specifically forces the
   defence question).
2. **Falsification record.** "Always attack enemy first" is an
   intuitively appealing rule that keeps showing up in heuristic-
   strategy thinking. Keeping the module + this writeup means the
   next agent doesn't waste a slot re-deriving why it's wrong on
   its own.

## What it does NOT do

- **No production or ROI weighting.** Within ownership groups, only
  distance is used.
- **No defence trigger.** When attacked, garrison stays put.
- **No coordination across owned planets.** Two of our planets
  may both pick the same enemy target with `target.ships + 1`
  ships each, double-counting the cost.
- **All structural gaps from the simple-strategy panel** (no arrival
  ledger, no comet lead, no sun-avoidance, no mission classification).
  Inherited; resolved at v2.
