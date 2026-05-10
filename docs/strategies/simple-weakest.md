# simple/weakest — snipe-the-soft-target

> File: `agents/simple/weakest.py`.
> Role: weakness-greedy variant in the simple-strategy panel.
> Status: local-only. **Falsified at 8 seeds** (15.6% mean panel
> winrate). Kept in **Open** until 32-seed confirmation, but
> already demoted to opponent-panel diversity rather than
> submission candidate.

## One-liner

For each owned planet, pick the **target with the smallest garrison
right now**. Cheap captures hypothetically snowball: a 1-ship rock
costs 2 ships to flip and starts producing immediately, so capturing
many cheap rocks fast should out-expand a strategy that picks fights
with bigger garrisons.

## Mechanism

Per turn:

1. Build planet list and split into `my_planets` / `targets`.
2. Per-turn RNG seeded by `step ^ (player + 1) * 1009`.
3. For each owned planet, score every target by `(target.ships,
   distance)` — sort ascending. Smallest garrison wins; among
   garrison ties, nearest wins; among full ties, RNG breaks.
4. Emit `Intent(src_id, target_id, ships=target.ships + 1)` and
   pass through the standard mechanism stack.

## Why it works (or doesn't)

The hypothesis was: starting ship counts on neutrals are skewed low
(comp spec: range [5, 99] but heavy left-skew), so many planets are
flippable for under 20 ships. A strategy that exclusively pursues
these cheap captures should rack up territory faster than a
strategy that wades into garrisoned planets.

The hypothesis is **wrong** — and the panel result makes the
mechanism legible:

- **Captures without production are dead capital.** A 1-ship neutral
  with production 1 yields 1 ship/turn. A 99-ship neutral with
  production 5 yields 5 ships/turn — the latter pays back the 100-
  ship investment in 20 turns and then keeps producing for the
  remaining 480. Weakest-greedy captures the 1-ship rock and ignores
  the production-5 prize, so its long-run yield is anaemic.
- **The opponent owns the production.** While weakest-greedy is
  picking off 1-ship rocks across the map, every panel agent that
  prefers production (or ROI) is camping on the production-5 planets
  and out-producing weakest 5x per turn. By step 200 the opponent
  has more ships than weakest can possibly muster.
- **Distance is only a tiebreaker.** A 1-ship rock 80 units away is
  preferred over a 10-ship neutral 8 units away. The fleet spends
  20-30 turns in transit for a yield that will take 20+ more turns
  to repay. Net negative on most maps.

The 8-seed panel shows weakest losing to every non-baseline agent:
0% vs `nearest`, 0% vs `production`, 0% vs `roi`, 12% vs
`enemy_first`, 0% vs `v1_orbitfix`. The *only* agent it beats
consistently is the shipped baseline (81% / 13-of-16) — and even
then, it's still weaker than every other panel member's beat-the-
baseline rate.

## Gotchas

- **The hypothesis was reasonable; the mechanism was wrong.** Cheap
  captures *do* compound — but the production rate of the captured
  planet is the load-bearing variable, not the ship cost. `simple-
  roi.md` is the right way to value cheap captures: ROI of a 1-ship
  production-1 rock is `1 / (d + 1)` ≈ 0.05 at d=20; ROI of a 99-
  ship production-5 planet at d=20 is `5 / 21` ≈ 0.24, *five times
  better* even after distance.
- **The `+1` ships sizing is just enough to flip empty rocks.** The
  pre-mechanism intent is `target.ships + 1`; for a 1-ship neutral
  that's 2 ships out of garrison. Useful for the panel's diversity,
  not viable as a primary strategy.
- **Self-play P0/P1 split is 5/3/0.** Both sides race for the same
  cheap rocks; the seat with the small map-asymmetry advantage wins
  the race more often. The 0 draws are diagnostic — weakest never
  reaches the step-limit because the loser runs out of planets first.

## Evidence

8-seed smoke (2026-05-10):
`audit/tournaments/20260510T123059Z.json`.

| vs            | aggregated winrate |
| ------------- | ------------------ |
| `nearest`     | 0% (0/16)          |
| `production`  | 0% (0/16)          |
| `roi`         | 0% (0/16)          |
| `enemy_first` | 12% (2/16)         |
| `baseline`    | 81% (13/16)        |
| `v1_orbitfix` | 0% (0/16)          |
| self-play     | 5 P0 / 3 P1 / 0 draws |

Mean panel winrate: **15.6%**. p95 turn ~0.3 ms.

The 0% rows are striking: weakest does not steal *any* games from
`nearest`, `production`, `roi`, or `v1_orbitfix` across 16 games
each. With Wilson 95% CI on 0/16 of [0.00, 0.21] this is
statistically distinguishable from "small advantage" — the
strategy is genuinely worse than every panel member except the
shipped baseline.

## Why we keep it (after losing)

Despite the falsification, weakest stays in the local panel:

1. **Hold-out opponent diversity (D.4).** Future agents — heuristic
   v2, IL warm-start, RL self-play — should not be over-fit to a
   single dominant opponent. A panel that includes a *consistently
   weaker* but *strategically distinct* agent surfaces brittleness
   that "v1.1 vs shipped baseline" alone wouldn't catch.
2. **Falsification record.** The hypothesis "small garrisons compound
   first" is intuitive enough that we're likely to re-derive it.
   Keeping the falsified module in-tree, with this writeup attached,
   means the next agent can read the rationale instead of re-running
   the experiment.

## What it does NOT do

- **No production weighting.** Cheap captures are worthless without
  yield; this is the load-bearing oversight.
- **No distance weighting beyond tiebreak.** A small garrison on the
  far side of the board outranks a slightly larger garrison next door.
- **All structural gaps from the simple-strategy panel** (no arrival
  ledger, no defence, no fleet coordination, no comet lead, no sun-
  avoidance). Inherited from the panel shape; resolved at v2.
