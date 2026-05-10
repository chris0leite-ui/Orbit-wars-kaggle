# simple/nearest — distance-greedy control

> File: `agents/simple/nearest.py`.
> Role: control strategy in the simple-strategy panel.
> Status: local-only. Not submitted to the live ladder. Tracked under
> `state/hypothesis-board.md::Open` and `state/mechanism-ledger.md`
> (`simple-greedy-target-selection-variants`).

## One-liner

For each owned planet, pick the **closest non-owned planet** as the
target and send `target.ships + 1` ships at it. Identical targeting
rule to the comp-shipped baseline, but riding on top of v1.1's
mechanism stack so it inherits orbit-aware aim, production-aware
sizing, and ownership/garrison validation for free.

## Mechanism

Per turn:

1. Build the planet list from `obs["planets"]`. Split into
   `my_planets` (`owner == player`) and `targets` (everything else,
   including neutrals and enemies).
2. If `my_planets` is empty or `targets` is empty, return no actions.
3. Construct a per-turn RNG seeded by `step ^ (player + 1) * 1009`
   (the same A.6 anti-mirror seed v1 uses — see `v1_orbitfix.md`).
4. For each owned planet, score every target by 2D distance
   (`lib.geometry.dist`). Tie-breaker is the per-turn RNG draw, so
   two equidistant neutrals don't always go to the same one.
5. Emit one `Intent(src_id, target_id, ships=target.ships + 1)` per
   owned planet.
6. Hand the list of intents to `realize(intents, obs,
   mechanisms=DEFAULT_MECHANISMS)`. The mechanism stack
   (`[validate, arrival_size, lead_aim]`) drops invalid intents,
   bumps fleet size for production growth on enemy targets, and
   populates the orbit-aware aim angle.

The strategy is **identical** to `agents/v1_orbitfix/main.py` —
this module exists as the explicit control: same score function,
same RNG seed, same mechanism stack. Any divergence between
`nearest` and `v1_orbitfix` in the panel reflects RNG-stream noise,
not a strategy difference.

## Why it works (or doesn't)

The argument *for* nearest-first targeting is: short fleets arrive
intact (less time exposed to sun-clip and orbit drift), and the
mechanism stack's orbit lead handles the small drift on the way.

The arguments *against*:

- **Distance ignores production.** A nearby 1-production rock pays
  back forever at 1 ship/turn; a moderately-far 5-production planet
  pays back 5x faster. Once captured, production differences compound;
  capture order matters early.
- **Distance ignores garrison strength.** A neighbouring 99-ship
  enemy planet costs roughly 100 ships to flip; a slightly-farther
  10-ship enemy planet costs 11 ships. Distance-greedy spends
  resources on the wrong fights.

The 8-seed panel (audit/tournaments/20260510T123059Z.json) shows
`nearest` at 56.2% mean panel winrate — beats the shipped baseline
and the lossy strategies (`enemy_first`, `weakest`), but loses to
`production` (31% / 69%) and `roi` (0% / 100%). It's a competent
control, not a frontier strategy.

## Gotchas

- **Tied with v1_orbitfix by construction.** A panel run that shows
  `nearest` decisively beating or losing to `v1_orbitfix` indicates
  RNG drift or hidden state in one of them; investigate before
  trusting the panel.
- **Same mechanism stack as v1.1.** This module does not avoid the
  sun, does not forecast arrivals, does not coordinate fleets. Those
  gaps are inherited and motivate v2.

## Evidence

8-seed smoke (2026-05-10):
`audit/tournaments/20260510T123059Z.json`.

| vs            | nearest as P0 | nearest as P1 | aggregated |
| ------------- | ------------- | ------------- | ---------- |
| `production`  | 5/8 = 63%     | 3/8 (mirror)  | 31% (5/16) |
| `roi`         | 0/8 = 0%      | 0/8 (mirror)  | 0% (0/16)  |
| `weakest`     | 8/8 = 100%    | 8/8 (mirror)  | 100% (16/16) |
| `enemy_first` | 7/8 = 88%     | 7/8 (mirror)  | 88% (14/16) |
| `baseline`    | 8/8 = 100%    | 8/8 (mirror)  | 100% (16/16) |
| `v1_orbitfix` | 1/8 = 13%     | 2/8 (mirror)  | 19% (3/16)  |
| self-play     | 2/1/5 (P0/P1/draw) | — | — |

Self-play split is balanced (the A.6 fix carries through), p95 turn
~0.4 ms (well under the 1-s budget).

## What it does NOT do

This is the control strategy. It deliberately does **nothing** beyond
distance-based targeting — that's the point: it isolates the targeting
axis. Every gap of `agents/v1_orbitfix/main.py` (no arrival ledger,
no defence, no fleet coordination, no comet-path lead, no
sun-avoidance, no mission classification) is also a gap of `nearest`.

For the structural fixes, see `roadmap.md` (v2 arrival ledger, v3
missions). Within the simple-strategy panel, the lift comes from
swapping the targeting axis: see `simple-roi.md` and
`simple-production.md`.
