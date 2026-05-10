# Shipped baseline — Nearest Planet Sniper

> File: `data/main.py` (comp-shipped, **do not modify**).
> Submitted as ID 52497828 on 2026-05-10 as a calibration probe → live μ = **303.2** (started μ₀ = 600).

## One-liner

Send exactly `target.ships + 1` ships from each owned planet to its
closest non-owned planet, aimed at where the target currently is.

## Mechanism

- **Perception:** read `obs["planets"]`. Split into `mine` (owner == player)
  and `targets` (everything else, including neutrals and enemies).
- **Planning (per owned planet):** find the nearest target by 2D Euclidean
  distance. Compute `ships_needed = target.ships + 1`. If our garrison can
  cover it, queue a launch.
- **Action:** angle = `atan2(target.y - mine.y, target.x - mine.x)` —
  aimed at the **current** position of the target.

## Why it works against `random`

Random fleets often hit the sun, leave the board, or arrive at low ship
counts. Nearest-planet greedy reliably captures neutrals before they do.
Result: 6/6 vs random in the day-1 audit, 8/8 vs random in the smoke
tournament across both sides.

## Why it underperforms on the live ladder (μ = 303 < starting 600)

1. **Aims at the wrong point** — for orbiting planets (inner ring), the
   target moves while the fleet is in flight. The shipped agent fires
   at the *current* position, so its fleet arrives where the target *was*,
   not where it now *is*. Larger fleets travel slower (1 ship → 1 unit/turn,
   1000 ships → 6 units/turn), so even small lead errors compound. Public
   notebooks (Pilkwang structured-baseline, Roman 1224, Djenk ow-proto)
   all spend a chapter on arrival-time prediction for this reason.
2. **Deterministic tie-break P0/P1 asymmetry** — when two targets sit at
   equal distance, the shipped agent picks whichever was iterated first.
   Both players' fleets get routed to the same neutral; lower-id (P0)
   launches first → fleet arrives first → P0 captures, P1 swerves to
   the second-best target. Empirically (audit/2026-05-10-day-1-rollouts.json):
   baseline-vs-baseline P1 wins 4/6, P0 wins 1/6, ties 1/6.
3. **No combat-arrival forecasting** — sends ships toward a planet
   ignoring incoming enemy fleets that will alter ownership before
   our fleet lands.
4. **No sun-avoidance** — `atan2` aim plus continuous segment-vs-sun
   collision means any source-target pair on opposite sides of the sun
   loses the fleet. Public top agents detour around the sun.
5. **No comet handling** — comets follow `obs["comets"][].paths`, not
   the rotation formula; aiming at current comet position is even
   shorter-lived than aiming at current orbital-planet position.

## Evidence

- Day-1 rollouts: 6/6 vs random; baseline-vs-baseline 4/6 P1 wins.
- Tournament smoke (2026-05-10T06:47Z): 8/8 vs random across both sides.
- Live ladder (Submission 52497828): μ = 303.2.

## What it does NOT do

Lead the target's motion; randomise tie-breaks; check sun-clearance;
forecast arrival-time ownership; coordinate fleets across owned planets;
defend home; track enemy fleet trajectories. **All of these are open
work in the next version family — see `roadmap.md`.**
