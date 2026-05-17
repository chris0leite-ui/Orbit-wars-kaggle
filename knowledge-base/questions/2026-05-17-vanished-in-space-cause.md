# QUESTION — what causes `vanished_in_space` in live replays?

Date: 2026-05-17
Filed-by: claude/audit-workflow-performance-btjeK
Source data: `audit/replays/replay-mine-2026-05-17.{json,md}`

## The number

92 v15 live replays → **838 fleets vanish mid-flight** without
hitting any planet. That's 8.8% of all our fleets, but **18% of
our ships** — we send larger fleets on the vanishing trajectories.
This is ~6x larger than sun-deaths (13 fleets) and the dominant
single-category waste.

Currently classified `vanished_in_space` by
`scripts/episode_postmortem.attribute_fleets`:

> Disappeared after launch; not co-located with any planet at vanish_t,
> sun_d ≥ 10.5, in_bounds is True. Bucket of last resort.

## Suspected causes (need replay-level verification)

1. **Comet collision.** Comets travel on elliptical paths and destroy
   fleets in their orbit. The simulator (`data/main.py` /
   `kaggle_environments.envs.orbit_wars.orbit_wars`) checks fleet–
   comet proximity each tick. If a comet is on our fleet's path,
   our fleet vanishes. `lib/missions/snipe.py` accounts for comets,
   but the baseline proposer's cheap-rank may not project comet paths.
2. **Lead-aim error for orbiting target.** If the proposer aims at
   the *current* position of an orbiting planet but the fleet's eta
   is long, the planet has rotated away by arrival. Fleet flies
   past empty space. The `lib.mech.lead_aim` mechanism exists; if
   the baseline proposer's `wait_N>0` candidates skip it for some
   src/tgt pairs, this is the cause.
3. **OOB just barely outside `in_bounds`**. Our threshold is `0.0
   ≤ x ≤ 100.0` strict — a fleet at (100.0001, 50.0) on tick N
   counts as in_bounds but the simulator may have killed it. Edge
   classification issue, not a real strategy bug.

## How to answer

For each `vanished_in_space` fleet:

1. Walk its trajectory step-by-step from `replay["steps"]`.
2. At each step, compute the closest comet's distance (if comets
   exist on the replay).
3. If `comet_dist < some_threshold` near the vanish_t, classify as
   `vanish_by_comet`.
4. Otherwise check the would-have-arrived planet's actual position at
   the original aim's eta vs the planet's *predicted* position at
   eta. Large delta → lead-aim failure.
5. Otherwise → genuine OOB / unknown.

The output of this would replace the single `vanished_in_space`
bucket with three sub-buckets, giving the PI a concrete next-fix
target.

## Why this question matters for the plan

The diagnostic plan ranked sun-death (pivot #5) but
`vanished_in_space` is 6x larger. If the cause is **comet
collision**, the fix is in the proposer / mission scoring
(skip trajectories that intersect a comet path in the next eta
turns). If the cause is **lead-aim failure**, the fix is in
`lib.mech.lead_aim`'s applicability scoring. Both are
modeling-correctness fixes per Rule 40, not constant bumps.

## Owner

PI to decide whether to claim. Estimated work: ~4-6h to extend
`attribute_fleets` with comet/lead-aim sub-classification, then
re-run on the 92-replay corpus. Output: split the 838-fleet
bucket into named sub-causes.
