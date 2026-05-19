"""greedy_expand — MVP foundation test.

PI directive: strip every architectural layer (predicate / portfolio /
sequencer / defense) and verify whether the physics infrastructure
alone (`World.from_obs` + `_solve_single_source` + physics validation)
beats anything. If this beats `nearest`, the foundation is solid and
the chooser stack above has been subtracting value. If this loses,
something in the primitives is broken.

The agent:
  - For every (mine_source, not_mine_target) pair, compute closed-form
    capture sizing via `_solve_single_source`.
  - Drop candidates that don't physically reach the target (sun, OOB,
    intervening planet) via `launch_reaches_target`.
  - Sort by ROI = target.production / (ships + arrival_turn + 1).
  - Emit ONE launch per source per turn, highest ROI first.

No defense, no waiting, no multi-source bundles, no schedule. The
absence of those layers is the point.
"""

from __future__ import annotations

from agents.trajectory_roi.main import (
    _build_centrality_cache, _solve_single_source,
)
from lib.goal_planner.validate import launch_reaches_target
from lib.trajectory_layer import World


def agent(obs, configuration=None):
    world = World.from_obs(obs, configuration)
    my_id = world.my_id
    other_owners = [p.owner for p in world.planets
                    if p.owner not in (-1, my_id)]
    if not other_owners:
        return []
    opp_id = max(set(other_owners), key=other_owners.count)
    if my_id == opp_id:
        return []

    centrality_cache = _build_centrality_cache(world)
    my_planets = [p for p in world.planets if p.owner == my_id]
    targets = [p for p in world.planets if p.owner != my_id]

    candidates: list[tuple[float, int, int, int, float]] = []
    for src in my_planets:
        for tgt in targets:
            cand = _solve_single_source(
                src, tgt, world, my_id, centrality_cache,
                target_is_ours=False,
            )
            if cand is None:
                continue
            a = cand.allocations[0]
            if not launch_reaches_target(src, tgt, a.aim_angle,
                                           a.ships, world):
                continue
            roi = float(tgt.production) / float(
                cand.total_ships + cand.arrival_turn + 1
            )
            candidates.append((roi, src.id, tgt.id, a.ships, a.aim_angle))

    candidates.sort(reverse=True)

    used_sources: set[int] = set()
    emits: list[list] = []
    for _roi, src_id, _tgt_id, ships, angle in candidates:
        if src_id in used_sources:
            continue
        emits.append([src_id, angle, ships])
        used_sources.add(src_id)
    return emits
