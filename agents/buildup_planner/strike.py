"""STRIKE phase — atomic-drop emission of a coordinated wave.

`step(world, plan)` re-validates every `Shot` in `plan.shots` against the
current world via `lib.trajectory.predict_fleet_fate`. ALL shots in a
wave fire on the same turn (the predicate searches for a single arrival
step T and produces shots with varying `eta` so they converge there).
There is no multi-turn persistence — the strike turn is one decisive
emission, and control returns to CONSOLIDATION next turn.

Two failure modes both trigger an atomic drop (return `[]`, log warning):

1. **Budget overflow** — Step-2 `evaluate_inflection` permits the same
   source planet to be counted toward multiple targets (the optimistic
   upper bound, see `predicates.py` module docstring). At emit time we
   enforce the actual constraint: per-source cumulative `ship_count`
   must not exceed the source's current garrison.

2. **Physics fail** — `predict_fleet_fate` returns any `outcome` other
   than `"target"`. Mirrors the joint-candidate atomic-drop in
   `agents/baseline/chooser_trajectory.py:612-628`.

The drop cost is one wasted predicate elect, not a wasted turn — the
dispatcher transitions back to CONSOLIDATION and emits normal moves
next turn.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from lib.trajectory import predict_fleet_fate

from agents.buildup_planner.predicates import StrikePlan


_LOG = logging.getLogger("buildup_planner.strike")


def step(world, plan: StrikePlan) -> list[list]:
    """Emit the wave's moves, or atomic-drop on any failure."""
    if plan is None or not plan.shots:
        return []

    # Pass 1: per-source ship-budget check. Step-2 predicate may have
    # double-counted a source across multi-target plans (|S|=2 case).
    usage: dict[int, int] = defaultdict(int)
    for shot in plan.shots:
        usage[int(shot.src_id)] += int(shot.ship_count)
    for src_id, used in usage.items():
        src = world.planets_by_id.get(int(src_id))
        if src is None:
            _LOG.warning(
                "atomic-drop: src %d not in world (target_ids=%s arrival=%d)",
                src_id, sorted(plan.target_ids), plan.arrival_step,
            )
            return []
        if used > int(src.ships):
            _LOG.warning(
                "atomic-drop: budget_overflow src=%d used=%d garrison=%d "
                "(target_ids=%s arrival=%d)",
                src_id, used, int(src.ships),
                sorted(plan.target_ids), plan.arrival_step,
            )
            return []

    # Pass 2: per-shot physics re-validation via the engine-mirroring
    # ray-cast. ANY non-"target" outcome drops the whole wave.
    for shot in plan.shots:
        src = world.planets_by_id.get(int(shot.src_id))
        tgt = world.planets_by_id.get(int(shot.tgt_id))
        if src is None or tgt is None:
            _LOG.warning(
                "atomic-drop: missing src=%s or tgt=%s",
                shot.src_id, shot.tgt_id,
            )
            return []
        fate = predict_fleet_fate(
            src, tgt, float(shot.angle), int(shot.ship_count), world,
        )
        if fate.outcome != "target":
            _LOG.warning(
                "atomic-drop: physics_fail src=%d tgt=%d outcome=%s hit=%s "
                "step=%d (target_ids=%s arrival=%d)",
                shot.src_id, shot.tgt_id, fate.outcome, fate.hit_planet_id,
                fate.step, sorted(plan.target_ids), plan.arrival_step,
            )
            return []

    # All-pass: emit the wave.
    return [[int(shot.src_id), float(shot.angle), int(shot.ship_count)]
            for shot in plan.shots]
