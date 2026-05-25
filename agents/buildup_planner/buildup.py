"""BUILDUP phase: deterministic multi-turn opening MILP, solved ONCE.

Wraps `lib.joint_solver.opening_planner.opening_plan` as a
commit-and-execute schedule: the planner runs at the FIRST BUILDUP
call of a game, its schedule is cached in the caller's per-seat state
dict, and subsequent turns emit only the cached entries whose
`fire_step == step_now`. Eliminates 29 of the 30 MILP solves per
opening that the previous re-derive-every-turn implementation paid for
(2026-05-25 transition-fix made the wallclock cost observable: p95
turn-time was 1101 ms — over the 1000 ms Kaggle budget).

Matches the original design intent stated in
`lib/joint_solver/opening_planner.py:4-7` ("a multi-turn
commit-and-execute schedule, computed once, then executed").
"""
from __future__ import annotations

from typing import Optional

from lib.joint_solver.opening_planner import OPENING_HORIZON, opening_plan


def step(world, model, me: int, num_seats: int,
         step_now: int, state: dict) -> Optional[list[list]]:
    """Return immediate launches from the cached opening schedule.

    `state` is the caller's `_PHASE_STATE[me]` dict (see
    `agents/buildup_planner/main.py:_initial_state`). It carries
    `opening_schedule`:
      - `None` before the first solve;
      - `[]` if the planner returned no candidates;
      - a `list[ScheduleEntry]` containing the un-consumed entries.

    Return semantics (unchanged from the pre-cache contract):
      - `None`  -> opening exhausted (step_now >= OPENING_HORIZON) OR
                   solver crash. main.py transitions to CONSOLIDATION
                   (or stays in BUILDUP if step_now < OPENING_HORIZON,
                   per the 2026-05-25 transition-logic fix at
                   main.py:186-204).
      - `[]`    -> no fire-this-turn entries; stay in BUILDUP, wait.
      - `[...]` -> emit these launches now.
    """
    if step_now >= OPENING_HORIZON:
        return None

    # Solve once per game, on the first BUILDUP call.
    if state.get("opening_schedule") is None:
        try:
            op = opening_plan(world, model, me, num_seats)
        except Exception:
            # Solver crash: transition to CONSOLIDATION via the caller's
            # None-handling path. Don't cache so a re-entry can retry.
            return None
        if op is None or not op.schedule:
            state["opening_schedule"] = []
        else:
            # Copy to a mutable list so we can drop consumed entries.
            state["opening_schedule"] = list(op.schedule)

    schedule = state["opening_schedule"]

    # Emit entries matching fire_step, with source-ownership guard.
    # Drop consumed and ownership-invalidated entries from the cache.
    emitted: list[list] = []
    remaining = []
    for entry in schedule:
        fire = int(entry.fire_step)
        if fire == step_now:
            # Ownership guard: opp may have captured our source between
            # solve-time and fire-time. Env silently drops launches from
            # non-owned planets; we make that drop explicit here.
            src = world.planets_by_id.get(int(entry.src_id))
            if src is None or int(src.owner) != int(me):
                continue  # entry dropped, do not re-add to schedule
            emitted.append([int(entry.src_id), float(entry.angle),
                            int(entry.ships)])
            # entry consumed; do not re-add.
        elif fire > step_now:
            remaining.append(entry)
        # fire < step_now: stale entry (missed its window); drop.

    state["opening_schedule"] = remaining
    return emitted
