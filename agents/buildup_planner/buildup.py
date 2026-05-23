"""BUILDUP phase: deterministic multi-turn opening MILP.

Wraps `lib.joint_solver.opening_planner.opening_plan` unchanged. Emits
this turn's `fire_step == step_now` entries verbatim. Returns None when
no entries fire (callers route to CONSOLIDATION).
"""
from __future__ import annotations

from typing import Optional

from lib.joint_solver.opening_planner import OPENING_HORIZON, opening_plan


def step(world, model, me: int, num_seats: int,
         step_now: int) -> Optional[list[list]]:
    """Return immediate launches from the opening schedule, or None.

    None signals "fall through to CONSOLIDATION." Empty list `[]` would
    suppress all moves this turn — never desired in the opening.
    """
    if step_now >= OPENING_HORIZON:
        return None
    try:
        op = opening_plan(world, model, me, num_seats)
    except Exception:
        return None
    if op is None or not op.schedule:
        return None
    moves = [
        [int(e.src_id), float(e.angle), int(e.ships)]
        for e in op.schedule if int(e.fire_step) == int(step_now)
    ]
    return moves if moves else None
