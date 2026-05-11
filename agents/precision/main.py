"""Precision agent entry point.

Usage: from this module import `agent` and submit it (or pack as main.py).
"""
from __future__ import annotations

import time

from agents.precision import intercept, planner


def agent(obs):
    """Return a list of [src_id, angle, ships] launches."""
    t0 = time.perf_counter()
    try:
        world = intercept.parse_world(obs)
    except Exception:
        return []

    # Skip first turn (obs.step=0 has the engine's rotation-free edge case).
    if world["step"] == 0:
        return []

    # Budget: ~0.85s actTimeout, leaving headroom for action serialization.
    deadline = t0 + 0.85
    try:
        plan = planner.plan_turn(world, deadline=deadline)
    except Exception:
        plan = []
    return planner.emit_actions(plan)
