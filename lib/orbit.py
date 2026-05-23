"""Orbit-prediction primitives for Orbit Wars.

Two prediction modes — both verified against `env.steps[N]` on seed 42 to
0.0 error in `scripts/orbit_prediction_check.py` (audit/2026-05-10-day-1-data-inventory.md):

- `predict_absolute(initial_planet, omega, env_step_n)` — uses the
  empirically-correct N-1 offset. Naive `omega*N` is OFF BY ONE step.
- `predict_relative(current_planet, omega, lead_turns)` — projects forward
  from current observation; no step counter needed. **Preferred for agents.**

Static planets (orbital_radius + planet_radius >= ROTATION_RADIUS_LIMIT) do
not rotate; `is_orbiting` flags them.
"""

from __future__ import annotations

import math

from .geometry import CENTER, ROTATION_RADIUS_LIMIT, Point


def is_orbiting(planet) -> bool:
    """planet = [id, owner, x, y, radius, ships, production]."""
    px, py, pr = planet[2], planet[3], planet[4]
    orb_r = math.hypot(px - CENTER, py - CENTER)
    return (orb_r + pr) < ROTATION_RADIUS_LIMIT


def predict_relative(current_planet, angular_velocity: float, lead_turns: float) -> Point:
    """Predict (x, y) `lead_turns` after the obs that yielded `current_planet`.

    Safe for an agent that doesn't track absolute step count: read polar
    angle of the current planet position and rotate forward by
    `omega * lead_turns`. STATIC planets (those failing `is_orbiting`)
    return their raw position — the env does NOT rotate planets outside
    ROTATION_RADIUS_LIMIT, and the pre-2026-05-23 always-rotate behavior
    was a silent bug used by 4 non-pre-filtered call sites in
    `lib/aim.py` and `agents/baseline/main.py`. The bug surfaced as 81%
    turn divergence between KT-ON and KT-OFF (the kinematic table
    correctly stores constant for static planets, while this function
    used to rotate them); see /tmp/parity_kt.py.

    Returns the current (x, y) when omega == 0 too — the math
    `cur_angle + 0 * lead = cur_angle` round-trips through atan2/cos/sin
    introducing ULP drift, so we short-circuit.
    """
    if not is_orbiting(current_planet):
        return (float(current_planet[2]), float(current_planet[3]))
    px, py = current_planet[2], current_planet[3]
    dx, dy = px - CENTER, py - CENTER
    orb_r = math.hypot(dx, dy)
    cur_angle = math.atan2(dy, dx)
    new_angle = cur_angle + angular_velocity * lead_turns
    return (
        CENTER + orb_r * math.cos(new_angle),
        CENTER + orb_r * math.sin(new_angle),
    )


def predict_relative_cached(current_planet, angular_velocity: float,
                            lead_turns: float, *, table=None) -> Point:
    """Lookup-aware wrapper around `predict_relative`.

    When `table` is provided and the planet is in the table's current
    obs snapshot, returns the cached lookup (O(1), no trig). On any
    miss — `table is None`, planet pid not in table, lookup past
    `max_lead` — falls through to the slow-path `predict_relative` call.

    Bit-parity guarantee: the cached path is bit-identical to the
    fallback IFF `planet` is the same instance from
    `world.planets_by_id` that `table.begin_turn(world)` saw. Synthetic
    or predicted planet states (e.g. inside an aim-orbiting fixed-point
    loop where the "planet" position is a hypothetical future tick)
    MUST pass `table=None` to force the slow path.
    """
    if table is None:
        return predict_relative(current_planet, angular_velocity, lead_turns)
    pid_obj = None
    try:
        pid_obj = getattr(current_planet, "id", None)
        if pid_obj is None:
            pid_obj = current_planet[0]
        pid = int(pid_obj)
    except (TypeError, IndexError, KeyError):
        return predict_relative(current_planet, angular_velocity, lead_turns)
    if not table.has(pid):
        return predict_relative(current_planet, angular_velocity, lead_turns)
    try:
        return table.lookup_relative(pid, lead_turns)
    except (IndexError, KeyError):
        return predict_relative(current_planet, angular_velocity, lead_turns)


def predict_absolute(initial_planet, angular_velocity: float, env_step_n: int) -> Point:
    """Predict (x, y) at `env.steps[env_step_n]` from `initial_planets`.

    Uses the empirically-correct N-1 rotation count for N>=1 (see
    `audit/2026-05-10-day-1-data-inventory.md::A.1`). The naive `omega*N`
    form is off by exactly one step's rotation (~1.27 board units on
    inner planets at the default angular velocity).
    """
    px, py = initial_planet[2], initial_planet[3]
    dx, dy = px - CENTER, py - CENTER
    orb_r = math.hypot(dx, dy)
    init_angle = math.atan2(dy, dx)
    n_rot = max(env_step_n - 1, 0)
    cur_angle = init_angle + angular_velocity * n_rot
    return (
        CENTER + orb_r * math.cos(cur_angle),
        CENTER + orb_r * math.sin(cur_angle),
    )


def predict_relative_smart(current_planet, angular_velocity: float,
                           lead_turns: float) -> Point:
    """Env-gated cached wrapper around `predict_relative`.

    When `KINEMATIC_TABLE_ENABLED=1`, routes through the singleton
    kinematic_table via `predict_relative_cached` (which itself falls
    through to `predict_relative` on miss — pid not primed, lead past
    `max_lead`, etc.). Otherwise identical to `predict_relative`.

    Bit-parity: when the table is primed via `begin_turn(world)` AND
    `current_planet` is `world.planets_by_id[pid]` for some pid in the
    table, the cached path is bit-identical to `predict_relative` for
    any lead in `[0, table.max_lead]`. Synthetic / hypothetical planet
    states (constructed mid-fixed-point loops) bypass the cache because
    their pid is not in the table.

    Mirrors the Phase γ gating pattern in `lib/trajectory.py` so the
    whole kinematic-table feature has ONE global knob.
    """
    import os
    if os.environ.get("KINEMATIC_TABLE_ENABLED", "").strip().lower() not in (
        "1", "true", "on", "yes",
    ):
        return predict_relative(current_planet, angular_velocity, lead_turns)
    from lib.kinematic_table import get_default
    return predict_relative_cached(
        current_planet, angular_velocity, lead_turns, table=get_default(),
    )
