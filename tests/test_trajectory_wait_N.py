"""Regression tests for `predict_fleet_fate(wait_N=...)` orbital drift.

When wait_N > 0, the fleet is scheduled to launch wait_N ticks in the
future. Source position, target position, and the spawn point should
all be advanced by wait_N orbital ticks before the ray-cast begins.
This is the contract that commit aac3c1e introduced (the "ships do
not hit targets" fix on our side).

These tests pin the wait_N path with a scenario where the outcome
QUALITATIVELY differs between wait_N=0 and wait_N=N — proving the
fleet actually launches from the advanced geometry, not the current
snapshot.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from lib.intent import World
from lib.trajectory import predict_fleet_fate


def _world(my_id, planets, omega=0.0):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": [],
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "step": 0,
    }
    return World.from_obs(obs)


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    return SimpleNamespace(
        id=pid, owner=owner, x=x, y=y, radius=radius,
        ships=ships, production=production,
    )


# --- Orbiting source, half-rotation flips the outcome ----------------------


def test_wait_N_advances_orbiting_source_position():
    """Orbiting source at (70, 50). With omega = pi/10, wait_N=10
    advances the source half an orbit to (30, 50).

    Aiming east:
    - wait_N=0: spawn just east of (70, 50); path is x in [70, ...], y=50.
      The sun at (50, 50, r=10) is BEHIND the spawn. Fleet flies east,
      hits the static target near the east edge → outcome="target".
    - wait_N=10: source has orbited to (30, 50). Spawn at (~31.6, 50)
      east-bound; the path now CROSSES the sun at x≈40-50. Outcome="sun".

    This proves the source position advances under wait_N>0.
    """
    src = _planet(0, 0, 70.0, 50.0, radius=1.5)
    target = _planet(1, -1, 97.0, 50.0, radius=3.0)
    omega = math.pi / 10.0
    world = _world(my_id=0, planets=[src, target], omega=omega)
    angle = 0.0  # straight east

    fate_now = predict_fleet_fate(src, target, angle, ships=10, world=world,
                                  wait_N=0)
    fate_late = predict_fleet_fate(src, target, angle, ships=10, world=world,
                                   wait_N=10)
    assert fate_now.outcome == "target", (
        f"wait_N=0 should hit target, got {fate_now}")
    assert fate_late.outcome == "sun", (
        f"wait_N=10 should die in sun (source half-orbited), got {fate_late}")


def test_wait_N_zero_matches_no_arg_default():
    """`wait_N=0` is the explicit default; semantics must be identical to
    omitting the argument. Sanity guard against silent param shifts.
    """
    src = _planet(0, 0, 10.0, 20.0, radius=2.0)
    target = _planet(1, 1, 80.0, 20.0, radius=2.0)
    world = _world(my_id=0, planets=[src, target], omega=0.05)
    angle = math.atan2(target.y - src.y, target.x - src.x)

    fate_default = predict_fleet_fate(src, target, angle, ships=10, world=world)
    fate_zero = predict_fleet_fate(src, target, angle, ships=10, world=world,
                                   wait_N=0)
    assert fate_default == fate_zero, (
        f"wait_N=0 should match default; got default={fate_default} vs "
        f"wait_N=0={fate_zero}")


def test_wait_N_no_op_on_static_source_and_target():
    """Static planets (orbital radius + planet radius ≥ ROTATION_RADIUS_LIMIT=50)
    don't rotate. With both source and target static, wait_N > 0 must give
    the SAME outcome as wait_N=0.

    Place both at the outer ring so they don't orbit.
    """
    src = _planet(0, 0, 50.0, 1.0, radius=3.0)      # orb_r=49, 49+3=52 → static
    target = _planet(1, 1, 50.0, 99.0, radius=3.0)  # same; static
    world = _world(my_id=0, planets=[src, target], omega=math.pi / 10.0)
    angle = math.atan2(target.y - src.y, target.x - src.x)

    fate_now = predict_fleet_fate(src, target, angle, ships=10, world=world,
                                  wait_N=0)
    fate_late = predict_fleet_fate(src, target, angle, ships=10, world=world,
                                   wait_N=10)
    assert fate_now == fate_late, (
        f"Static planets must be wait_N-invariant; got {fate_now} vs {fate_late}")


def test_wait_N_advances_orbiting_target_position():
    """Static source, orbiting target. The fleet flies a straight path
    aimed at the target's CURRENT position. If we wait wait_N ticks, the
    target moves out of that path.

    Setup: source at (50, 1) static, target orbiting at (70, 50). Aim at
    the target's t=0 position. With omega=pi/10, at wait_N=10 the target
    has rotated half-orbit to (30, 50) — well off the aim line.

    Expected:
    - wait_N=0: outcome="target" (aim is correct at t=0).
    - wait_N=10: target's t=10 position is NOT on the aim line → outcome
      will be something else (e.g., "planet" if the aim line happens to
      intersect another body, otherwise "oob" or "timeout").
    """
    src = _planet(0, 0, 50.0, 1.0, radius=3.0)      # static (orb_r=49)
    target = _planet(1, 1, 70.0, 50.0, radius=1.5)  # orbiting (orb_r=20)
    omega = math.pi / 10.0
    world = _world(my_id=0, planets=[src, target], omega=omega)
    # Aim at target's CURRENT (t=0) position.
    angle = math.atan2(target.y - src.y, target.x - src.x)

    fate_now = predict_fleet_fate(src, target, angle, ships=10, world=world,
                                  wait_N=0)
    fate_late = predict_fleet_fate(src, target, angle, ships=10, world=world,
                                   wait_N=10)
    assert fate_now.outcome == "target", (
        f"wait_N=0: aim at t=0 target should hit, got {fate_now}")
    assert fate_late.outcome != "target", (
        f"wait_N=10: target has half-orbited away; aim line shouldn't hit it. "
        f"got {fate_late}")
