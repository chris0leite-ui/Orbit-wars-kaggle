"""Unit tests for `launch_reaches_target` physics validation."""

from __future__ import annotations

import math

from lib.goal_planner.validate import launch_reaches_target
from lib.trajectory_layer import World
from tests.scenarios.base import _obs, _planet


def _world(planets, step=0, player=0, episode_steps=500):
    obs = _obs(planets=planets, step=step, player=player)
    cfg = {"episodeSteps": episode_steps}
    return World.from_obs(obs, cfg)


def test_validate_clear_straight_line_passes():
    # p0 (10, 50) -> p1 (30, 50): straight east, no sun in the way.
    planets = [
        _planet(0, owner=0, x=10.0, y=50.0, ships=100, production=1),
        _planet(1, owner=-1, x=30.0, y=50.0, ships=5, production=1),
    ]
    world = _world(planets)
    src = world._planet_by_id[0]
    tgt = world._planet_by_id[1]
    # Angle 0 = east.
    assert launch_reaches_target(src, tgt, 0.0, 10, world) is True


def test_validate_through_sun_fails():
    # Smoking-gun case from seed 0, turn 0:
    # p0 (73, 74) launching at -2.35 rad (south-west) toward a target
    # on the far side of the board passes straight through the sun at
    # (50, 50). Validation must return False.
    planets = [
        _planet(0, owner=0, x=73.0, y=74.0, ships=100, production=1),
        _planet(1, owner=-1, x=26.0, y=26.0, ships=5, production=1),
    ]
    world = _world(planets)
    src = world._planet_by_id[0]
    tgt = world._planet_by_id[1]
    # Aim from (73,74) toward (26,26): angle = atan2(26-74, 26-73)
    # ≈ atan2(-48, -47) ≈ -2.35 rad. This line passes through (~50, 50).
    angle = math.atan2(tgt.current_y - src.current_y,
                       tgt.current_x - src.current_x)
    assert abs(angle - (-2.35)) < 0.05, (
        f"angle should be near -2.35 (through sun); got {angle}"
    )
    assert launch_reaches_target(src, tgt, angle, 10, world) is False


def test_validate_intervening_planet_fails():
    # p0 (10, 50) → p2 (90, 50). p1 (50, 50) sits directly between
    # them. Fleet should hit p1, not p2.
    planets = [
        _planet(0, owner=0, x=10.0, y=50.0, ships=100, production=1),
        _planet(1, owner=1, x=50.0, y=50.0, ships=20, production=1, radius=5.0),
        _planet(2, owner=-1, x=90.0, y=50.0, ships=5, production=1),
    ]
    world = _world(planets)
    src = world._planet_by_id[0]
    tgt = world._planet_by_id[2]
    # Aim straight east (angle=0); will hit p1 first.
    assert launch_reaches_target(src, tgt, 0.0, 10, world) is False
