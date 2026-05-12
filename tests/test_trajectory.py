"""Tests for lib/trajectory.predict_fleet_fate — full-trajectory ray-cast.

The earlier guards (sun_avoid, oob_guard, path_clears_other_planets) only
simulated up to the target's predicted arrival point. Live-replay
evidence showed ~10.7% of our fleets either flew OOB or into the sun
because the prediction-tail wasn't checked. predict_fleet_fate walks
the full straight-line trajectory until the first collision.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from agent import World, predict_fleet_fate


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


def test_fleet_reaches_target_when_aimed_directly():
    """Straight shot at a static planet (path off the sun) → 'target'."""
    src = _planet(0, 0, 10.0, 20.0, radius=2.0)
    target = _planet(1, 1, 80.0, 20.0, radius=2.0)
    world = _world(my_id=0, planets=[src, target])
    angle = math.atan2(target.y - src.y, target.x - src.x)
    fate = predict_fleet_fate(src, target, angle, ships=10, world=world)
    assert fate.outcome == "target"
    assert fate.hit_planet_id == 1


def test_fleet_oob_when_no_planet_in_path():
    """Aim into empty space → fleet flies until OOB."""
    # Source off-centre, aim away from any planet AND away from the sun.
    src = _planet(0, 0, 20.0, 20.0, radius=1.5)
    target = _planet(1, 1, 80.0, 80.0, radius=1.5)  # NE corner
    world = _world(my_id=0, planets=[src, target])
    # Aim straight west — leaves the source heading toward x<0.
    angle = math.pi  # -x direction
    fate = predict_fleet_fate(src, target, angle, ships=10, world=world)
    assert fate.outcome == "oob"


def test_fleet_dies_in_sun_on_through_path():
    """Aim through the sun → outcome 'sun'."""
    # Source NW corner, target SE corner — path crosses the sun at (50, 50).
    src = _planet(0, 0, 10.0, 10.0, radius=1.5)
    target = _planet(1, 1, 90.0, 90.0, radius=1.5)
    world = _world(my_id=0, planets=[src, target])
    angle = math.atan2(target.y - src.y, target.x - src.x)
    fate = predict_fleet_fate(src, target, angle, ships=10, world=world)
    assert fate.outcome == "sun"


def test_fleet_hits_other_planet_when_path_blocked():
    """Aim at target but another planet sits in the way → outcome 'planet'."""
    # Pick a y-coordinate clear of the sun (sun at y=50; use y=20).
    src = _planet(0, 0, 10.0, 20.0, radius=1.5)
    blocker = _planet(2, -1, 45.0, 20.0, radius=2.0)
    target = _planet(1, 1, 80.0, 20.0, radius=2.0)
    world = _world(my_id=0, planets=[src, blocker, target])
    angle = math.atan2(target.y - src.y, target.x - src.x)
    fate = predict_fleet_fate(src, target, angle, ships=10, world=world)
    assert fate.outcome == "planet"
    assert fate.hit_planet_id == 2  # the blocker, not the intended target


def test_overshoot_past_target_then_oob():
    """Fleet aimed slightly off target (tangent miss) keeps flying past
    and exits the board. This is the bug the old oob_guard missed —
    endpoint check thought the predicted spot was inside the box, but
    the actual flight overshot.

    Source low-y, target high-y. Aim a few degrees off so the path misses
    the target by enough that no swept-pair hit fires. We stay east of the
    sun (x = 20 column) so the path doesn't accidentally die in it.
    """
    src = _planet(0, 0, 20.0, 10.0, radius=1.5)
    target = _planet(1, 1, 20.0, 80.0, radius=1.5)
    world = _world(my_id=0, planets=[src, target])
    angle = math.atan2(target.y - src.y, target.x - src.x) + math.radians(10)
    fate = predict_fleet_fate(src, target, angle, ships=10, world=world)
    # Either OOB or a different planet collision is acceptable — the
    # critical thing is the fate is NOT 'target'.
    assert fate.outcome != "target"
    if fate.outcome == "planet":
        assert fate.hit_planet_id != target.id
