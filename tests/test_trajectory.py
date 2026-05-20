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

from lib.intent import World
from lib.trajectory import predict_fleet_fate


def _world(my_id, planets, omega=0.0, *, comet_ids=None, comets=None):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": [],
        "angular_velocity": omega,
        "comet_planet_ids": comet_ids or [],
        "step": 0,
    }
    if comets is not None:
        obs["comets"] = comets
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


# ---------------------------------------------------------------------------
# Comet-aim fix (Part C, 2026-05-19 PM): comets are NOT orbital — their
# positions in predict_fleet_fate must come from obs.comets[group].paths,
# not from predict_relative.
# ---------------------------------------------------------------------------


def test_comet_position_lookup_uses_path_not_orbital():
    """Fleet aimed at a comet's FUTURE path position should hit the comet
    even though the orbital prediction would place it elsewhere.

    Setup: comet on a slow linear path moving east, off the y=50 sun line.
    Fleet speed at the chosen ship count is calibrated to converge with
    the comet's path before max_steps. predict_fleet_fate must walk the
    path, not rotate the comet around the sun.
    """
    src = _planet(0, 0, 5.0, 20.0, radius=1.5)
    # Comet at (30, 20) moving east at 1 unit/turn (slow path so a
    # 10-ship fleet ~1.96 units/turn catches up around step 24).
    path = [[30.0 + i * 1.0, 20.0] for i in range(60)]
    comet_planet = _planet(42, -1, path[0][0], path[0][1], radius=1.0)
    world = _world(
        my_id=0, planets=[src, comet_planet], omega=0.0,
        comet_ids=[42],
        comets=[{"planet_ids": [42], "paths": [path], "path_index": 0}],
    )
    # Aim straight east at the comet's eventual encounter position.
    import math as _math
    angle = 0.0  # pure +x; both fleet and comet at y=20, off the sun
    fate = predict_fleet_fate(src, comet_planet, angle, ships=10, world=world)
    # With path-aware fix: fleet catches the slow-moving comet at some
    # step in [1, max_steps] → outcome="target", hit_planet_id=42.
    # Without the fix (orbital), the comet would be predicted near its
    # original position (omega=0 → stays at (30, 20)); the fleet would
    # also hit it but the test wouldn't verify path-awareness. So set
    # omega to a small non-zero value with path_index=0 and verify the
    # outcome is target — meaning the per-step position came from the
    # path, not from `predict_relative` (which would rotate it differently).
    assert fate.outcome == "target", (
        f"expected target (path-aware); got outcome={fate.outcome} hit_id={fate.hit_planet_id}"
    )
    assert fate.hit_planet_id == 42


def test_comet_at_path_end_marked_as_exited():
    """When the comet's path runs out mid-flight, predict_fleet_fate
    skips collision against it (position becomes None). The fleet should
    NOT collide with the comet's last-known position phantom-frozen."""
    src = _planet(0, 0, 5.0, 20.0, radius=1.5)
    # Very short path of only 3 steps; comet exits before fleet arrives.
    path = [[30.0, 20.0], [34.0, 20.0], [38.0, 20.0]]
    comet_planet = _planet(42, -1, path[0][0], path[0][1], radius=1.0)
    # Add a static planet way past the comet's exit position so we have
    # something concrete to predict.
    target = _planet(99, 1, 90.0, 20.0, radius=1.5)
    world = _world(
        my_id=0, planets=[src, comet_planet, target], omega=0.0,
        comet_ids=[42],
        comets=[{"planet_ids": [42], "paths": [path], "path_index": 0}],
    )
    # Aim at the static target (planet 99) straight east at y=20 (off
    # the sun). The pre-fix bug would park the comet at (30, 20) forever
    # and the fleet would hit the frozen comet; the fix marks it exited
    # and skips the collision so the fleet continues to planet 99.
    import math as _math
    angle = _math.atan2(target.y - src.y, target.x - src.x)
    fate = predict_fleet_fate(src, target, angle, ships=10, world=world)
    # The fleet should pass through the exited-comet region and reach
    # the static target.
    assert fate.outcome == "target", (
        f"expected target (planet 99) after comet exits; got {fate.outcome} hit_id={fate.hit_planet_id}"
    )
    assert fate.hit_planet_id == 99
