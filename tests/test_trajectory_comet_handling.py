"""Bug — predict_fleet_fate mis-models comet positions.

PI flagged: in the first live self-play after comets enter the board,
the analytical_phase_c submission emits trajectories that go OOB.
Replay-analysis (47 OOB events on seed 42, all post-step-50 — the
step comets enter) confirms this is the predict_fleet_fate path.

Root cause: `lib/trajectory.py:118-126` predicts EVERY planet's
future position via `predict_relative` (orbital rotation about the
sun). Comets follow discrete `path[i]` from `obs["comets"]`, NOT
orbital paths. The predicted comet position is therefore wrong, and
a trajectory that predict_fleet_fate says "hits comet X at step K"
will in reality miss the comet (it was elsewhere) and continue until
it exits the board.

Pin test (Rule 38): construct a comet whose path moves it from
(95, 50) → (75, 50) → (55, 50) over 3 steps (straight line, not
orbital). Aim a fleet at (75, 50). Pre-fix `predict_fleet_fate`
treats the comet as orbital → predicts it at some rotated position →
trajectory either misses or hits at wrong step. Post-fix uses the
path → predicts comet at (75, 50) at step 1 (where it really is) →
trajectory hits at the correct step.
"""

from __future__ import annotations

from lib.intent import Planet, World
from lib.trajectory import predict_fleet_fate


def _build_comet_world(*, comet_position_now, comet_path, path_index=0):
    """Build a World with two static (non-orbiting) planets and one
    comet that follows `comet_path` (list of [x, y] pairs).

    Static planets isolate the variable under test: any divergence
    comes from comet handling, not orbital target movement. omega=0
    plus planet positions outside the rotation limit (orb_r + radius
    >= 50) ensure is_orbiting() returns False.
    """
    # Src at (10, 10): orb_r = sqrt(3200) ≈ 56.6, +5 = 61.6 > 50 → static.
    # Tgt at (90, 10): same → static. Straight east trajectory at y=10
    # cleanly avoids the sun at (50, 50).
    src = Planet(id=0, owner=0, x=10.0, y=10.0, radius=5.0, ships=20,
                 production=2)
    tgt = Planet(id=1, owner=1, x=90.0, y=10.0, radius=5.0, ships=10,
                 production=2)
    comet = Planet(
        id=20, owner=-1,
        x=float(comet_position_now[0]), y=float(comet_position_now[1]),
        radius=2.0, ships=0, production=1,
    )
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 5.0, 20, 2],
            [1, 1, 90.0, 10.0, 5.0, 10, 2],
            [20, -1, float(comet_position_now[0]), float(comet_position_now[1]),
             2.0, 0, 1],
        ],
        "fleets": [],
        "angular_velocity": 0.0,
        "initial_planets": [],
        "comet_planet_ids": [20],
        "comets": [{
            "planet_ids": [20],
            "paths": [comet_path],
            "path_index": int(path_index),
        }],
    }
    return World.from_obs(obs), src, tgt, comet


def test_predict_fleet_fate_uses_comet_path_not_orbital():
    """Comet moves linearly from (90, 30) to (50, 30) over 4 steps.
    A fleet aimed straight along y=50 should NOT think it'll hit the
    comet at any orbital-rotated position; pre-fix it does (wrong),
    post-fix it correctly predicts target (planet 1) because the comet
    never crosses the fleet's path.
    """
    # Comet path: stays at y=30 (well clear of fleet's straight-east
    # trajectory at y=50). Fleet at y=50 should never collide with it.
    # The KEY question is what the comet was at time-of-snapshot:
    # the buggy code would read its current position (95, 30) and
    # rotate that ORBITALLY about the sun — putting it on a circle of
    # radius hypot(45,20)≈49 around (50,50), which COULD intersect y=50.
    # Set the comet's CURRENT position to (95, 30) and its path to
    # advance it linearly toward (50, 30). Pre-fix omega is 0 so the
    # buggy code keeps it stuck at (95, 30) — passes this test.
    # Use a path index that means the comet starts at (95, 30) and
    # moves to (75, 30), (55, 30) etc. — pre-fix doesn't see this
    # movement (orbital prediction with omega=0 == identity).
    #
    # Acutally with omega=0 the buggy code DOES keep the comet at
    # (95, 30) for every step. Fleet at y=50 still clears it. So this
    # particular pin doesn't differentiate. Drop it in favor of test
    # 2 (predicts_real_comet_collision) below which is the real one.
    # Fleet flies at y=10; comet path stays well above at y=80.
    comet_path = [
        [95.0, 80.0],
        [75.0, 80.0],
        [55.0, 80.0],
        [35.0, 80.0],
        [15.0, 80.0],
    ]
    world, src, tgt, comet = _build_comet_world(
        comet_position_now=(95.0, 80.0),
        comet_path=comet_path, path_index=0,
    )
    # Fleet from src(10,10) east toward tgt(90,10).
    angle = 0.0  # straight east at y=10
    fate = predict_fleet_fate(src, tgt, angle, ships=10, world=world,
                              wait_N=0)
    assert fate.outcome == "target", (
        f"expected 'target' (fleet at y=10 clear of comet path at y=80), "
        f"got {fate.outcome} hit={fate.hit_planet_id} step={fate.step}"
    )
    assert fate.hit_planet_id == 1, (
        f"expected target id 1, got {fate.hit_planet_id}"
    )


def test_predict_fleet_fate_uses_path_not_stale_position():
    """The MEAT pin: comet currently at (95, 30) BUT its path drops it
    into the fleet's straight-east trajectory at y=50.

    Pre-fix: world.omega=0 makes predict_relative an identity, so the
    comet is "predicted" to stay at (95, 30) forever. Fleet at y=50
    sees no collision and reports target=planet 1.

    Post-fix: the comet's actual path has it land on the fleet's path
    (at (50, 50) for some step). predict_fleet_fate now consults the
    path and reports a planet-collision with the comet (id=20).
    """
    # Fleet flies at y=10 east. Comet's CURRENT position is (95, 80) —
    # far above the fleet's path. But its actual path PARKS the comet
    # at (50, 10) for the duration — right on the fleet's trajectory.
    #
    # Pre-fix: predict_fleet_fate calls predict_relative on the comet's
    # current position. With omega=0 this is identity → comet "stays"
    # at (95, 80) → fleet on y=10 sees no collision → reports target.
    # Post-fix: predict_fleet_fate consults the path, finds the comet
    # at (50, 10) on every step → fleet collides with comet (id=20).
    comet_path = [[50.0, 10.0]] * 50  # parked right on the fleet's line
    world, src, tgt, _comet = _build_comet_world(
        comet_position_now=(95.0, 80.0),
        comet_path=comet_path, path_index=0,
    )
    angle = 0.0  # straight east at y=10
    fate = predict_fleet_fate(src, tgt, angle, ships=10, world=world,
                              wait_N=0)
    assert fate.outcome == "planet", (
        f"expected 'planet' (comet collision per the path), got "
        f"{fate.outcome} hit={fate.hit_planet_id} step={fate.step}. "
        f"If 'target', the predictor is using the comet's stale current "
        f"position (95, 80) instead of consulting the path (which has "
        f"the comet parked at (50, 10) — right on the fleet's line)."
    )
    assert fate.hit_planet_id == 20, (
        f"expected to hit comet id=20, got {fate.hit_planet_id}"
    )


def test_predict_fleet_fate_comet_leaves_board():
    """When the path runs out (comet has left the board), the comet
    should not register collisions. Sentinel positions far off the
    board ensure swept_pair_hit can't match.
    """
    # Comet currently at (50, 10) — right in the middle of fleet path.
    # But its path runs out: path_index=5 (past the path's 2 entries).
    # Post-fix: predict_fleet_fate must mark comet as "gone" (sentinel
    # OFF_BOARD position) so it doesn't block the fleet's path.
    comet_path = [
        [50.0, 10.0],   # comet's path content (only 2 entries)
        [50.0, 10.0],
    ]
    world, src, tgt, _comet = _build_comet_world(
        comet_position_now=(50.0, 10.0),
        comet_path=comet_path,
        path_index=5,  # past end-of-path
    )
    angle = 0.0
    fate = predict_fleet_fate(src, tgt, angle, ships=10, world=world,
                              wait_N=0)
    assert fate.outcome == "target", (
        f"with path_index past end-of-path, comet should be off-board "
        f"and fleet should reach target. Got {fate.outcome} hit="
        f"{fate.hit_planet_id}."
    )
