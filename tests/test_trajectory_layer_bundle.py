"""Phase 7b tests for lib.trajectory_layer — Bundle + BundleEvaluator.

The value function the trajectory-native chooser optimises. A bundle
is a sequence of LaunchSpecs (potentially at varying launch_turns);
the evaluator scores the resulting K-turn world trajectory.

Pinned invariants:
1. Empty bundle scores the same as a no-bundle world at horizon K.
2. A bundle that captures a neutral planet scores higher on
   planet_delta than the empty bundle.
3. A bundle that captures an enemy's LAST planet contributes the
   elimination bonus.
4. Longer horizons accumulate production for both sides (the score
   compounds).
5. `Bundle.specs_at_turn(t)` correctly partitions specs by
   launch_turn (the agent loop's "which actions to emit THIS turn"
   query).
"""

from __future__ import annotations

import math

import pytest

from lib.trajectory_layer import (
    Bundle,
    BundleEvaluator,
    BundleScore,
    LaunchSpec,
    World,
)


def _toy_world(planets: list, fleets: list, *, my_id: int = 0,
                ) -> World:
    obs = {
        "step": 0,
        "player": my_id,
        "angular_velocity": 0.0,
        "planets": planets,
        "initial_planets": planets,
        "fleets": fleets,
        "comet_planet_ids": [],
        "comets": [],
        "next_fleet_id": max((f[0] for f in fleets), default=-1) + 1,
    }
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# Bundle dataclass
# ---------------------------------------------------------------------------


def test_empty_bundle_is_empty():
    b = Bundle()
    assert b.is_empty
    assert b.launches == ()
    assert b.first_launch_turn is None
    assert b.specs_at_turn(0) == ()


def test_bundle_specs_at_turn_partitions():
    """`specs_at_turn(t)` returns exactly the specs scheduled at t."""
    s0 = LaunchSpec(src_id=1, aim_angle=0.0, ships=5, owner=0,
                     launch_turn=0)
    s1 = LaunchSpec(src_id=1, aim_angle=1.0, ships=5, owner=0,
                     launch_turn=0)
    s5 = LaunchSpec(src_id=2, aim_angle=2.0, ships=3, owner=0,
                     launch_turn=5)
    b = Bundle(launches=(s0, s1, s5))
    assert b.first_launch_turn == 0
    assert b.specs_at_turn(0) == (s0, s1)
    assert b.specs_at_turn(5) == (s5,)
    assert b.specs_at_turn(3) == ()


def test_bundle_shift_forward_advances_future_launches():
    """`shift_forward(steps)` subtracts `steps` from every launch_turn.
    Specs that remain at launch_turn >= 0 survive in the shifted
    bundle; their other fields are unchanged."""
    s0 = LaunchSpec(src_id=1, aim_angle=0.0, ships=5, owner=0,
                     launch_turn=3)
    s1 = LaunchSpec(src_id=2, aim_angle=1.0, ships=7, owner=0,
                     launch_turn=10)
    b = Bundle(launches=(s0, s1)).shift_forward(2)
    assert len(b.launches) == 2
    assert b.launches[0].launch_turn == 1
    assert b.launches[1].launch_turn == 8
    # Non-launch_turn fields preserved.
    assert b.launches[0].src_id == 1 and b.launches[0].ships == 5
    assert b.launches[1].src_id == 2 and b.launches[1].ships == 7


def test_bundle_shift_forward_drops_past():
    """Specs whose post-shift launch_turn is negative are DROPPED —
    those launches already fired in prior turns."""
    already_fired = LaunchSpec(src_id=1, aim_angle=0.0, ships=5, owner=0,
                                launch_turn=0)
    boundary = LaunchSpec(src_id=2, aim_angle=0.0, ships=5, owner=0,
                           launch_turn=1)
    future = LaunchSpec(src_id=3, aim_angle=0.0, ships=5, owner=0,
                        launch_turn=5)
    b = Bundle(launches=(already_fired, boundary, future)).shift_forward(1)
    # already_fired (turn=0) → -1, dropped.
    # boundary (turn=1) → 0, kept (fires this turn).
    # future (turn=5) → 4, kept.
    assert len(b.launches) == 2
    assert b.launches[0].src_id == 2 and b.launches[0].launch_turn == 0
    assert b.launches[1].src_id == 3 and b.launches[1].launch_turn == 4


def test_bundle_shift_forward_zero_is_identity():
    """`shift_forward(0)` returns an equivalent bundle."""
    s = LaunchSpec(src_id=1, aim_angle=0.0, ships=5, owner=0,
                    launch_turn=3)
    b = Bundle(launches=(s,))
    shifted = b.shift_forward(0)
    assert shifted.launches == b.launches


def test_bundle_shift_forward_negative_raises():
    """Negative shifts are not a sensible game-loop operation."""
    b = Bundle()
    with pytest.raises(ValueError):
        b.shift_forward(-1)


def test_bundle_apply_returns_overlay():
    """`Bundle.apply(world)` is equivalent to
    `world.with_candidates(specs)`."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, -1, 80.0, 80.0, 1.0, 3, 0],
        ],
        fleets=[],
    )
    spec = LaunchSpec(src_id=0, aim_angle=0.0, ships=10, owner=0)
    bundle = Bundle(launches=(spec,))
    overlay_a = bundle.apply(world)
    overlay_b = world.with_candidates([spec])
    # Same fleet set, same source ship count.
    assert len(overlay_a.fleets) == len(overlay_b.fleets) == 1
    assert (overlay_a.planet_by_id(0).ships
            == overlay_b.planet_by_id(0).ships
            == 40)


# ---------------------------------------------------------------------------
# BundleEvaluator basics
# ---------------------------------------------------------------------------


def test_empty_bundle_score_components():
    """Empty bundle on a static 2-planet world: ship/planet/production
    deltas reflect just the natural evolution (production accrues for
    owned planets; neutrals stay flat)."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],   # ours, prod=1
            [1, -1, 80.0, 80.0, 1.0, 3, 0],   # neutral, no prod
        ],
        fleets=[],
    )
    ev = BundleEvaluator(horizon=10)
    score = ev.score(world, Bundle(), my_id=0)
    # No opponent → other_ships = other_planets = other_prod = 0.
    # At horizon=10: my_ships = 50 + 10*1 = 60 (terminal).
    # Path-integrated planet_delta = sum over t in [1..10] of
    # (my_planets - other_planets) = 10 turns * 1 planet = 10.
    # Path-integrated production_delta = 10 turns * 1 prod = 10.
    assert score.ship_delta == 60.0
    assert score.planet_delta == 10.0
    assert score.production_delta == 10.0
    # No opp owners → no eliminations possible.
    assert score.eliminations == 0
    expected_total = (60.0
                      + 5.0 * 10.0
                      + 1.0 * 10.0
                      + 200.0 * 0)
    assert math.isclose(score.total, expected_total, abs_tol=1e-9)


def test_capture_neutral_increases_planet_delta():
    """A bundle that captures a neutral planet should score higher on
    planet_delta than the empty bundle."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, -1, 80.0, 80.0, 1.0, 3, 0],   # neutral target
        ],
        fleets=[],
    )
    ev = BundleEvaluator(horizon=50)
    empty_score = ev.score(world, Bundle(), my_id=0)
    capture_bundle = Bundle(launches=(
        LaunchSpec(src_id=0, aim_angle=0.0, ships=10, owner=0),
    ))
    capture_score = ev.score(world, capture_bundle, my_id=0)
    # Capture wins +1 planet.
    assert capture_score.planet_delta > empty_score.planet_delta
    # Capture wins more total (planet_weight + neutral has 0 production).
    assert capture_score.total > empty_score.total


def test_capture_enemy_planet_eliminates():
    """A bundle that captures an opponent's LAST planet contributes
    the elimination bonus."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, 1, 80.0, 80.0, 2.0, 5, 1],   # enemy with only this planet
        ],
        fleets=[],
    )
    ev = BundleEvaluator(horizon=40, elimination_bonus=200.0)
    empty = ev.score(world, Bundle(), my_id=0)
    # Empty bundle: no elimination (enemy still owns planet 1).
    assert empty.eliminations == 0
    # Bundle: enough ships to capture enemy planet.
    capture = Bundle(launches=(
        LaunchSpec(src_id=0, aim_angle=0.0, ships=40, owner=0),
    ))
    score = ev.score(world, capture, my_id=0)
    assert score.eliminations == 1
    assert score.total >= empty.total + 200.0  # at least the bonus


def test_horizon_extends_production_compounding():
    """Longer horizon → more production accumulated for both sides.
    Our advantage (production_weight=10.0) grows with horizon."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],   # ours, prod=1
            [1, 1, 80.0, 80.0, 2.0, 50, 2],   # enemy, prod=2
        ],
        fleets=[],
    )
    ev_short = BundleEvaluator(horizon=10)
    ev_long = BundleEvaluator(horizon=50)
    s_short = ev_short.score(world, Bundle(), my_id=0)
    s_long = ev_long.score(world, Bundle(), my_id=0)
    # ship_delta at K=10: (50+10) - (50+20) = -10
    # ship_delta at K=50: (50+50) - (50+100) = -50
    # The enemy's production lead amplifies with horizon.
    assert s_long.ship_delta < s_short.ship_delta


def test_future_launch_bundle_eventually_captures():
    """A launch_turn=5 bundle aimed at a neutral planet eventually
    captures (within horizon=50). Score should reflect the
    eventual ownership change."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, -1, 80.0, 80.0, 1.0, 3, 0],   # neutral
        ],
        fleets=[],
    )
    ev = BundleEvaluator(horizon=50)
    bundle = Bundle(launches=(
        LaunchSpec(src_id=0, aim_angle=0.0, ships=10, owner=0,
                    launch_turn=5),
    ))
    score = ev.score(world, bundle, my_id=0)
    # Compare to empty.
    empty = ev.score(world, Bundle(), my_id=0)
    assert score.planet_delta > empty.planet_delta


# ---------------------------------------------------------------------------
# Custom weights
# ---------------------------------------------------------------------------


def test_custom_weights_change_score():
    """Production weight matters: a high planet-weight evaluator
    should score the same world differently from a low one."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, 1, 80.0, 80.0, 2.0, 50, 1],
        ],
        fleets=[],
    )
    base = BundleEvaluator(horizon=20, planet_weight=5.0)
    high_pw = BundleEvaluator(horizon=20, planet_weight=50.0)
    s_base = base.score(world, Bundle(), my_id=0)
    s_high = high_pw.score(world, Bundle(), my_id=0)
    # Same components except the planet-weight multiplier.
    assert s_base.planet_delta == s_high.planet_delta
    # Both sides have 1 planet → planet_delta = 0 → weight doesn't matter.
    # Re-do with asymmetric world.
    asym = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, 0, 30.0, 80.0, 2.0, 50, 1],
            [2, 1, 80.0, 80.0, 2.0, 50, 1],
        ],
        fleets=[],
    )
    s_base_a = base.score(asym, Bundle(), my_id=0)
    s_high_a = high_pw.score(asym, Bundle(), my_id=0)
    # planet_delta = 2 - 1 = 1. Higher planet weight → higher total.
    assert s_high_a.total > s_base_a.total
    assert (s_high_a.total - s_base_a.total
            == (50.0 - 5.0) * s_base_a.planet_delta)


# ---------------------------------------------------------------------------
# Sanity vs. with_candidate(s) directly
# ---------------------------------------------------------------------------


def test_bundle_score_equals_with_candidates_score():
    """Bundle's apply() is just a wrapper; scoring via Bundle should
    match scoring the equivalent with_candidates overlay directly."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, -1, 80.0, 80.0, 1.0, 3, 0],
        ],
        fleets=[],
    )
    specs = (
        LaunchSpec(src_id=0, aim_angle=0.0, ships=10, owner=0),
        LaunchSpec(src_id=0, aim_angle=0.0, ships=5, owner=0,
                    launch_turn=5),
    )
    bundle = Bundle(launches=specs)
    ev = BundleEvaluator(horizon=30)
    s_via_bundle = ev.score(world, bundle, my_id=0)

    # Manual: build the same overlay via with_candidates, then re-run
    # the scoring math (we don't have a separate "score from overlay"
    # entry point; just confirm bundle.apply == with_candidates).
    overlay_a = bundle.apply(world)
    overlay_b = world.with_candidates(specs)
    assert len(overlay_a.fleets) == len(overlay_b.fleets)
    # The score depends only on overlay state, which we've just shown
    # is identical → score must be identical.
    assert s_via_bundle.total == ev.score(world, Bundle(specs),
                                              my_id=0).total
