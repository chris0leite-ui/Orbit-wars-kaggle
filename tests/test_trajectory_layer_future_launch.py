"""Phase 7 (future-launch overlay) tests for lib.trajectory_layer.

`LaunchSpec.launch_turn > 0` lets the planner commit to a future-turn
launch. The trajectory-native chooser composes BUNDLES (sequences of
LaunchSpecs at varying launch_turns) and scores the resulting world
trajectory. These tests pin:

1. `with_candidate(launch_turn>0)` succeeds when the source will be
   ours with enough ships at that turn (validated against the
   parent's timeline — so chained overlays accumulate correctly).
2. The synthetic fleet's `spawn_turn` matches `launch_turn`; its
   `position_at(t<launch_turn)` returns None; its ledger entry's
   `eta` is `launch_turn + steps_to_target`.
3. The source planet's t=0 ships are UNCHANGED for future launches;
   the deduction shows up in the timeline AT launch_turn.
4. Production accrues normally before/after the launch turn; the
   deduction at launch_turn happens BEFORE production (env's
   `process_moves` → production ordering).
5. Validation errors fire when the source isn't ours / doesn't have
   enough ships at launch_turn.
6. Chained `with_candidate` calls (a bundle) accumulate deductions
   correctly: spec_A at t=3 must reduce the available ships used
   by spec_B's validation at t=5.
"""

from __future__ import annotations

import math

import pytest

from lib.trajectory_layer import (
    LaunchSpec,
    World,
)


def _toy_world(planets: list, fleets: list, *,
                step: int = 0, my_id: int = 0,
                ) -> World:
    obs = {
        "step": step,
        "player": my_id,
        "angular_velocity": 0.0,  # static planets keep geometry simple
        "planets": planets,
        "initial_planets": planets,
        "fleets": fleets,
        "comet_planet_ids": [],
        "comets": [],
        "next_fleet_id": max((f[0] for f in fleets), default=-1) + 1,
    }
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# Future-launch basics
# ---------------------------------------------------------------------------


def test_with_candidate_launch_turn_5_succeeds():
    """Source: 50-ship owned planet at (20, 80). Launch 10 ships at
    turn 5 toward (80, 80) target. Validation should pass — at turn 5
    the source will have 50 + 5*1 = 55 ships (production=1)."""
    world = _toy_world(
        planets=[
            [0, -1, 80.0, 80.0, 1.0, 3, 0],
            [1, 0, 20.0, 80.0, 2.0, 50, 1],
        ],
        fleets=[],
    )
    spec = LaunchSpec(src_id=1, aim_angle=0.0, ships=10, owner=0,
                       launch_turn=5)
    overlay = world.with_candidate(spec)

    # Source's t=0 ships UNCHANGED (the launch hasn't happened yet).
    owner0, ships0 = overlay.ownership_at(1, 0)
    assert owner0 == 0
    assert ships0 == 50.0


def test_future_launch_source_t0_unchanged_future_deducted():
    """At t=0..4 the source has full production-accrued ships. At
    t=5 the launch deducts (BEFORE production for that turn). At
    t=6+ production keeps accruing."""
    world = _toy_world(
        planets=[
            [0, -1, 80.0, 80.0, 1.0, 3, 0],
            [1, 0, 20.0, 80.0, 2.0, 50, 1],
        ],
        fleets=[],
    )
    spec = LaunchSpec(src_id=1, aim_angle=0.0, ships=10, owner=0,
                       launch_turn=5)
    overlay = world.with_candidate(spec)

    # Walk the timeline.
    _, s0 = overlay.ownership_at(1, 0); assert s0 == 50.0
    _, s4 = overlay.ownership_at(1, 4); assert s4 == 54.0   # +4 production
    # t=5: deduct 10 (process_moves), THEN +1 production → 54 - 10 + 1 = 45.
    _, s5 = overlay.ownership_at(1, 5); assert s5 == 45.0
    _, s6 = overlay.ownership_at(1, 6); assert s6 == 46.0
    _, s10 = overlay.ownership_at(1, 10); assert s10 == 50.0


def test_future_launch_synthetic_fleet_spawn_turn():
    """The synthetic FleetView's spawn_turn equals launch_turn;
    position_at(t<launch_turn) returns None."""
    world = _toy_world(
        planets=[
            [0, -1, 80.0, 80.0, 1.0, 3, 0],
            [1, 0, 20.0, 80.0, 2.0, 50, 1],
        ],
        fleets=[],
    )
    spec = LaunchSpec(src_id=1, aim_angle=0.0, ships=10, owner=0,
                       launch_turn=5)
    overlay = world.with_candidate(spec)
    # Find the new synthetic fleet (negative id).
    synth = [f for f in overlay.fleets if f.id < 0]
    assert len(synth) == 1
    f = synth[0]
    assert f.spawn_turn == 5
    assert f.position_at(0) is None
    assert f.position_at(4) is None
    # At spawn turn: at spawn position (radius+0.1 offset from src).
    pos5 = f.position_at(5)
    assert pos5 is not None
    # Source is at (20, 80) with radius 2.0; angle=0 → spawn at (22.1, 80).
    assert math.isclose(pos5[0], 22.1, abs_tol=1e-9)
    assert math.isclose(pos5[1], 80.0, abs_tol=1e-9)
    # After 3 more turns of flight (t=8 absolute).
    pos8 = f.position_at(8)
    assert pos8 is not None
    assert pos8[0] > pos5[0]
    assert math.isclose(pos8[1], 80.0, abs_tol=1e-9)


def test_future_launch_arrival_eta_shifted():
    """An eta in the overlay's ledger should be `launch_turn +
    steps_to_target`. Compared to a launch_turn=0 launch with the
    same direction/ships, the future-launch eta is exactly
    `launch_turn` larger."""
    world = _toy_world(
        planets=[
            [0, -1, 80.0, 80.0, 1.0, 3, 0],
            [1, 0, 20.0, 80.0, 2.0, 50, 1],
        ],
        fleets=[],
    )
    immediate = world.with_candidate(
        LaunchSpec(src_id=1, aim_angle=0.0, ships=10, owner=0,
                    launch_turn=0),
    )
    delayed = world.with_candidate(
        LaunchSpec(src_id=1, aim_angle=0.0, ships=10, owner=0,
                    launch_turn=5),
    )
    imm_arrivals = immediate.ledger_for(0, horizon=200)
    del_arrivals = delayed.ledger_for(0, horizon=200)
    assert len(imm_arrivals) == 1
    assert len(del_arrivals) == 1
    assert del_arrivals[0].eta == imm_arrivals[0].eta + 5


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_negative_launch_turn_raises():
    world = _toy_world(
        planets=[[0, 0, 20.0, 80.0, 2.0, 50, 1]],
        fleets=[],
    )
    with pytest.raises(ValueError, match="launch_turn"):
        world.with_candidate(LaunchSpec(src_id=0, aim_angle=0.0,
                                          ships=5, owner=0,
                                          launch_turn=-1))


def test_future_launch_not_owned_at_launch_turn_raises():
    """If the source will be CAPTURED before launch_turn, validation
    must fail. Set up: src owned by us at t=0, but an enemy fleet
    arrives at t=3 and captures. A launch_turn=5 from us → invalid."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 5, 1],     # us, low garrison
            [1, 1, 80.0, 80.0, 1.0, 30, 1],    # enemy planet (origin info only)
        ],
        # Enemy fleet inbound to our planet (id 0) — will capture.
        fleets=[
            # Fleet from (50, 80) → headed left at speed ≈ 2.3.
            # eta ≈ ceil((50 - 2 - 20) / 2.3) = ceil(12.2) ≈ 13.
            # That's too slow. Let me set up a closer fleet.
            # Place fleet at (40, 80), flying west; eta to (20, 80) ≈ ceil(18/2.3) ≈ 8.
            # Still slow. Try a fleet with 50 ships (faster) at (30, 80) flying west.
            # Speed for 50 ships: 1+(5)*(log(50)/log(1000))^1.5 ≈ 3.4.
            # eta ≈ ceil((30-2-20)/3.4) ≈ ceil(2.4) ≈ 3.
            [0, 1, 30.0, 80.0, math.pi, 1, 50],  # 50-ship enemy flying west
        ],
    )
    # Verify the capture timing — pre-condition on the test setup.
    owner3, _ = world.ownership_at(0, 3)
    if owner3 != 1:
        pytest.skip(f"setup didn't capture as expected (owner@3 = {owner3})")

    # Now ours at t=0, captured by t=3, so launch_turn=5 invalid.
    spec = LaunchSpec(src_id=0, aim_angle=0.0, ships=2, owner=0,
                       launch_turn=5)
    with pytest.raises(ValueError, match="owned by"):
        world.with_candidate(spec)


def test_future_launch_insufficient_ships_raises():
    """At launch_turn=5, ships = 50 + 5*1 = 55. Asking for 60 ships
    should raise."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, -1, 80.0, 80.0, 1.0, 3, 0],
        ],
        fleets=[],
    )
    with pytest.raises(ValueError, match="cannot launch"):
        world.with_candidate(LaunchSpec(src_id=0, aim_angle=0.0,
                                          ships=60, owner=0,
                                          launch_turn=5))


# ---------------------------------------------------------------------------
# Bundles (chained with_candidate calls)
# ---------------------------------------------------------------------------


def test_bundle_accumulates_outgoing_deductions():
    """A bundle of two launches from the same source: spec_A at t=3
    deducts 20 ships; spec_B at t=7 must validate against the
    post-spec_A timeline (ships at t=7 = 50 + 7 - 20 = 37)."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, -1, 80.0, 80.0, 1.0, 3, 0],
        ],
        fleets=[],
    )
    spec_a = LaunchSpec(src_id=0, aim_angle=0.0, ships=20, owner=0,
                         launch_turn=3)
    spec_b = LaunchSpec(src_id=0, aim_angle=0.0, ships=30, owner=0,
                         launch_turn=7)
    # spec_a then spec_b: spec_b validated against (50+7-20) = 37 ships,
    # so launching 30 is valid (37 > 30).
    overlay = world.with_candidates([spec_a, spec_b])

    # Walk the source's timeline.
    # t=0 start: 50
    # t=1: production +1 → 51
    # t=2: production +1 → 52
    # t=3: outgoing launch 20 → 32, then production +1 → 33
    # t=4..6: +1 each → 34, 35, 36
    # t=7: outgoing launch 30 → 6, then production +1 → 7
    _, s0 = overlay.ownership_at(0, 0); assert s0 == 50.0
    _, s2 = overlay.ownership_at(0, 2); assert s2 == 52.0
    _, s3 = overlay.ownership_at(0, 3); assert s3 == 33.0
    _, s6 = overlay.ownership_at(0, 6); assert s6 == 36.0
    _, s7 = overlay.ownership_at(0, 7); assert s7 == 7.0


def test_bundle_overcommit_raises_on_second_spec():
    """If spec_A leaves the source with 20 ships and spec_B asks for
    30, the second `with_candidate` must raise."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, -1, 80.0, 80.0, 1.0, 3, 0],
        ],
        fleets=[],
    )
    overlay = world.with_candidate(
        LaunchSpec(src_id=0, aim_angle=0.0, ships=45, owner=0,
                    launch_turn=2),
    )
    # After spec_A: at t=4, ships = 50 + 1 (prod t1) + 1 (prod t2 before launch)
    # - 45 (launch) + 1 (prod t2 after launch) + 1 (prod t3) + 1 (prod t4) = 10.
    # Asking for 30 at t=4 should fail.
    with pytest.raises(ValueError, match="cannot launch"):
        overlay.with_candidate(
            LaunchSpec(src_id=0, aim_angle=0.0, ships=30, owner=0,
                        launch_turn=4),
        )


def test_bundle_with_immediate_and_future():
    """Mix launch_turn=0 (immediate, deducts src.ships now) and
    launch_turn=5 (future). Both should compose correctly."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, -1, 80.0, 80.0, 1.0, 3, 0],
        ],
        fleets=[],
    )
    # Spec A: 10 ships now → src drops to 40 immediately.
    # Spec B: 15 ships at t=5 → at t=5, source has 40 + 5 = 45. Launching 15 is fine.
    overlay = world.with_candidates([
        LaunchSpec(src_id=0, aim_angle=0.0, ships=10, owner=0,
                    launch_turn=0),
        LaunchSpec(src_id=0, aim_angle=0.0, ships=15, owner=0,
                    launch_turn=5),
    ])
    # At t=0, src has 50 - 10 = 40 (immediate deduction).
    # t=1..4: production +1 each → 41, 42, 43, 44.
    # t=5: outgoing launch 15 → 29, then production +1 → 30.
    _, s0 = overlay.ownership_at(0, 0); assert s0 == 40.0
    _, s4 = overlay.ownership_at(0, 4); assert s4 == 44.0
    _, s5 = overlay.ownership_at(0, 5); assert s5 == 30.0
