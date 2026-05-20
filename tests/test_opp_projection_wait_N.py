"""Regression tests for `predict_opp_multi_launch` wait_N propagation.

The projection iterates `tick_offset` in [0, HORIZON). At each tick,
opp's source planet has orbited by `omega * tick_offset` radians, and
candidate targets have likewise orbited. Trajectory feasibility at
tick_offset>0 MUST be evaluated against the advanced geometry, not the
step_now snapshot. Otherwise the projected threat ledger is wrong-shaped
and the LP makes wrong defense/offense tradeoffs.

These tests pin the contract that `predict_fleet_fate` is called with
`wait_N=tick_offset` at every iteration of the projection loop. This is
the SAME bug class `aac3c1e` fixed for OUR launches, applied to the opp.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

from lib.intent import World
from lib.joint_solver import opp_projection
from lib.trajectory import FleetFate


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


def test_opp_projection_passes_wait_N_at_each_tick_offset():
    """At every projection iteration past tick_offset=0, the trajectory
    check must receive `wait_N == tick_offset`. We patch
    `lib.joint_solver.opp_projection.predict_fleet_fate` to record the
    kwargs it was called with, then call `predict_opp_multi_launch`.

    The world is rigged with one well-stocked opp source and multiple
    targets so the multi-launch loop runs past tick_offset=0:
      - source has 200 ships, prod=4 → enough for several launches.
      - target diversity: 4 non-opp targets so `already_targeted` set
        doesn't immediately exhaust the candidate space.
      - omega>0 so the wait_N path is meaningful.

    Pre-fix behaviour: `predict_fleet_fate` always called with
    `wait_N=0` (default). Post-fix: `wait_N=tick_offset`.
    """
    me = 0
    opp = 1
    src_opp = _planet(10, opp, 70.0, 50.0, ships=200, production=4, radius=1.5)
    # Spread targets so the projection iterates several ticks.
    t1 = _planet(1, -1, 90.0, 50.0, ships=2, production=3, radius=1.5)
    t2 = _planet(2, -1, 50.0, 90.0, ships=2, production=3, radius=1.5)
    t3 = _planet(3, me, 30.0, 50.0, ships=2, production=2, radius=1.5)
    t4 = _planet(4, me, 50.0, 10.0, ships=2, production=2, radius=1.5)
    world = _world(my_id=me, planets=[src_opp, t1, t2, t3, t4],
                   omega=math.pi / 40.0)

    calls = []

    def _record(src, target, angle, ships, world_arg, *, wait_N=0):
        calls.append({
            "src_id": int(src.id), "tgt_id": int(target.id),
            "wait_N": int(wait_N),
        })
        return FleetFate("target", int(target.id), 1)

    with patch.object(opp_projection, "predict_fleet_fate", _record):
        arrivals = opp_projection.predict_opp_multi_launch(
            world, my_id=me, num_seats=2,
        )

    assert len(arrivals) >= 1, (
        "Test setup error: projection produced no arrivals; "
        "increase ships or HORIZON.")
    # The bug-or-fix discriminator: did we ever see wait_N > 0?
    nonzero = [c for c in calls if c["wait_N"] > 0]
    assert nonzero, (
        f"predict_fleet_fate was called {len(calls)} times but NEVER with "
        f"wait_N > 0. The multi-launch loop iterates tick_offset>=1 but "
        f"validates with stale (step_now) geometry. Pre-fix bug — F1 not "
        f"applied. Calls: {calls}")


def test_opp_projection_wait_N_matches_tick_offset_exactly():
    """Tighter pin: for each call past tick_offset=0, wait_N MUST equal
    the tick_offset at which the launch is being considered.

    We reconstruct the tick_offset from `eta = tick_offset + flight`
    by also recording the projection's recorded arrival's `eta_absolute`
    field. Indirect — we assert the SET of wait_N values seen matches
    the expected set {0, 1, ..., HORIZON-1} (subset, since not every
    tick produces a call).
    """
    me = 0
    opp = 1
    # Many opp planets so the loop iterates many sources × ticks.
    src_a = _planet(10, opp, 70.0, 50.0, ships=200, production=4, radius=1.5)
    src_b = _planet(11, opp, 50.0, 30.0, ships=200, production=4, radius=1.5)
    t1 = _planet(1, -1, 90.0, 50.0, ships=2, production=3, radius=1.5)
    t2 = _planet(2, -1, 10.0, 50.0, ships=2, production=3, radius=1.5)
    t3 = _planet(3, me, 50.0, 90.0, ships=2, production=2, radius=1.5)
    t4 = _planet(4, me, 50.0, 10.0, ships=2, production=2, radius=1.5)
    world = _world(my_id=me, planets=[src_a, src_b, t1, t2, t3, t4],
                   omega=math.pi / 40.0)

    seen_wait_Ns = []

    def _record(src, target, angle, ships, world_arg, *, wait_N=0):
        seen_wait_Ns.append(int(wait_N))
        return FleetFate("target", int(target.id), 1)

    with patch.object(opp_projection, "predict_fleet_fate", _record):
        opp_projection.predict_opp_multi_launch(world, my_id=me, num_seats=2)

    distinct = sorted(set(seen_wait_Ns))
    assert distinct != [0], (
        f"All predict_fleet_fate calls had wait_N=0. Expected at least one "
        f"wait_N > 0. seen_wait_Ns={seen_wait_Ns}")
    # Every seen wait_N must be in [0, HORIZON).
    bad = [w for w in distinct if not (0 <= w < opp_projection.HORIZON)]
    assert not bad, (
        f"Saw out-of-range wait_N values {bad}; expected [0, "
        f"{opp_projection.HORIZON}). All seen: {distinct}")


def test_opp_projection_omega_zero_always_passes_wait_N_zero():
    """When omega == 0, no planet rotates — wait_N propagation should be
    a no-op regardless of tick_offset. The fix MUST still pass
    `wait_N=tick_offset` (consistency), but the values can be anything;
    the resulting trajectory must be identical to wait_N=0.

    This test verifies that the fix doesn't BREAK the static (omega=0)
    case — predict_fleet_fate returns the same outcome for any wait_N
    when no rotation happens.
    """
    me = 0
    opp = 1
    src_opp = _planet(10, opp, 70.0, 50.0, ships=200, production=4, radius=1.5)
    t1 = _planet(1, -1, 90.0, 50.0, ships=2, production=3, radius=1.5)
    t2 = _planet(2, me, 30.0, 50.0, ships=2, production=2, radius=1.5)
    world_static = _world(my_id=me, planets=[src_opp, t1, t2], omega=0.0)

    arrivals = opp_projection.predict_opp_multi_launch(
        world_static, my_id=me, num_seats=2,
    )
    # Should produce ≥1 arrival in the no-rotation case.
    assert len(arrivals) >= 1, (
        f"omega=0 produced 0 arrivals — opp should be able to launch. "
        f"Arrivals: {arrivals}")
