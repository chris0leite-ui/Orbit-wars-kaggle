"""Phase ζ.v2 pin tests — hold-aware prod_stream.

The structural root cause of analytical_phase_c losing to orbitfix
(3/16 at n=16) was the LP leaf's scale-invariance in ship count: for
a successful capture, `prod_stream_me = production × (HORIZON − arrival)`
yields the SAME value for `ships=3` and `ships=11`, so under
`SHIP_COST > 0` the LP picks the smallest viable variant. Orbitfix's
heuristic chooser doesn't have this defect.

The fix: inject a synthetic opp-counter `Arrival` into per-planet
`fixed_arrivals` so the existing `_simulate_one` per-tick resolution
distinguishes subsets by their post-capture residual garrison.
Bigger fleet → bigger residual → survives opp counter → keeps
accruing prod_stream → LP naturally picks bigger fires.

Coverage (per the ACTIVE PLAN's 6-pin spec):
1. `_hold_aware_enabled()` reads env per call (default OFF).
2. `_predict_opp_counter` returns `(None, 0)` when no opp sources.
3. `_predict_opp_counter` returns `(eta, ships)` matching closest opp
   for a known scenario.
4. Default OFF: `enumerate_outcomes` returns the same prod_stream
   regardless of subset ship count (existing behavior).
5. Hold-aware ON + threat present: `prod_stream_me[big_fleet]` >
   `prod_stream_me[small_fleet]` for the same (src, tgt) capture.
6. Hold-aware ON + no threat: parity with OFF.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.intent import World
from lib.joint_solver.lp_outcome import (
    _hold_aware_enabled,
    _predict_opp_counter,
)
from lib.joint_solver.outcome_table import Arrival, enumerate_outcomes


# ---------------------------------------------------------------------------
# Test fixtures (mirror tests/test_lp_outcome.py patterns).
# ---------------------------------------------------------------------------


def _planet(pid, owner, *, ships=10, production=2, x=50.0, y=50.0, radius=1.5):
    return Planet(pid, owner, x, y, radius, ships, production)


def _world_from_planets(planets, *, fleets=None, omega=0.0):
    obs = {
        "player": 0,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": fleets or [],
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "step": 0,
    }
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# (1) Env-var gating.
# ---------------------------------------------------------------------------


def test_hold_aware_default_off(monkeypatch):
    monkeypatch.delenv("LP_HOLD_AWARE", raising=False)
    assert _hold_aware_enabled() is False


def test_hold_aware_env_on(monkeypatch):
    monkeypatch.setenv("LP_HOLD_AWARE", "1")
    assert _hold_aware_enabled() is True


# ---------------------------------------------------------------------------
# (2,3) `_predict_opp_counter`.
# ---------------------------------------------------------------------------


def test_predict_opp_counter_no_opp_sources():
    """With only my own planets + neutrals, predict returns (None, 0)."""
    me = _planet(0, 0, ships=10, x=10.0, y=10.0)
    tgt = _planet(1, -1, ships=0, x=50.0, y=50.0)
    world = _world_from_planets([me, tgt])
    eta, ships = _predict_opp_counter(1, world, my_id=0)
    assert eta is None
    assert ships == 0


def test_predict_opp_counter_returns_closest_opp(monkeypatch):
    """Two opp sources at different distances → returns the CLOSEST
    by flight time, with that source's ship count."""
    me = _planet(0, 0, ships=10, x=10.0, y=10.0)
    tgt = _planet(1, -1, ships=0, x=50.0, y=50.0)
    # opp1 far away, big garrison
    opp1 = _planet(2, 1, ships=50, x=95.0, y=95.0)
    # opp2 close, smaller garrison
    opp2 = _planet(3, 1, ships=20, x=55.0, y=55.0)
    world = _world_from_planets([me, tgt, opp1, opp2])

    eta, ships = _predict_opp_counter(1, world, my_id=0)
    # opp2 is closer (~7 units away vs ~63 units), should win.
    assert eta is not None
    assert ships == 20, f"expected opp2's 20 ships, got {ships}"


# ---------------------------------------------------------------------------
# (4) Default-OFF parity: prod_stream is scale-invariant.
# ---------------------------------------------------------------------------


def test_default_off_prod_stream_scale_invariant_in_ships():
    """The CURRENT (default-OFF) behavior: for a successful capture,
    prod_stream_me is the same for small vs large ship counts. This
    is the bug hold-aware fixes — pin the current behavior so we can
    see the change post-fix."""
    horizon = 50
    production = 3
    # Same arrival eta (10 ticks), same outcome (capture neutral),
    # different ship counts.
    cands_small = [Arrival(eta=10, owner=0, ships=3, column_id=0)]
    cands_big = [Arrival(eta=10, owner=0, ships=11, column_id=1)]

    table_small = enumerate_outcomes(
        initial_owner=-1, initial_ships=0.0, production=production,
        horizon=horizon, fixed_arrivals=[], candidate_arrivals=cands_small,
    )
    table_big = enumerate_outcomes(
        initial_owner=-1, initial_ships=0.0, production=production,
        horizon=horizon, fixed_arrivals=[], candidate_arrivals=cands_big,
    )
    prod_small = table_small[(0,)].prod_stream.get(0, 0)
    prod_big = table_big[(1,)].prod_stream.get(0, 0)

    # CURRENT BUG: identical prod_stream regardless of ship count.
    assert prod_small == prod_big, (
        f"Phase ζ.v2 baseline: prod_stream should be scale-invariant; "
        f"got small={prod_small} big={prod_big}"
    )


# ---------------------------------------------------------------------------
# (5) Hold-aware ON + threat → bigger fleet ⇒ bigger prod_stream.
#
# This is the KEY test. We simulate the fix by manually injecting the
# synthetic opp-counter arrival into fixed_arrivals (the same thing
# `_build_per_planet_arrivals` will do when hold-aware is enabled).
# Then prod_stream is no longer scale-invariant.
# ---------------------------------------------------------------------------


def test_hold_aware_bigger_fleet_holds_longer_when_threatened():
    """Bigger ship_fired → bigger post-capture residual → survives
    opp counter → keeps accruing prod_stream.

    Setup: neutral target with garrison=0, production=3. We capture
    at eta=10 with either ships=3 or ships=11. Opp counter (8 ships)
    arrives at eta=15.

    Hold-aware injects the counter as `Arrival(eta=15, owner=1, ships=8)`
    into `fixed_arrivals`. The existing _simulate_one combat math
    decides whether we hold:
      ships=3 capture: residual = 3-0 = 3; by t=15: 3 + 3·5 = 18.
                       Combat 18 vs 8 → we hold. prod_stream keeps accruing.
                       Wait that's not the right setup — both hold.
    Let me use a stronger counter (50 ships) so SMALL fleet loses
    but BIG fleet (with bigger residual) survives.
    """
    horizon = 50
    production = 3
    # Smaller fleet: residual 3 at t=10, then +3/tick. By t=15: 3+15=18.
    # Counter 50 → small loses (18 < 50).
    # Bigger fleet: residual 11 at t=10, then +3/tick. By t=15: 11+15=26.
    # Counter 50 → big still loses (26 < 50). Need to widen the gap.
    # Let me use ship_fired = 60 (residual 60), counter 50.
    cands_small = [Arrival(eta=10, owner=0, ships=11, column_id=0)]
    cands_big = [Arrival(eta=10, owner=0, ships=60, column_id=1)]
    # Counter at eta=15 with 50 ships.
    counter = Arrival(eta=15, owner=1, ships=50, column_id=None)

    table_small = enumerate_outcomes(
        initial_owner=-1, initial_ships=0.0, production=production,
        horizon=horizon, fixed_arrivals=[counter],
        candidate_arrivals=cands_small,
    )
    table_big = enumerate_outcomes(
        initial_owner=-1, initial_ships=0.0, production=production,
        horizon=horizon, fixed_arrivals=[counter],
        candidate_arrivals=cands_big,
    )
    prod_small = table_small[(0,)].prod_stream.get(0, 0)
    prod_big = table_big[(1,)].prod_stream.get(0, 0)

    assert prod_big > prod_small, (
        f"hold-aware: bigger fleet should yield bigger prod_stream "
        f"when threatened; got small={prod_small} big={prod_big}"
    )


# ---------------------------------------------------------------------------
# (6) Hold-aware ON + no threat → parity with OFF (scale-invariant).
# ---------------------------------------------------------------------------


def test_hold_aware_no_threat_is_parity():
    """When there's no opp counter (counter=None from
    `_predict_opp_counter`), `_build_per_planet_arrivals` does NOT
    inject anything → behavior identical to OFF: prod_stream still
    scale-invariant in ship count for the same capture.

    Test by NOT injecting any counter arrival. Result should be the
    same scale-invariance as the default-OFF test."""
    horizon = 50
    production = 3
    cands_small = [Arrival(eta=10, owner=0, ships=3, column_id=0)]
    cands_big = [Arrival(eta=10, owner=0, ships=11, column_id=1)]

    table_small = enumerate_outcomes(
        initial_owner=-1, initial_ships=0.0, production=production,
        horizon=horizon, fixed_arrivals=[], candidate_arrivals=cands_small,
    )
    table_big = enumerate_outcomes(
        initial_owner=-1, initial_ships=0.0, production=production,
        horizon=horizon, fixed_arrivals=[], candidate_arrivals=cands_big,
    )
    prod_small = table_small[(0,)].prod_stream.get(0, 0)
    prod_big = table_big[(1,)].prod_stream.get(0, 0)

    assert prod_small == prod_big, (
        f"hold-aware with no threat injected: should still be scale-"
        f"invariant; got small={prod_small} big={prod_big}"
    )
