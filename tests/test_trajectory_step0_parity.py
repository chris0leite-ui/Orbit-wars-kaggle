"""Env-parity regression tests for `lib.trajectory.predict_fleet_fate`
at obs.step == 0.

The env's interpreter rotates planets by `omega * obs.step`, so at
obs.step == 0 the first tick keeps planets stationary (rotation
count is 0). At subsequent ticks (obs.step >= 1) the agent's current
planet position already reflects the prior ticks' rotation.

`predict_fleet_fate` originally applied `omega * t` rotation to
every relative tick — correct for obs.step >= 1, off by one rotation
step for obs.step == 0. These tests pin the fix that aligns the lib
with the env at game start.

Ground truth: each scenario was independently simulated by feeding
the same state into `kaggle_environments.envs.orbit_wars.interpreter`
and observing the env's behaviour. See the diagnosis transcript in
the conversation that produced this file (2026-05-17).
"""

from __future__ import annotations

import math

from lib.intent import World
from lib.world_model import WorldModel


def _obs(*, step: int, omega: float, planets, fleets):
    return {
        "player": 0,
        "planets": planets,
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "step": step,
        "comets": [],
        "fleets": list(fleets),
    }


# ---------------------------------------------------------------------------
# obs.step == 0 + orbital planet: lib must match the env's stationary
# first tick + rotated subsequent ticks.
# ---------------------------------------------------------------------------


def test_step_zero_orbital_capture_matches_env():
    """Friendly fleet at (65,10) heading east toward orbital enemy
    planet at (70,10) (orbital_radius ≈ 44.7, rotates with omega=0.05).
    At obs.step=0 the env keeps the planet stationary on tick 1 then
    rotates from tick 2 onwards, giving the fleet enough time to hit
    on tick 2.

    Ground truth (kaggle_environments interpreter run): planet 1
    flips to owner=0 at tick 2. The lib's WorldModel must reflect
    this in `ledger[1]` and `owner_at`.
    """
    obs = _obs(
        step=0,
        omega=0.05,
        planets=[
            [0, 0, 10.0, 10.0, 1.0, 50, 1],
            [1, 1, 70.0, 10.0, 1.0, 5, 1],
        ],
        fleets=[[0, 0, 65.0, 10.0, 0.0, 2, 100]],
    )
    w = World.from_obs(obs)
    m = WorldModel.from_world(w)
    arrivals = m.ledger.get(1)
    assert arrivals, f"expected ledger to record the capture; got {arrivals}"
    eta, owner, ships = arrivals[0]
    assert eta == 2 and owner == 0 and ships == 100
    # Owner-at queries at and beyond the capture turn must say ours.
    assert m.owner_at(1, 2) == 0
    assert m.owner_at(1, 21) == 0


def test_step_zero_omega_zero_unchanged():
    """Regression guard: with omega=0 the orbital code path doesn't
    run (static raycast handles it). The parity shift must not affect
    this case."""
    obs = _obs(
        step=0,
        omega=0.0,
        planets=[
            [0, 0, 10.0, 10.0, 1.0, 50, 1],
            [1, 1, 70.0, 10.0, 1.0, 5, 1],
        ],
        fleets=[[0, 0, 65.0, 10.0, 0.0, 2, 100]],
    )
    w = World.from_obs(obs)
    m = WorldModel.from_world(w)
    arrivals = m.ledger.get(1)
    assert arrivals, f"expected ledger to record the capture; got {arrivals}"
    eta, owner, ships = arrivals[0]
    assert eta == 2 and owner == 0 and ships == 100


# ---------------------------------------------------------------------------
# obs.step >= 1 path is untouched (the lib was already parity-correct
# for non-zero steps; the fix's `rot_offset` is 0 there).
# ---------------------------------------------------------------------------


def test_step_nonzero_orbital_unchanged_no_capture():
    """At obs.step=5 with the same artificial state (initial_planets =
    current via the World.from_obs default), the planet's orbital
    rotation outpaces the fleet's eastward motion — no capture is
    expected by either the env or the lib. Pinning this confirms the
    fix didn't accidentally fire the shift on obs.step >= 1.
    """
    obs = _obs(
        step=5,
        omega=0.05,
        planets=[
            [0, 0, 10.0, 10.0, 1.0, 50, 1],
            [1, 1, 70.0, 10.0, 1.0, 5, 1],
        ],
        fleets=[[0, 0, 65.0, 10.0, 0.0, 2, 100]],
    )
    w = World.from_obs(obs)
    m = WorldModel.from_world(w)
    # The lib should NOT attribute the fleet to planet 1 here — the
    # planet rotates away faster than the fleet catches it.
    assert m.ledger.get(1) == [], (
        f"expected empty ledger at obs.step=5; got {m.ledger.get(1)}"
    )


# ---------------------------------------------------------------------------
# Static (non-orbital) target on the step=0 path: the shift only
# applies to orbiting planets, so this should be unchanged.
# ---------------------------------------------------------------------------


def test_step_zero_static_target_unchanged():
    """Non-orbital target (orbital_radius + r >= 50 → static). The
    static raycast handles this regardless of obs.step; the parity
    shift must not perturb the result."""
    # Planet at (95, 95): orbital_radius = sqrt(45**2 + 45**2) ≈ 63.6,
    # well above ROTATION_RADIUS_LIMIT = 50 → static.
    obs = _obs(
        step=0,
        omega=0.05,
        planets=[
            [0, 0, 90.0, 95.0, 1.0, 50, 1],
            [1, 1, 95.0, 95.0, 1.0, 5, 1],
        ],
        fleets=[[0, 0, 92.0, 95.0, 0.0, 2, 100]],
    )
    w = World.from_obs(obs)
    m = WorldModel.from_world(w)
    arrivals = m.ledger.get(1)
    assert arrivals, f"expected ledger to record the capture; got {arrivals}"
    _eta, owner, ships = arrivals[0]
    assert owner == 0 and ships == 100
