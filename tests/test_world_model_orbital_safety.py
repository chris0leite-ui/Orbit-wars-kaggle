"""Rule-38 pin tests for `WorldModel.time_to_enemy_threat` orbital
arrival safety (PI 2026-05-21 / commit f1774a7).

Bug: the function computed enemy-threat distances from the target's
CURRENT (x, y). For ORBITING targets that rotate INTO enemy territory
by our arrival tick, the threat ETA was falsely long → expected_hold
was falsely long → captures we'd immediately lose were scored as safe.

Fix: added `arrival_eta` parameter; when > 0 and `omega != 0`, predicts
target AND enemy positions at our arrival via `lib.orbit.predict_relative`.
Default `arrival_eta=0` preserves "current position" semantics for
source-safety callers.

Rule 38 cycle for these tests: temporarily set arrival_eta=0 (pre-fix
behaviour) — fixture 1 should fail (long ETA) and fixture 2 should
pass (no behaviour diff on non-rotating boards).
"""

from __future__ import annotations

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.geometry import CENTER
from lib.intent import World
from lib.world_model import WorldModel


def _world(my_id, planets, *, omega=0.04, step=0, fleets=None):
    """Build a World via `World.from_obs` to match the production code path."""
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": fleets or [],
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "step": step,
    }
    return World.from_obs(obs)


def _model(planet_ids):
    """Minimal WorldModel: empty ledger per planet, empty timelines."""
    return WorldModel(
        ledger={pid: [] for pid in planet_ids},
        timelines={},
        horizon=200,
    )


def _polar_planet(pid, owner, *, orb_r, theta, ships=10, production=2,
                  radius=1.0):
    """Place an orbiting planet at polar (orb_r, theta) around CENTER.

    `theta` in radians. Must satisfy `orb_r + radius < ROTATION_RADIUS_LIMIT`
    (50.0) so the planet qualifies as orbiting per `lib.orbit.is_orbiting`.
    """
    x = CENTER + orb_r * math.cos(theta)
    y = CENTER + orb_r * math.sin(theta)
    return Planet(pid, owner, x, y, radius, ships, production)


# ---------------------------------------------------------------------------
# Fixture 1 — orbital drift: arrival_eta shifts distances on a rotating board.
#
# Semantic: `time_to_enemy_threat(arrival_eta=k)` returns the tick FROM NOW
# at which the enemy can reach the target, assuming the enemy launches upon
# our capture (tick k). So the baseline offset is `arrival_eta + travel_time`.
#
# On a ROTATING board, when the target rotates relative to the enemy, the
# travel distance — and therefore travel_time — differs from the static case.
# So `eta_future - arrival_eta != eta_now` (i.e., the TRAVEL component
# differs between the rotated and static views).
#
# Setup: P0 mine, orbiting at theta=0 (east of center). P1 opp, static near
# P0's CURRENT position. Under omega=0.04, by tick 20 P0 rotates counter-
# clockwise away from P1, increasing the distance.
# ---------------------------------------------------------------------------

def test_time_to_enemy_threat_arrival_eta_shifts_with_rotation():
    """Orbiting target + arrival_eta > 0 must yield a different travel time
    than arrival_eta=0 on a rotating board.

    Invariant tested: `(eta_future - arrival_eta) != eta_now` — the travel
    component (distance / speed) differs because the predicted target
    position differs from the current position.

    Setup requires one ORBITING + one STATIC planet — if both orbit at the
    same omega, relative distances are preserved by symmetry and the bug
    is invisible. STATIC means `orb_r + radius >= ROTATION_RADIUS_LIMIT=50`.
    """
    # P0 (mine, orbiting at theta=0, orb_r=20).
    p0 = _polar_planet(0, 0, orb_r=20.0, theta=0.0)
    # P1 (opp, STATIC — orb_r=49, radius=2 → orb_r+radius=51 ≥ 50, so
    # is_orbiting returns False; placed near P0's CURRENT position so
    # static-view distance is short).
    p1 = Planet(1, 1, CENTER + 49.0, CENTER, 2.0, 50, 2)
    arrival_eta = 20

    world = _world(my_id=0, planets=[p0, p1], omega=0.04)
    model = _model([0, 1])

    eta_now = model.time_to_enemy_threat(0, 0, world, arrival_eta=0)
    eta_future = model.time_to_enemy_threat(0, 0, world, arrival_eta=arrival_eta)

    assert eta_now is not None, "fixture: opp should be reachable now"
    assert eta_future is not None, "fixture: opp should be reachable at +20"
    travel_future = eta_future - arrival_eta
    travel_now = eta_now
    assert travel_future != travel_now, (
        f"arrival_eta did not shift the travel-time component — "
        f"orbital prediction didn't apply. "
        f"travel_now={travel_now}, travel_future={travel_future} "
        f"(eta_future={eta_future}, arrival_eta={arrival_eta})."
    )


# ---------------------------------------------------------------------------
# Fixture 2 — non-rotating board: travel time is invariant of arrival_eta.
#
# omega=0.0 means positions never shift. The travel-time component
# (eta_future - arrival_eta) must equal eta_now exactly. The total threat
# ETA differs by the baseline offset (arrival_eta), but the travel time
# stays the same.
# ---------------------------------------------------------------------------

def test_time_to_enemy_threat_static_board_travel_invariant():
    """On a non-rotating board, the travel-time component must be invariant
    of `arrival_eta`. Locks the "no orbital prediction when not needed"
    contract."""
    p0 = _polar_planet(0, 0, orb_r=20.0, theta=0.0)
    p1 = Planet(1, 1, CENTER + 49.0, CENTER, 2.0, 50, 2)
    arrival_eta = 20

    world = _world(my_id=0, planets=[p0, p1], omega=0.0)
    model = _model([0, 1])

    eta_now = model.time_to_enemy_threat(0, 0, world, arrival_eta=0)
    eta_future = model.time_to_enemy_threat(0, 0, world,
                                            arrival_eta=arrival_eta)

    assert eta_now is not None and eta_future is not None
    travel_future = eta_future - arrival_eta
    assert travel_future == eta_now, (
        f"travel time changed on a non-rotating board (omega=0); expected "
        f"travel_future == eta_now. travel_now={eta_now}, "
        f"travel_future={travel_future}"
    )


# ---------------------------------------------------------------------------
# Fixture 3 — arrival_eta=0 default preserves backwards-compat.
# ---------------------------------------------------------------------------

def test_time_to_enemy_threat_default_arrival_eta_zero():
    """Calling without `arrival_eta` (the default `0`) must equal the
    legacy "current position" call. Pin guards against accidentally
    changing the default."""
    p0 = _polar_planet(0, 0, orb_r=20.0, theta=0.0)
    p1 = Planet(1, 1, CENTER + 49.0, CENTER, 2.0, 50, 2)
    world = _world(my_id=0, planets=[p0, p1], omega=0.04)
    model = _model([0, 1])

    eta_default = model.time_to_enemy_threat(0, 0, world)
    eta_explicit = model.time_to_enemy_threat(0, 0, world, arrival_eta=0)

    assert eta_default == eta_explicit, (
        "default arrival_eta should equal arrival_eta=0; default changed."
    )
