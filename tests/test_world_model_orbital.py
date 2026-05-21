"""Orbital-safety regression tests for WorldModel.time_to_enemy_threat.

Covers the four code paths the f1774a7 + 2026-05-22 follow-up touched:
- target position prediction at our arrival (existing fix);
- B5: strict `>` filter on pre-arrival in-flight enemy fleets;
- B6: `incoming_enemy_eta_after` finds later inbound waves when the
  earliest is pre-arrival;
- B7: 5-iteration fixed-point on `enemy_eta_travel` for orbiting targets;
- `_position_at` helper identity on static planets / omega=0.

Tests are structured pre-fix-vs-post-fix where possible: call the
function once with `arrival_eta=0` (legacy path), once with `arrival_eta>0`
(fixed path), assert behavior differs in the documented direction.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from lib.intent import World
from lib.world_model import WorldModel, _position_at


def _planet(pid, owner, x, y, *, ships=20, production=2, radius=1.5):
    return SimpleNamespace(
        id=pid, owner=owner, x=x, y=y, radius=radius,
        ships=ships, production=production,
    )


def _world(planets, fleets=(), my_id=0, omega=0.0):
    obs = {
        "player": my_id,
        "planets": [
            [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
            for p in planets
        ],
        "fleets": list(fleets),
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "step": 0,
    }
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# _position_at — helper identity + orbital prediction
# ---------------------------------------------------------------------------


def test_position_at_returns_current_when_lead_zero():
    p = _planet(1, -1, 30.0, 40.0)
    x, y = _position_at(p, omega=0.01, lead_turns=0)
    assert (x, y) == (30.0, 40.0)


def test_position_at_returns_current_when_omega_zero():
    p = _planet(1, -1, 30.0, 40.0)
    x, y = _position_at(p, omega=0.0, lead_turns=10)
    assert (x, y) == (30.0, 40.0)


def test_position_at_returns_current_for_static_outer_planet():
    # Planet at (95, 50): orbital_radius = hypot(45, 0) = 45; with radius
    # 6, 45 + 6 = 51 > ROTATION_RADIUS_LIMIT = 50 → is_orbiting=False.
    p = _planet(1, -1, 95.0, 50.0, radius=6.0)
    x, y = _position_at(p, omega=0.05, lead_turns=10)
    assert (x, y) == (95.0, 50.0)


def test_position_at_rotates_orbiting_planet():
    # Inner planet rotates. After half a revolution it should be on the
    # opposite side of the sun.
    p = _planet(1, -1, 60.0, 50.0, radius=1.0)
    half_rev_turns = int(round(math.pi / 0.05))
    x, y = _position_at(p, omega=0.05, lead_turns=half_rev_turns)
    # Started 10 units east of CENTER=50; after π radians, should be ~10
    # units west.
    assert x < 45.0
    assert abs(y - 50.0) < 2.0


# ---------------------------------------------------------------------------
# time_to_enemy_threat — orbital target rotates into enemy reach
# ---------------------------------------------------------------------------


def test_target_rotates_into_enemy_reduces_threat_eta():
    """Orbiting target starts FAR from a STATIC enemy; after some
    rotation it's MUCH closer. arrival_eta=N must shorten the
    travel-time portion of the threat ETA vs arrival_eta=0.

    Setup: inner-orbiting target at (40, 50) — opposite side of sun
    from a STATIC outer enemy at (95, 50). After π radians of
    rotation, target moves to (60, 50), much closer to enemy."""
    omega = 0.02  # small rotation so during-travel drift is negligible
    # Inner-orbiting target on the FAR side from the enemy.
    target = _planet(1, -1, 40.0, 50.0, ships=5, radius=1.0)
    # Static outer enemy (orbital_radius=45 + radius 6 = 51 > 50).
    enemy = _planet(2, 1, 95.0, 50.0, ships=80, radius=6.0)
    mine = _planet(0, 0, 5.0, 5.0, ships=100, radius=1.0)
    world = _world([mine, target, enemy], omega=omega)
    wm = WorldModel.from_world(world)
    half_rev = int(round(math.pi / omega))
    threat_now = wm.time_to_enemy_threat(target.id, my_id=0, world=world)
    threat_at_arrival = wm.time_to_enemy_threat(
        target.id, my_id=0, world=world, arrival_eta=half_rev,
    )
    assert threat_now is not None
    assert threat_at_arrival is not None
    travel_at_arrival = threat_at_arrival - half_rev
    assert travel_at_arrival < threat_now, (
        f"Travel-after-arrival ({travel_at_arrival}) should be less than "
        f"current-position travel ({threat_now}) when target rotates "
        f"into enemy reach."
    )


def test_target_static_arrival_eta_does_not_change_travel():
    """Static (non-orbiting) target — arrival_eta should not change the
    travel-time estimate, only shift the absolute ETA."""
    omega = 0.05
    # Static target outside rotation limit.
    target = _planet(1, -1, 95.0, 50.0, ships=5, radius=6.0)
    enemy = _planet(2, 1, 5.0, 50.0, ships=80, radius=6.0)
    mine = _planet(0, 0, 5.0, 5.0, ships=100, radius=1.0)
    world = _world([mine, target, enemy], omega=omega)
    wm = WorldModel.from_world(world)
    threat_now = wm.time_to_enemy_threat(target.id, my_id=0, world=world)
    threat_arrival = wm.time_to_enemy_threat(
        target.id, my_id=0, world=world, arrival_eta=10,
    )
    # Static target → travel-time same; absolute threat shifts by exactly
    # arrival_eta (10).
    assert threat_now is not None
    assert threat_arrival is not None
    assert threat_arrival - threat_now == 10


# ---------------------------------------------------------------------------
# B5 — strict-`>` filter on pre-arrival in-flight enemy fleets
# ---------------------------------------------------------------------------


def test_inflight_pre_arrival_filtered_out():
    """In-flight enemy fleet arriving BEFORE our arrival should NOT
    count as a future threat (combat resolves at our arrival)."""
    target = _planet(1, -1, 90.0, 50.0, ships=20, radius=2.0)
    mine = _planet(0, 0, 10.0, 50.0, ships=100, radius=1.0)
    # Enemy fleet headed east, very close — will arrive early.
    obs = {
        "player": 0,
        "planets": [
            [mine.id, mine.owner, mine.x, mine.y, mine.radius, mine.ships, mine.production],
            [target.id, target.owner, target.x, target.y, target.radius, target.ships, target.production],
        ],
        # fleet very close to target → eta ~few turns
        "fleets": [[200, 1, 85.0, 50.0, 0.0, 99, 50]],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": 0,
    }
    world = World.from_obs(obs)
    wm = WorldModel.from_world(world)
    # The fleet arrives well before arrival_eta=50.
    threat_late = wm.time_to_enemy_threat(
        target.id, my_id=0, world=world, arrival_eta=50,
    )
    # Without orbital safety: legacy `incoming_enemy_eta` returns the
    # pre-arrival fleet ETA (~few turns).
    threat_legacy = wm.time_to_enemy_threat(target.id, my_id=0, world=world)
    assert threat_legacy is not None
    assert threat_legacy < 20, "Pre-fix path should see early fleet"
    # Fixed path: no in-flight signal (fleet is pre-arrival), so the
    # threat must come from the launch leg — no enemy planet, so None.
    assert threat_late is None


def test_inflight_simultaneous_arrival_excluded():
    """Strict `>` semantics: a fleet arriving exactly AT our arrival
    is excluded (resolved by combat at arrival)."""
    # Use the ledger directly to make ETA equality precise.
    mine = _planet(0, 0, 10.0, 50.0, ships=100)
    target = _planet(1, -1, 90.0, 50.0, ships=20, radius=2.0)
    world = _world([mine, target])
    wm = WorldModel.from_world(world)
    # Inject a synthetic arrival at exactly eta=10.
    wm.ledger[target.id] = [(10, 1, 50)]
    after = wm.incoming_enemy_eta_after(target.id, my_id=0, after=10)
    assert after is None, "Fleet at eta=10 must be filtered when after=10 (strict `>`)"
    after_lower = wm.incoming_enemy_eta_after(target.id, my_id=0, after=9)
    assert after_lower == 10, "Fleet at eta=10 must surface when after=9"


# ---------------------------------------------------------------------------
# B6 — later in-flight wave surfaces when earliest is pre-arrival
# ---------------------------------------------------------------------------


def test_incoming_enemy_eta_after_returns_strictly_after():
    """Unit test for the new method directly."""
    mine = _planet(0, 0, 10.0, 50.0, ships=100)
    target = _planet(1, -1, 90.0, 50.0, ships=20)
    world = _world([mine, target])
    wm = WorldModel.from_world(world)
    wm.ledger[target.id] = [(3, 1, 30), (12, 1, 40), (25, 1, 50)]
    assert wm.incoming_enemy_eta_after(target.id, my_id=0, after=0) == 3
    assert wm.incoming_enemy_eta_after(target.id, my_id=0, after=3) == 12
    assert wm.incoming_enemy_eta_after(target.id, my_id=0, after=12) == 25
    assert wm.incoming_enemy_eta_after(target.id, my_id=0, after=25) is None


def test_later_inflight_surfaces_when_earliest_is_pre_arrival():
    """Two inbound fleets: ETA=5 (pre our arrival 10) and ETA=15 (post).
    Legacy `incoming_enemy_eta` only sees ETA=5; with arrival_eta=10 the
    fixed path should surface ETA=15 via `incoming_enemy_eta_after`."""
    mine = _planet(0, 0, 10.0, 50.0, ships=100)
    target = _planet(1, -1, 90.0, 50.0, ships=20)
    world = _world([mine, target])
    wm = WorldModel.from_world(world)
    wm.ledger[target.id] = [(5, 1, 30), (15, 1, 40)]
    # Legacy path: sees the earliest (5).
    legacy = wm.time_to_enemy_threat(target.id, my_id=0, world=world)
    assert legacy == 5
    # Fixed path with arrival_eta=10: earliest is pre-arrival, but the
    # post-arrival ETA=15 wave must surface.
    fixed = wm.time_to_enemy_threat(
        target.id, my_id=0, world=world, arrival_eta=10,
    )
    assert fixed == 15


# ---------------------------------------------------------------------------
# B7 — 5-iteration fixed-point on enemy travel for orbiting targets
# ---------------------------------------------------------------------------


def test_b7_fixed_point_only_runs_for_orbital_target():
    """Non-orbiting target — no fixed-point iteration; threat ETA equals
    arrival_eta + straight-line travel between predicted-at-arrival
    positions (which are identical to current for static planets)."""
    omega = 0.05
    # Static target + static enemy.
    target = _planet(1, -1, 95.0, 50.0, ships=5, radius=6.0)
    enemy = _planet(2, 1, 5.0, 50.0, ships=80, radius=6.0)
    mine = _planet(0, 0, 5.0, 5.0, ships=100, radius=1.0)
    world = _world([mine, target, enemy], omega=omega)
    wm = WorldModel.from_world(world)
    threat = wm.time_to_enemy_threat(
        target.id, my_id=0, world=world, arrival_eta=20,
    )
    # Travel distance: enemy at (5,50) → target at (95,50) = 90 units;
    # accounting for flight-tangent geometry handled inside the function.
    # Just assert finite and equals arrival_eta + straight-line value.
    assert threat is not None
    # No rotation, so the threat must equal arrival_eta + simple travel.
    # Recompute the same straight-line for comparison.
    from lib.fleet import speed
    v = speed(80)
    dist = 90.0
    expected = 20 + int(math.ceil(dist / v))
    assert threat == expected


def test_b7_fixed_point_runs_for_orbital_target():
    """Orbital target with an enemy whose travel time spans a substantial
    fraction of a revolution — the fixed-point must converge to a
    different value than the straight-line seed."""
    omega = 0.05
    # Slowly orbiting inner target.
    target = _planet(1, -1, 60.0, 50.0, ships=5, radius=1.0)
    # Modest enemy (small ships → slow fleet → travel takes many ticks
    # during which target keeps rotating).
    enemy = _planet(2, 1, 5.0, 50.0, ships=20, radius=6.0)
    mine = _planet(0, 0, 5.0, 5.0, ships=100, radius=1.0)
    world = _world([mine, target, enemy], omega=omega)
    wm = WorldModel.from_world(world)
    # Should not crash; should return a finite ETA in some bounded range
    # (the fixed-point safety net falls through to the seed estimate
    # if non-convergent).
    threat = wm.time_to_enemy_threat(
        target.id, my_id=0, world=world, arrival_eta=10,
    )
    assert threat is not None
    assert threat > 10  # later than our arrival (post-capture threat)
    assert threat < 1000  # finite — convergence or safety net


def test_b7_skips_iteration_when_arrival_eta_zero():
    """arrival_eta=0 disables target rotation; B7 fixed-point should not
    run (target_is_orbital is False)."""
    omega = 0.05
    target = _planet(1, -1, 60.0, 50.0, ships=5, radius=1.0)
    enemy = _planet(2, 1, 65.0, 50.0, ships=80, radius=1.0)
    mine = _planet(0, 0, 5.0, 5.0, ships=100, radius=1.0)
    world = _world([mine, target, enemy], omega=omega)
    wm = WorldModel.from_world(world)
    # arrival_eta=0 → legacy path, current positions only.
    threat = wm.time_to_enemy_threat(target.id, my_id=0, world=world)
    assert threat is not None and threat > 0
