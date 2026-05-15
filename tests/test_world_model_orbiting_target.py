"""Step 2 — fix orbiting-target attribution in `world_model.fleet_target_planet`.

The documented gap (`lib/world_model.py:51-55`): the static raycast
in `fleet_target_planet` assumes targets don't move. For inner
orbiting planets, this is wrong — by the time a long-range fleet
arrives, the orbital target has rotated to a different position.

Rule 38: this test FIRST reproduces the bug (static raycast says the
fleet hits an orbiting planet, but the orbit-aware
`lib.trajectory.predict_fleet_fate` says it misses), THEN asserts the
omega-aware path agrees with `predict_fleet_fate`.

Run order:
    1. Run before Step-2 fix → `test_repro_static_raycast_misattributes`
       passes (documenting the bug); `test_omega_aware_matches_truth`
       fails (function doesn't accept `omega` kwarg yet).
    2. Apply fix to `lib/world_model.py`.
    3. Run after → both pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from lib.trajectory import predict_fleet_fate
from lib.world_model import build_arrival_ledger, fleet_target_planet


# ---------------------------------------------------------------------------
# Test fixtures: hand-crafted orbital-drift scenarios
# ---------------------------------------------------------------------------


@dataclass
class _MockWorld:
    """Minimal duck-typed `World` for `predict_fleet_fate`."""

    omega: float
    planets_by_id: dict


def _orbiting_drift_scenario():
    """A scenario where the fleet's static raycast hits an orbiting
    planet at its CURRENT position, but by the time the fleet
    arrives the planet has rotated away.

    Geometry:
      - Sun at (50, 50), rotation limit < 50 from center.
      - Orbiting planet at (40, 50): orbital radius 10, well inside
        the rotation limit. Currently at angle π (relative to sun).
      - Fleet at (10, 50), aimed straight right (angle=0), with 1 ship
        → speed = 1.0 → would take ~30 turns to reach (40, 50).
      - With omega=0.05 rad/turn, after 30 turns the planet has
        rotated by 1.5 rad ≈ 86°. New planet position ≈ (49.3, 40.0)
        — well outside a 1-radius capture from the fleet's path
        (which is straight along y=50).
      - A second static planet far to the right (96, 50, radius=1)
        sits in the fleet's path past the orbiting planet's current
        x position; the fleet will eventually fly past 40 and continue
        toward 96. Whether it reaches it depends on `max_horizon`.
    """
    omega = 0.05
    orbiting_planet = Planet(
        id=7, owner=-1, x=40.0, y=50.0,
        radius=1.0, ships=10, production=1,
    )
    static_planet = Planet(
        id=8, owner=-1, x=96.0, y=50.0,
        radius=1.0, ships=10, production=1,
    )
    planets = [orbiting_planet, static_planet]
    fleet = Fleet(
        id=99, owner=0, x=10.0, y=50.0,
        angle=0.0, from_planet_id=0, ships=1,
    )
    return fleet, planets, omega, orbiting_planet, static_planet


# ---------------------------------------------------------------------------
# Rule 38 step (a) — reproduce the failure state
# ---------------------------------------------------------------------------


def test_repro_static_raycast_misattributes_orbiting_target():
    """Documents the pre-fix bug: static raycast in `fleet_target_planet`
    attributes an orbiting planet as the target, even though the orbit-
    aware `predict_fleet_fate` says the fleet misses that planet entirely.

    This test should PASS even before the Step-2 fix (it documents the
    incorrect behavior). After the fix, this test still passes because
    `omega=0.0` (the default) preserves the static raycast as the
    no-rotation fast path.
    """
    fleet, planets, omega, orbiting_planet, static_planet = _orbiting_drift_scenario()

    # Call the static-only path (no omega kwarg / omega=0.0).
    target_static, eta_static = fleet_target_planet(fleet, planets, max_horizon=120)

    # Static raycast attributes the orbiting planet as the target
    # (its CURRENT position is on the fleet's straight-line path).
    assert target_static is not None
    assert target_static.id == orbiting_planet.id
    # Distance from fleet (10, 50) to planet (40, 50) is 30 units;
    # at speed 1.0 with a 1-radius capture, eta ≈ 29.
    assert 28 <= eta_static <= 30

    # Ground truth from `predict_fleet_fate` (orbit-aware): the fleet
    # MISSES the orbiting planet because by step 30 the planet has
    # rotated away. The fleet should eventually hit the static planet
    # at (96, 50) ~85 steps later, or time out.
    world = _MockWorld(omega=omega, planets_by_id={p.id: p for p in planets})
    fake_src = Planet(
        id=999, owner=0, x=fleet.x - math.cos(fleet.angle) * 0.1,
        y=fleet.y - math.sin(fleet.angle) * 0.1,
        radius=0.0, ships=0, production=0,
    )
    fate = predict_fleet_fate(
        src=fake_src, target=orbiting_planet,
        aim_angle=fleet.angle, ships=fleet.ships,
        world=world, max_steps=120,
    )

    # The fleet does NOT hit the orbiting planet (orbital drift).
    assert not (fate.outcome == "target" and fate.hit_planet_id == orbiting_planet.id), (
        f"Expected orbital drift to make the fleet miss planet "
        f"{orbiting_planet.id}, but predict_fleet_fate said: {fate}"
    )
    # The fleet either hits the static far planet or times out / OOB.
    # Either way, the static raycast above was WRONG.


# ---------------------------------------------------------------------------
# Rule 38 step (c) — assert the omega-aware fix gives the right answer
# ---------------------------------------------------------------------------


def test_omega_aware_matches_predict_fleet_fate():
    """The Step-2 fix: with `omega` passed in, `fleet_target_planet`
    walks step-by-step using orbital projection, and agrees with
    `predict_fleet_fate` on which planet the fleet actually hits.
    """
    fleet, planets, omega, orbiting_planet, static_planet = _orbiting_drift_scenario()

    target, eta = fleet_target_planet(fleet, planets, max_horizon=120, omega=omega)

    world = _MockWorld(omega=omega, planets_by_id={p.id: p for p in planets})
    fake_src = Planet(
        id=999, owner=0,
        x=fleet.x - math.cos(fleet.angle) * 0.1,
        y=fleet.y - math.sin(fleet.angle) * 0.1,
        radius=0.0, ships=0, production=0,
    )
    fate = predict_fleet_fate(
        src=fake_src, target=orbiting_planet,
        aim_angle=fleet.angle, ships=fleet.ships,
        world=world, max_steps=120,
    )

    if fate.outcome in ("target", "planet"):
        assert target is not None
        assert target.id == fate.hit_planet_id, (
            f"omega-aware target={target.id} disagrees with truth "
            f"hit_planet_id={fate.hit_planet_id}; fate={fate}"
        )
        # ±1 step tolerance (rounding / spawn-offset differences).
        assert abs(eta - fate.step) <= 1, (
            f"eta={eta} disagrees with fate.step={fate.step}"
        )
    else:
        assert target is None, (
            f"omega-aware target={target} but predict_fleet_fate "
            f"said no planet hit (fate={fate})"
        )


def test_omega_zero_is_static_fast_path():
    """When omega=0.0 (no game rotation), the omega-aware path must
    return identical results to the static raycast — no behavior
    change for the non-rotating fast path."""
    omega = 0.0
    p1 = Planet(id=1, owner=-1, x=30.0, y=50.0, radius=1.0, ships=10, production=1)
    p2 = Planet(id=2, owner=-1, x=80.0, y=50.0, radius=1.0, ships=10, production=1)
    planets = [p1, p2]
    fleet = Fleet(id=99, owner=0, x=10.0, y=50.0, angle=0.0, from_planet_id=0, ships=1)

    target_default, eta_default = fleet_target_planet(fleet, planets, max_horizon=100)
    target_explicit, eta_explicit = fleet_target_planet(
        fleet, planets, max_horizon=100, omega=0.0,
    )

    assert target_default is not None and target_explicit is not None
    assert target_default.id == target_explicit.id == p1.id
    assert eta_default == eta_explicit


# ---------------------------------------------------------------------------
# Hypothesis fuzz: random orbiting-target scenarios
# ---------------------------------------------------------------------------


hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    derandomize=True,
)
@given(
    src_x=st.floats(min_value=5.0, max_value=95.0),
    src_y=st.floats(min_value=5.0, max_value=95.0),
    target_initial_angle=st.floats(min_value=0.0, max_value=2 * math.pi),
    target_orbit_radius=st.floats(min_value=12.0, max_value=35.0),
    target_radius=st.floats(min_value=1.0, max_value=2.5),
    fleet_angle=st.floats(min_value=0.0, max_value=2 * math.pi),
    ships=st.integers(min_value=1, max_value=200),
    omega=st.sampled_from([0.03, 0.05, 0.08]),
)
def test_omega_aware_fleet_target_agrees_with_predict_fleet_fate_fuzz(
    src_x, src_y, target_initial_angle, target_orbit_radius,
    target_radius, fleet_angle, ships, omega,
):
    """Random orbiting-target scenarios: omega-aware fleet_target_planet
    must agree with predict_fleet_fate on (target_id, step) to within
    ±1 step.

    Excludes scenarios where the fleet's spawn position is inside a
    planet's radius (degenerate)."""
    # Place one orbiting planet and a couple of static far planets.
    cx = 50.0 + target_orbit_radius * math.cos(target_initial_angle)
    cy = 50.0 + target_orbit_radius * math.sin(target_initial_angle)

    # Skip degenerate: source too close to the orbiting planet.
    if math.hypot(src_x - cx, src_y - cy) < target_radius + 1.0:
        pytest.skip("degenerate: source overlaps orbiting planet")

    orbiting = Planet(
        id=1, owner=-1, x=cx, y=cy,
        radius=target_radius, ships=10, production=1,
    )
    # Two far static planets to give the fleet alternative things to hit.
    static_far1 = Planet(
        id=2, owner=-1, x=80.0, y=80.0,
        radius=1.5, ships=10, production=1,
    )
    static_far2 = Planet(
        id=3, owner=-1, x=20.0, y=20.0,
        radius=1.5, ships=10, production=1,
    )
    planets = [orbiting, static_far1, static_far2]
    fleet = Fleet(
        id=99, owner=0, x=src_x, y=src_y,
        angle=fleet_angle, from_planet_id=0, ships=ships,
    )

    target, eta = fleet_target_planet(fleet, planets, max_horizon=150, omega=omega)

    world = _MockWorld(omega=omega, planets_by_id={p.id: p for p in planets})
    fake_src = Planet(
        id=999, owner=0,
        x=src_x - math.cos(fleet_angle) * 0.1,
        y=src_y - math.sin(fleet_angle) * 0.1,
        radius=0.0, ships=0, production=0,
    )
    fate = predict_fleet_fate(
        src=fake_src, target=orbiting,
        aim_angle=fleet_angle, ships=ships,
        world=world, max_steps=150,
    )

    if fate.outcome in ("target", "planet"):
        assert target is not None, (
            f"truth says hit planet {fate.hit_planet_id} at step {fate.step}, "
            f"but fleet_target_planet returned None"
        )
        assert target.id == fate.hit_planet_id, (
            f"target.id={target.id} vs truth.hit_planet_id={fate.hit_planet_id}"
        )
        assert abs(eta - fate.step) <= 1, (
            f"eta={eta} vs truth.step={fate.step}"
        )
    else:
        # Fleet doesn't hit any planet (sun / oob / timeout).
        assert target is None, (
            f"truth says outcome={fate.outcome} (no planet hit), "
            f"but fleet_target_planet returned target.id={target.id}"
        )


# ---------------------------------------------------------------------------
# build_arrival_ledger forwards omega
# ---------------------------------------------------------------------------


def test_build_arrival_ledger_uses_omega():
    """The ledger builder must thread omega through to
    fleet_target_planet — otherwise WorldModel timelines would still
    use the static raycast for orbiting targets."""
    fleet, planets, omega, orbiting_planet, _ = _orbiting_drift_scenario()

    ledger_static = build_arrival_ledger([fleet], planets, horizon=120)
    ledger_orbital = build_arrival_ledger([fleet], planets, horizon=120, omega=omega)

    # Static path: fleet attributed to orbiting planet.
    static_arrivals = ledger_static[orbiting_planet.id]
    assert len(static_arrivals) == 1, f"expected 1 arrival, got {static_arrivals}"

    # Orbital path: fleet NOT attributed to orbiting planet (it misses).
    orbital_arrivals = ledger_orbital[orbiting_planet.id]
    assert orbital_arrivals == [], (
        f"orbital ledger should not attribute the missed orbiting "
        f"planet, got {orbital_arrivals}"
    )
