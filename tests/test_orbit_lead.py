"""Tests for lib/orbit_lead.py — orbital phase-lead targeting primitives.

The load-bearing test is `test_closest_approach_matches_env_seed42`: we
predict the closest-approach turn from a step-0 observation, step the
env forward to that exact turn, and verify the actual distance matches
our prediction.
"""

from __future__ import annotations

import math

import pytest

from kaggle_environments import make

from lib import geometry as G
from lib import orbit
from lib import orbit_lead

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _planet(pid: int, x: float, y: float, radius: float = 1.0, prod: int = 1, owner: int = -1):
    return [pid, owner, x, y, radius, 10, prod]


# ---------------------------------------------------------------------------
# Pure-formula tests — no env runs
# ---------------------------------------------------------------------------


def test_position_at_static_planet_does_not_move():
    p = _planet(0, x=G.CENTER + 49.0, y=G.CENTER, radius=2.0)  # 49 + 2 = 51 ≥ 50 → static
    x, y = orbit_lead.position_at(p, omega=0.04, lead_turns=10)
    assert x == pytest.approx(p[2])
    assert y == pytest.approx(p[3])


def test_position_at_orbiting_planet_rotates():
    # Orbiting planet at radius 30, angle 0 → after omega*t = pi/2 the
    # planet should be at (CENTER, CENTER + 30).
    p = _planet(0, x=G.CENTER + 30.0, y=G.CENTER, radius=1.0)
    x, y = orbit_lead.position_at(p, omega=math.pi / 2, lead_turns=1)
    assert x == pytest.approx(G.CENTER, abs=1e-9)
    assert y == pytest.approx(G.CENTER + 30.0, abs=1e-9)


def test_closest_approach_static_pair_is_at_t0():
    """Static-static pair: distance is constant, argmin = 0."""
    a = _planet(0, x=G.CENTER + 49.0, y=G.CENTER, radius=2.0)
    b = _planet(1, x=G.CENTER - 49.0, y=G.CENTER, radius=2.0)
    t_star, d_min = orbit_lead.closest_approach(a, b, omega=0.04, horizon=100)
    assert t_star == 0
    assert d_min == pytest.approx(98.0)


def test_closest_approach_static_source_orbiting_target():
    """Static source at (50, 90), orbiting target at radius 30 starting
    at angle 0 (so target_pos(0) = (80, 50)). Target sweeps in 100 turns
    by omega*100 = 0.04*100 = 4 rad ≈ 1.27 revolutions. The minimum
    distance from (50, 90) to ring-of-radius-30 is |90-50| - 30 = 10,
    achieved at angle pi/2 → omega*t = pi/2 → t = pi/(2*0.04) ≈ 39.27.
    """
    src = _planet(0, x=G.CENTER, y=G.CENTER + 40.0, radius=2.0)  # 40 + 2 = 42 < 50 → orbits
    # We need src STATIC; bump radius so total > 50.
    src = _planet(0, x=G.CENTER, y=G.CENTER + 49.0, radius=2.0)  # 49 + 2 = 51 → static
    tgt = _planet(1, x=G.CENTER + 30.0, y=G.CENTER, radius=1.0)  # 30 + 1 = 31 < 50 → orbits

    t_star, d_min = orbit_lead.closest_approach(src, tgt, omega=0.04, horizon=100)
    # Predicted t* ≈ 39 (integer scan).
    assert abs(t_star - 39) <= 1
    # Min distance ≈ |99 - 50| - 30 = 19 (src at y=99 not 90 because CENTER=50).
    expected_min = (49.0 + 0.0) - 30.0  # 19
    assert d_min == pytest.approx(expected_min, abs=0.5)


def test_closest_approach_zero_omega_returns_t0():
    """With omega=0 the orbiting planet is effectively static; closest
    approach is at t=0."""
    src = _planet(0, x=G.CENTER, y=G.CENTER + 49.0, radius=2.0)
    tgt = _planet(1, x=G.CENTER + 30.0, y=G.CENTER, radius=1.0)
    t_star, d_min = orbit_lead.closest_approach(src, tgt, omega=0.0, horizon=100)
    assert t_star == 0
    expected = math.hypot(G.CENTER + 30.0 - G.CENTER, G.CENTER - (G.CENTER + 49.0))
    assert d_min == pytest.approx(expected)


def test_best_launch_plan_static_source_orbiting_target_beats_naive():
    """The phase-lead plan should produce a SHORTER travel distance than
    the fire-now naive plan when the target is moving toward the source."""
    src = _planet(0, x=G.CENTER, y=G.CENTER + 49.0, radius=2.0)  # static
    tgt = _planet(1, x=G.CENTER + 30.0, y=G.CENTER, radius=1.0)  # orbits
    ships = 30

    naive = orbit_lead.naive_launch_plan(src, tgt, omega=0.04, ships=ships)
    best = orbit_lead.best_launch_plan(src, tgt, omega=0.04, ships=ships, horizon=120)
    assert best is not None
    assert best.distance < naive.distance, (
        f"phase-lead distance {best.distance:.2f} should beat naive {naive.distance:.2f}"
    )


def test_best_launch_plan_returns_none_when_unreachable():
    """A huge fleet that even at max speed cannot cover the distance in
    the horizon should yield None. We force this with a tiny horizon."""
    src = _planet(0, x=G.CENTER + 49.0, y=G.CENTER, radius=2.0)
    tgt = _planet(1, x=G.CENTER - 49.0, y=G.CENTER, radius=2.0)
    # 98 units; speed at 10 ships ~ 1.6; ETA ~ 62 turns. Horizon 30 → unreachable.
    plan = orbit_lead.best_launch_plan(src, tgt, omega=0.04, ships=10, horizon=30)
    assert plan is None


def test_best_launch_plan_launch_offset_non_negative():
    src = _planet(0, x=G.CENTER, y=G.CENTER + 49.0, radius=2.0)
    tgt = _planet(1, x=G.CENTER + 30.0, y=G.CENTER, radius=1.0)
    plan = orbit_lead.best_launch_plan(src, tgt, omega=0.04, ships=30, horizon=120)
    assert plan is not None
    assert plan.launch_offset >= 0
    assert plan.eta >= 1
    assert plan.arrival_turn == plan.launch_offset + plan.eta


# ---------------------------------------------------------------------------
# Live-env integration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env_seed_42():
    env = make("orbit_wars", configuration={"seed": SEED}, debug=False)
    env.run([lambda obs: [], lambda obs: []])
    return env


def test_closest_approach_matches_env_seed42(env_seed_42):
    """From step 1 (the first turn an agent acts on), predict closest-approach
    turn and distance between my home (planet 0) and every orbiting planet.
    Step the env to that turn and verify actual distance matches prediction.

    We anchor on step 1 rather than step 0 to dodge the env's N-1 rotation
    offset documented in `lib/orbit.py::predict_absolute`: env.steps[0] and
    env.steps[1] both have 0 rotations applied, so `predict_relative` from
    obs0 with `lead_turns=t` aligns with env.steps[t+1], not env.steps[t].
    From step 1+ the relative formula is exact (proven in
    `tests/test_orbit.py::test_predict_relative_zero_error_against_env_at_step_100`).
    """
    env = env_seed_42
    obs1 = env.steps[1][0].observation
    obs0 = env.steps[0][0].observation
    omega = obs0["angular_velocity"]
    home = obs1["planets"][0]

    checked = 0
    max_err = 0.0
    for p in obs1["planets"]:
        if p[0] == 0:
            continue
        if not orbit.is_orbiting(p):
            continue
        t_star, d_pred = orbit_lead.closest_approach(home, p, omega, horizon=200)
        env_step_idx = 1 + t_star  # t_star is lead-turns from obs1
        if env_step_idx >= len(env.steps):
            continue
        actual_obs = env.steps[env_step_idx][0].observation
        actual_planet = next((q for q in actual_obs["planets"] if q[0] == p[0]), None)
        if actual_planet is None:
            continue
        actual_home = next((q for q in actual_obs["planets"] if q[0] == 0), None)
        if actual_home is None:
            continue
        d_actual = math.hypot(
            actual_planet[2] - actual_home[2],
            actual_planet[3] - actual_home[3],
        )
        err = abs(d_pred - d_actual)
        max_err = max(max_err, err)
        checked += 1
    assert checked >= 3, f"only {checked} orbiting planets checked on seed 42"
    assert max_err < 1e-6, (
        f"closest-approach prediction drifted by {max_err:.9f} on seed 42 "
        f"(checked {checked} pairs)"
    )


def test_closest_approach_savings_match_empirical_distribution(env_seed_42):
    """Sanity-check the empirical-report headline (35% mean / 75% max
    savings on seed 42's orbiting targets). We don't expect EXACT match
    on a single seed, but at least one target should show >= 30% savings."""
    env = env_seed_42
    obs1 = env.steps[1][0].observation
    omega = env.steps[0][0].observation["angular_velocity"]
    home = obs1["planets"][0]

    gaps = []
    for p in obs1["planets"]:
        if p[0] == 0 or not orbit.is_orbiting(p):
            continue
        d0 = orbit_lead.distance_at(home, p, omega, 0)
        _, d_min = orbit_lead.closest_approach(home, p, omega, horizon=500)
        if d0 > 1e-6:
            gaps.append((d0 - d_min) / d0)
    assert gaps, "no orbiting targets for the savings check"
    assert max(gaps) >= 0.30, (
        f"no orbiting target shows >= 30% closest-approach savings on seed 42; "
        f"max savings = {max(gaps):.1%}"
    )
