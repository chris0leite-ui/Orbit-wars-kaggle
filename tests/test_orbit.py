"""Tests for lib/orbit.py — orbit-prediction primitives.

The load-bearing test is `test_predict_relative_zero_error_against_env`:
it runs the live env to step 100 and asserts the relative formula matches
exactly (was 0.0 error in scripts/orbit_prediction_check.py on seed 42).
If this regresses, we lose orbit-aware planning for v1 onward.
"""

from __future__ import annotations

import math

import pytest

from kaggle_environments import make

from lib import geometry as G
from lib import orbit


SEED = 42


# ---------------------------------------------------------------------------
# Pure-formula tests — no env runs
# ---------------------------------------------------------------------------


def _planet(pid: int, x: float, y: float, radius: float = 1.0) -> list:
    """Helper: planet tuple shape [id, owner, x, y, radius, ships, production]."""
    return [pid, -1, x, y, radius, 0, 1]


def test_is_orbiting_inner_planet_orbits():
    # 30 units from centre + radius 1 = 31 < 50 → orbits.
    p = _planet(0, x=G.CENTER + 30.0, y=G.CENTER, radius=1.0)
    assert orbit.is_orbiting(p) is True


def test_is_orbiting_outer_planet_static():
    # 49 units + radius 2 = 51 >= 50 → static.
    p = _planet(0, x=G.CENTER + 49.0, y=G.CENTER, radius=2.0)
    assert orbit.is_orbiting(p) is False


def test_is_orbiting_boundary_exactly_at_limit_is_static():
    # orbital_radius + planet_radius == 50 → README says "< 50" so this is static.
    p = _planet(0, x=G.CENTER + 49.0, y=G.CENTER, radius=1.0)  # 49 + 1 = 50
    assert orbit.is_orbiting(p) is False


def test_predict_relative_zero_lead_returns_current_position():
    p = _planet(0, x=G.CENTER + 10.0, y=G.CENTER + 5.0, radius=1.0)
    x, y = orbit.predict_relative(p, angular_velocity=0.04, lead_turns=0)
    assert x == pytest.approx(p[2])
    assert y == pytest.approx(p[3])


def test_predict_relative_zero_omega_is_a_noop():
    p = _planet(0, x=G.CENTER + 10.0, y=G.CENTER + 5.0, radius=1.0)
    x, y = orbit.predict_relative(p, angular_velocity=0.0, lead_turns=42)
    assert x == pytest.approx(p[2])
    assert y == pytest.approx(p[3])


def test_predict_relative_full_revolution_returns_to_start():
    p = _planet(0, x=G.CENTER + 10.0, y=G.CENTER, radius=1.0)
    x, y = orbit.predict_relative(p, angular_velocity=2 * math.pi, lead_turns=1)
    assert x == pytest.approx(p[2], abs=1e-9)
    assert y == pytest.approx(p[3], abs=1e-9)


def test_predict_absolute_step_zero_returns_init_position():
    init = _planet(0, x=G.CENTER + 10.0, y=G.CENTER, radius=1.0)
    x, y = orbit.predict_absolute(init, angular_velocity=0.04, env_step_n=0)
    assert x == pytest.approx(init[2])
    assert y == pytest.approx(init[3])


def test_predict_absolute_step_one_no_rotation_yet():
    """env.steps[1] is the snapshot BEFORE step 1's rotation applies (n_rot = 0)."""
    init = _planet(0, x=G.CENTER + 10.0, y=G.CENTER, radius=1.0)
    x, y = orbit.predict_absolute(init, angular_velocity=0.04, env_step_n=1)
    assert x == pytest.approx(init[2])
    assert y == pytest.approx(init[3])


# ---------------------------------------------------------------------------
# Live-env integration: load-bearing finding from A.1.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env_seed_42():
    """Run the env to its full length once with no-op agents; reuse env.steps."""
    env = make("orbit_wars", configuration={"seed": SEED}, debug=False)
    env.run([lambda obs: [], lambda obs: []])
    return env


def test_predict_relative_zero_error_against_env_at_step_100(env_seed_42):
    """The relative formula must match `env.steps[100]` exactly (0.0 error)
    on the published seed. Regression here = orbit-aware planning is broken.
    """
    env = env_seed_42
    obs0 = env.steps[0][0].observation
    omega = obs0["angular_velocity"]
    obs50 = env.steps[50][0].observation
    obs100 = env.steps[100][0].observation
    rel_by_id = {p[0]: p for p in obs50["planets"]}
    actual_by_id = {p[0]: p for p in obs100["planets"]}

    max_err = 0.0
    checked = 0
    for p in obs0["initial_planets"]:
        if not orbit.is_orbiting(p):
            continue
        pid = p[0]
        if pid not in rel_by_id or pid not in actual_by_id:
            continue
        px, py = orbit.predict_relative(rel_by_id[pid], omega, lead_turns=50)
        ax, ay = actual_by_id[pid][2], actual_by_id[pid][3]
        err = math.hypot(px - ax, py - ay)
        max_err = max(max_err, err)
        checked += 1
    assert checked > 0, "no orbiting planets found on seed 42 — fixture broken"
    assert max_err == pytest.approx(0.0, abs=1e-9), (
        f"relative orbit formula drifted by {max_err:.6f} on seed 42 over 50 turns "
        f"(checked {checked} planets) — this used to be 0.0"
    )


def test_predict_absolute_zero_error_with_n_minus_one_offset(env_seed_42):
    """The absolute formula with N-1 offset must also match exactly.
    The naive `omega*N` form is off by `omega * orb_r` board units.
    """
    env = env_seed_42
    obs0 = env.steps[0][0].observation
    omega = obs0["angular_velocity"]
    actual_by_id = {p[0]: p for p in env.steps[100][0].observation["planets"]}

    max_err = 0.0
    for init in obs0["initial_planets"]:
        if not orbit.is_orbiting(init):
            continue
        pid = init[0]
        if pid not in actual_by_id:
            continue
        px, py = orbit.predict_absolute(init, omega, env_step_n=100)
        ax, ay = actual_by_id[pid][2], actual_by_id[pid][3]
        max_err = max(max_err, math.hypot(px - ax, py - ay))
    assert max_err == pytest.approx(0.0, abs=1e-9)


def test_static_planets_do_not_drift(env_seed_42):
    env = env_seed_42
    init_planets = env.steps[0][0].observation["initial_planets"]
    actual_by_id = {p[0]: p for p in env.steps[100][0].observation["planets"]}

    max_drift = 0.0
    for init in init_planets:
        if orbit.is_orbiting(init):
            continue
        pid = init[0]
        if pid not in actual_by_id:
            continue
        ax, ay = actual_by_id[pid][2], actual_by_id[pid][3]
        ix, iy = init[2], init[3]
        max_drift = max(max_drift, math.hypot(ax - ix, ay - iy))
    # Static planets should stay put unless captured/comet-merged; tolerate
    # tiny float noise but anything > 1e-6 means the env actually moved them.
    assert max_drift < 1e-6
