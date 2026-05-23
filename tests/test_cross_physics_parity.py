"""Cross-engine parity gate for the precision sim vs the lib orbit predictor.

`agents/precision/sim.py` ships its own deterministic orbit predictor
(`predict_planet_pos`) so the intercept solver doesn't depend on `lib/`.
`lib/orbit.py:predict_relative` is the production substrate every other
agent uses. If they ever drift, the strike plan computed in
`agents/precision/intercept.py` will land in a different (x, y) than the
engine actually places the planet at, and shots will miss.

This test pins the two together at 1e-9 absolute on 100 random
(orbit_radius, phase, omega, lead) tuples at `obs_step >= 1`.

Note on `obs_step`: precision/sim documents an intentional first-call
off-by-one at `obs_step == 0` (the engine applies 0 rotation on the very
first interpreter call). `lib/orbit.predict_relative` is always
"current → +lead rotations" regardless of step. They are only required
to agree for `obs_step >= 1`, which is every tick after the engine's
first action processing — i.e. every real strike-planning call.

**Failing test blocks Step 3** (the strike phase relies on the precision
sim to pick the launch angle that lands on the engine's planet position).
"""
from __future__ import annotations

import math
import random

import pytest

from agents.precision import sim as psim
from lib import orbit as lorb


def _gen_cases(n: int = 100, seed: int = 20260523) -> list[tuple[float, float, float, int]]:
    """100 random (orbit_radius, phase, omega, lead) tuples.

    orbit_radius ∈ [12, 40]: stays inside the rotation limit even with a
    3-unit planet radius (43 < 50).
    phase ∈ [0, 2π).
    omega ∈ [0.005, 0.05]: spans the live distribution; default is ~0.01.
    lead ∈ [1, 250]: 1 covers the smallest reachable arrival; 250 covers
    half the game's max horizon (well past any real strike-plan window).
    """
    rng = random.Random(seed)
    out: list[tuple[float, float, float, int]] = []
    for _ in range(n):
        r = rng.uniform(12.0, 40.0)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        omega = rng.uniform(0.005, 0.05)
        lead = rng.randint(1, 250)
        out.append((r, phase, omega, lead))
    return out


@pytest.mark.parametrize("obs_step", [1, 2, 7, 42, 199])
def test_predict_planet_pos_matches_predict_relative(obs_step: int):
    """At obs_step >= 1 the two predictors must agree to 1e-9 absolute."""
    planet_radius = 3.0  # within rotation limit at all orbit_radius values
    for orbit_radius, phase, omega, lead in _gen_cases():
        # Current position from (orbit_radius, phase).
        x = lorb.CENTER + orbit_radius * math.cos(phase)
        y = lorb.CENTER + orbit_radius * math.sin(phase)

        # precision/sim path.
        sim_x, sim_y = psim.predict_planet_pos(
            x, y, planet_radius, omega, lead, obs_step=obs_step
        )

        # lib/orbit path. predict_relative takes a planet tuple shape
        # `[id, owner, x, y, radius, ships, production]`; only [2], [3]
        # are read, but pass the full shape for clarity.
        lib_x, lib_y = lorb.predict_relative(
            [0, 0, x, y, planet_radius, 0, 0], omega, lead
        )

        assert abs(sim_x - lib_x) < 1e-9, (
            f"x-drift at obs_step={obs_step}, r={orbit_radius:.4f}, "
            f"phase={phase:.4f}, omega={omega:.5f}, lead={lead}: "
            f"sim={sim_x!r}, lib={lib_x!r}, delta={sim_x - lib_x!r}"
        )
        assert abs(sim_y - lib_y) < 1e-9, (
            f"y-drift at obs_step={obs_step}, r={orbit_radius:.4f}, "
            f"phase={phase:.4f}, omega={omega:.5f}, lead={lead}: "
            f"sim={sim_y!r}, lib={lib_y!r}, delta={sim_y - lib_y!r}"
        )


def test_static_planet_returns_observed_position():
    """Both predictors must leave non-orbiting planets put.

    orbit_radius + planet_radius >= 50 → static (engine rule).
    """
    planet_radius = 3.0
    orbit_radius = 48.0  # 48 + 3 = 51 > 50 → static
    phase = 1.234
    omega = 0.01
    lead = 100
    obs_step = 5

    x = lorb.CENTER + orbit_radius * math.cos(phase)
    y = lorb.CENTER + orbit_radius * math.sin(phase)

    sim_x, sim_y = psim.predict_planet_pos(x, y, planet_radius, omega, lead, obs_step=obs_step)
    # lib/orbit.predict_relative doesn't gate on `is_orbiting` (it
    # rotates regardless), so for the static case we only assert sim
    # stays put. The caller in lib uses `is_orbiting` to gate.
    assert sim_x == x
    assert sim_y == y
    assert lorb.is_orbiting([0, 0, x, y, planet_radius, 0, 0]) is False


def test_zero_lead_returns_observed_position():
    """lead=0 (precision) and lead=0 (lib) both = current obs."""
    planet_radius = 3.0
    orbit_radius = 25.0
    phase = 0.7
    omega = 0.012
    x = lorb.CENTER + orbit_radius * math.cos(phase)
    y = lorb.CENTER + orbit_radius * math.sin(phase)

    sim_x, sim_y = psim.predict_planet_pos(x, y, planet_radius, omega, 0, obs_step=5)
    lib_x, lib_y = lorb.predict_relative([0, 0, x, y, planet_radius, 0, 0], omega, 0)
    assert abs(sim_x - x) < 1e-12 and abs(sim_y - y) < 1e-12
    assert abs(lib_x - x) < 1e-12 and abs(lib_y - y) < 1e-12
