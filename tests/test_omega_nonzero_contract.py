"""Surveillance test for Fragility #5 (omega==0.0 silent fallback).

`agents/baseline/proposer.py` gates the orbital-safety geometry on
`orbital_safety and omega != 0.0 and arrival_step > 0` at three sites
(`:672`, `:764`, `:932`). If the engine ever returns `angular_velocity ==
0.0` (e.g. a future map variant with stationary planets), all three
gates silently collapse to current-position math and
`BASELINE_ORBITAL_SAFETY=1` reverts to its pre-fix behavior — the
behavior PI flagged as broken at peak.

We don't *fix* the silent fallback here (that's a separate plan); we
*surveille* it. On the standard 2P seed panel, `angular_velocity` at
turn 0 must be non-zero. If the engine ever ships a seed where it's
zero, this test fires and forces an explicit decision about whether
to assert in `WorldModel.from_world` or to handle the regime
explicitly.
"""

from __future__ import annotations

import pytest
from kaggle_environments import make


PANEL_SEEDS = (0, 1, 2, 3, 7, 11, 23, 41)


@pytest.mark.parametrize("seed", PANEL_SEEDS)
def test_angular_velocity_nonzero_at_turn_zero_2p(seed):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    obs = env.state[0]["observation"]
    omega = float(obs.get("angular_velocity", 0.0))
    assert omega != 0.0, (
        f"seed={seed}: angular_velocity==0.0 at turn 0. "
        "BASELINE_ORBITAL_SAFETY's predict-based gates "
        "(proposer.py:672, :764, :932) silently fall back to "
        "current-position math when omega==0. See Fragility #5 in "
        "state/PEAK_BASELINE.md — decide whether to assert in "
        "WorldModel.from_world or handle the regime explicitly."
    )


@pytest.mark.parametrize("seed", (0, 1, 7))
def test_angular_velocity_nonzero_at_turn_zero_4p(seed):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=4)
    obs = env.state[0]["observation"]
    omega = float(obs.get("angular_velocity", 0.0))
    assert omega != 0.0, (
        f"seed={seed} (4P): angular_velocity==0.0 at turn 0. "
        "See Fragility #5 in state/PEAK_BASELINE.md."
    )
