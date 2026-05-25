"""Repo-wide pytest fixtures.

`reset_kinematic_table` (autouse): clears the module-level KinematicTable
singleton between tests. 2026-05-25 K1 wiring (`agents/buildup_planner`
sets `KINEMATIC_TABLE_ENABLED=1` and primes per turn) leaks the
singleton's state across tests in the same pytest run. Production code
ALWAYS calls `begin_turn(world)` before `predict_fleet_fate`, so the
fingerprint-driven rebuild keeps positions fresh. Tests that call
`predict_fleet_fate` directly (e.g. `tests/test_trajectory.py`) skip
priming and would otherwise consume stale positions from a previous
test's game. Reset is per-test to avoid cross-contamination.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_kinematic_table():
    """Reset the kinematic_table singleton before each test."""
    try:
        from lib.kinematic_table import get_default
        get_default().reset()
    except Exception:
        # Library may not be importable in lightweight CI; ignore.
        pass
    yield
