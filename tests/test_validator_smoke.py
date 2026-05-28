"""Smoke tests for the shot validator pipeline.

Verifies:
  - lib.shot_features.encode_shot_features round-trip on a synthetic obs
  - agents.baseline_validated.main imports cleanly with placeholder weights
  - self-reinforce emits bypass the filter when weights are not loaded
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pytest

from lib.shot_features import (
    FEATURE_DIM,
    encode_shot_features,
    fleet_speed,
    infer_target_pid,
    target_owned_by,
)


def _synthetic_obs(focal_seat: int = 0) -> dict:
    """Two-planet world: planet 0 owned by seat 0, planet 1 by seat 1."""
    return {
        "step": 30,
        "player": focal_seat,
        "planets": [
            # (id, owner, x, y, radius, ships, production)
            [0, 0, 30.0, 50.0, 1.0, 25.0, 3.0],
            [1, 1, 70.0, 50.0, 1.0, 18.0, 2.5],
        ],
        "fleets": [
            # (id, owner, x, y, angle, from_pid, ships)
            [10, 0, 40.0, 50.0, 0.0, 0, 8.0],
        ],
    }


def test_fleet_speed_monotonic():
    assert fleet_speed(1) < fleet_speed(100) < fleet_speed(1000)


def test_infer_target_forward_planet():
    obs = _synthetic_obs()
    # Aim straight right from planet 0 -> planet 1 at angle 0
    pid = infer_target_pid((30.0, 50.0), 0.0, obs["planets"])
    assert pid == 1


def test_encode_shape_and_range():
    obs = _synthetic_obs(focal_seat=0)
    feats = encode_shot_features([0, 0.0, 10.0], obs, focal_seat=0)
    assert feats is not None
    assert feats.shape == (FEATURE_DIM,)
    assert feats.dtype == np.float32
    # All non-ship_diff features in [0, 1]; ship_diff in [-1, 1]
    for i, v in enumerate(feats):
        if i == 21:
            assert -1.0 <= v <= 1.0
        else:
            assert 0.0 <= v <= 1.0, f"feat {i}={v} out of [0,1]"


def test_target_owned_by_self_reinforce():
    obs = _synthetic_obs(focal_seat=0)
    # Aim straight left from planet 1 -> planet 0 (which we own)
    # But we're focal_seat=0, so the SHOT is from seat=1's POV launching at
    # planet 0. From seat 1's POV, target owner == self? No, target is seat 0.
    # Use angle pi to fire from p1 -> p0; from seat 1 perspective.
    assert target_owned_by([1, math.pi, 10.0], obs, focal_seat=1) is False
    # From seat 0's POV, firing p0 -> p0 (self) would be invalid (src == tgt
    # is rejected by infer_target_pid). So fire p0 at angle 0 toward p1
    # (owned by seat 1) — owner == seat 0? No, owner == 1.
    assert target_owned_by([0, 0.0, 10.0], obs, focal_seat=0) is False
    # Real self-reinforce: build a 3-planet obs where we own both p0 and p2
    obs2 = {
        "step": 30, "player": 0,
        "planets": [
            [0, 0, 30.0, 50.0, 1.0, 25.0, 3.0],
            [1, 1, 70.0, 50.0, 1.0, 18.0, 2.5],
            [2, 0, 30.0, 30.0, 1.0, 12.0, 2.0],
        ],
        "fleets": [],
    }
    # From p0 aim at p2 (straight down): angle = -pi/2 (Y-axis down)
    # Standard math convention: angle is measured from +x axis.
    # p0 = (30, 50), p2 = (30, 30) -> dy = -20, dx = 0 -> angle = atan2(-20, 0) = -pi/2
    assert target_owned_by([0, -math.pi / 2, 5.0], obs2, focal_seat=0) is True


def test_validator_agent_imports():
    # Should import even with placeholder weights (no exception).
    from agents.baseline_validated import main as bv  # noqa: F401


def test_validator_passes_through_when_no_weights():
    """With empty _WEIGHTS_B64, the wrapper must fall through to inner."""
    from agents.baseline_validated import main as bv

    # Force-reset cache to "not yet loaded"
    bv._MODELS = None
    bv._LOAD_FAILED = False

    # Stub the inner agent to return a known list.
    sentinel = [[0, 0.0, 10.0]]
    original = bv._inner_agent
    bv._inner_agent = lambda obs, cfg=None: list(sentinel)
    try:
        out = bv.agent(_synthetic_obs(), None)
        assert out == sentinel
    finally:
        bv._inner_agent = original
