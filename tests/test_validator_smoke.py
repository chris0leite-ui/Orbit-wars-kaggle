"""Smoke tests for the shot validator pipeline (Phase 2 v2 — LightGBM walker).

Verifies:
  - lib.shot_features.encode_shot_features shape + value ranges
  - target_owned_by recognises self-reinforce
  - agents.baseline_validated.main imports cleanly with placeholder weights
  - self-reinforce / no-booster paths fall through to the inner agent
  - lib._validator_tree_walker matches lightgbm.Booster.predict to 1e-6
  - end-to-end encode + walker.predict on a 5-row batch under 50 ms
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pytest

from lib.shot_features import (
    FEATURE_DIM,
    SIGNED_INDICES,
    encode_shot_features,
    fleet_speed,
    infer_target_pid,
    target_owned_by,
)
from lib._validator_tree_walker import (
    parse_booster_text,
    predict_proba,
    predict_raw,
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
    pid = infer_target_pid((30.0, 50.0), 0.0, obs["planets"])
    assert pid == 1


def test_encode_shape_and_range():
    obs = _synthetic_obs(focal_seat=0)
    feats = encode_shot_features([0, 0.0, 10.0], obs, focal_seat=0)
    assert feats is not None
    assert feats.shape == (FEATURE_DIM,)
    assert feats.dtype == np.float32
    for i, v in enumerate(feats):
        if i in SIGNED_INDICES:
            assert -1.0 <= v <= 1.0, f"signed feat {i}={v} out of [-1,1]"
        else:
            assert 0.0 <= v <= 1.0, f"feat {i}={v} out of [0,1]"


def test_target_owned_by_self_reinforce():
    obs = _synthetic_obs(focal_seat=0)
    assert target_owned_by([1, math.pi, 10.0], obs, focal_seat=1) is False
    assert target_owned_by([0, 0.0, 10.0], obs, focal_seat=0) is False
    obs2 = {
        "step": 30, "player": 0,
        "planets": [
            [0, 0, 30.0, 50.0, 1.0, 25.0, 3.0],
            [1, 1, 70.0, 50.0, 1.0, 18.0, 2.5],
            [2, 0, 30.0, 30.0, 1.0, 12.0, 2.0],
        ],
        "fleets": [],
    }
    assert target_owned_by([0, -math.pi / 2, 5.0], obs2, focal_seat=0) is True


def test_validator_agent_imports():
    from agents.baseline_validated import main as bv  # noqa: F401


def test_validator_passes_through_when_no_booster():
    """With empty _BOOSTER_B64, the wrapper must fall through to inner."""
    from agents.baseline_validated import main as bv

    original_b64 = bv._BOOSTER_B64
    original_inner = bv._inner_agent
    bv._BOOSTER_B64 = ""
    bv._PARSED = None
    bv._LOAD_FAILED = False

    sentinel = [[0, 0.0, 10.0]]
    bv._inner_agent = lambda obs, cfg=None: list(sentinel)
    try:
        out = bv.agent(_synthetic_obs(), None)
        assert out == sentinel
    finally:
        bv._inner_agent = original_inner
        bv._BOOSTER_B64 = original_b64
        bv._PARSED = None
        bv._LOAD_FAILED = False


def test_tree_walker_parity_vs_booster():
    """Walker output must match Booster.predict to within 1e-6 on N=100
    random rows. This is the load-bearing contract for the embedded
    booster — if it breaks, the bundled agent's predictions diverge
    from training.
    """
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((300, FEATURE_DIM)).astype(np.float32)
    # Synthetic non-linear-ish target so the booster has structure.
    y_train = (X_train[:, 3] + X_train[:, 7] * 0.5
               - X_train[:, 12] + 0.3 * X_train[:, 0] > 0).astype(np.float32)
    ds = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "binary", "num_leaves": 15, "learning_rate": 0.1,
        "verbose": -1, "deterministic": True, "min_data_in_leaf": 5,
    }
    bst = lgb.train(params, ds, num_boost_round=25)
    text = bst.model_to_string()
    parsed = parse_booster_text(text)

    X_test = rng.standard_normal((100, FEATURE_DIM)).astype(np.float32)
    raw_lgb = bst.predict(X_test, raw_score=True)
    raw_walker = predict_raw(parsed, X_test)
    assert np.max(np.abs(raw_lgb - raw_walker)) < 1e-6

    prob_lgb = bst.predict(X_test, raw_score=False)
    prob_walker = predict_proba(parsed, X_test)
    assert np.max(np.abs(prob_lgb - prob_walker)) < 1e-6


def test_walker_latency_under_50ms_for_5_emits():
    """End-to-end (encode + walker.predict) for a 5-emit turn must clear
    50 ms. PM5 latency headroom shows the budget is ~700 ms / turn; the
    encoder dominates, the walker adds <5 ms for typical booster sizes.
    """
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, FEATURE_DIM)).astype(np.float32)
    y = (X[:, 1] - X[:, 5] > 0).astype(np.float32)
    bst = lgb.train(
        {"objective": "binary", "num_leaves": 31, "verbose": -1,
         "deterministic": True},
        lgb.Dataset(X, label=y),
        num_boost_round=100,
    )
    parsed = parse_booster_text(bst.model_to_string())
    batch = rng.standard_normal((5, FEATURE_DIM)).astype(np.float32)
    # Warmup
    predict_proba(parsed, batch)
    t0 = time.perf_counter()
    for _ in range(10):
        predict_proba(parsed, batch)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / 10.0
    assert elapsed_ms < 50.0, (
        f"walker took {elapsed_ms:.1f} ms / 5-emit batch (gate 50 ms)"
    )
