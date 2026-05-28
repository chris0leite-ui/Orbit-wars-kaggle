"""Pre/post-bundle parity for the learned value head.

What this guarantees:
  - `agents.baseline.value_learned.favor_learned` and the inlined copy
    inside `submissions/baseline.py` produce bit-identical outputs on
    real game observations.
  - The embedded base64 weight blob round-trips through the bundler
    without corruption.
  - The `BASELINE_VALUE_HEAD=learned` dispatcher path actually selects
    `favor_learned` (not the silent-fallback `favor`).

This is the only test that exercises the post-bundle path; it must run
AFTER `scripts/bundle_agent.py` has been invoked at least once on
`agents/baseline`. CI ordering: feature tests -> bundler smoke -> this.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
BUNDLED = ROOT / "submissions" / "baseline.py"


def _load_bundled():
    if not BUNDLED.exists():
        pytest.skip(
            f"{BUNDLED} not built; run "
            "`python scripts/bundle_agent.py agents/baseline --force` first"
        )
    spec = importlib.util.spec_from_file_location("_bundled_baseline", BUNDLED)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample_observations() -> list[dict]:
    """A handful of structurally distinct boards covering the cases the
    feature extractor branches over."""
    return [
        {  # empty board
            "planets": [], "fleets": [], "step": 0,
        },
        {  # me-only
            "planets": [(0, 0, 20.0, 20.0, 2.0, 10.0, 2.0)],
            "fleets": [], "step": 10,
        },
        {  # symmetric 2P with some fleets
            "planets": [
                (0, 0, 20.0, 20.0, 2.0, 10.0, 2.0),
                (1, 1, 80.0, 80.0, 2.0, 12.0, 1.5),
                (2, -1, 50.0, 30.0, 2.0, 3.0, 0.5),
            ],
            "fleets": [
                (0, 0, 40.0, 40.0, 0.5, 0, 5.0),
                (1, 1, 60.0, 60.0, 1.5, 1, 4.0),
            ],
            "step": 100,
        },
        {  # late-game asymmetric
            "planets": [
                (0, 0, 20.0, 20.0, 2.0, 60.0, 3.0),
                (1, 0, 30.0, 70.0, 2.0, 40.0, 2.5),
                (2, 1, 80.0, 80.0, 2.0, 5.0, 0.5),
            ],
            "fleets": [
                (0, 0, 50.0, 50.0, 0.0, 0, 20.0),
            ],
            "step": 400,
        },
    ]


def test_bundled_module_exposes_favor_learned():
    m = _load_bundled()
    assert hasattr(m, "favor_learned"), "bundled module missing favor_learned"
    assert hasattr(m, "weights_loaded"), "bundled module missing weights_loaded"
    assert hasattr(m, "agent"), "bundled module missing agent callable"


def test_bundled_weights_loaded():
    m = _load_bundled()
    if not m.weights_loaded():
        pytest.skip(
            "bundle ships ZERO_FALLBACK weights — expected pre-training; "
            "rerun `scripts/embed_value_head_weights.py` after training "
            "and re-bundle"
        )


def test_bundled_favor_matches_source():
    """Bit-exact parity: bundled inline copy must match the source module."""
    from agents.baseline.value_learned import favor_learned as src_favor

    m = _load_bundled()
    for obs in _sample_observations():
        for me in (0, 1):
            v_src = src_favor(obs, me=me, num_seats=2)
            v_bun = m.favor_learned(obs, me=me, num_seats=2)
            assert v_src == v_bun, (
                f"parity mismatch on obs step={obs['step']}, me={me}: "
                f"src={v_src!r} bundled={v_bun!r}"
            )


def test_dispatcher_selects_learned_head():
    """BASELINE_VALUE_HEAD=learned must route through favor_learned."""
    m = _load_bundled()
    # The dispatcher is bundled too — it should resolve to favor_learned.
    prev = os.environ.get("BASELINE_VALUE_HEAD")
    os.environ["BASELINE_VALUE_HEAD"] = "learned"
    try:
        fn = m.select_favor_fn()
        # Sanity: calling fn on a real-ish obs must return something
        # numeric (NOT raise, NOT return None).
        obs = _sample_observations()[2]
        val = fn(obs, me=0, num_seats=2)
        assert isinstance(val, float)
        assert np.isfinite(val)
        # And it should be exactly equal to a direct favor_learned call.
        assert val == m.favor_learned(obs, me=0, num_seats=2)
    finally:
        if prev is None:
            os.environ.pop("BASELINE_VALUE_HEAD", None)
        else:
            os.environ["BASELINE_VALUE_HEAD"] = prev


def test_forward_batch_matches_scalar_path():
    """Batch forward over N inputs must equal N scalar forwards."""
    from agents.baseline.value_learned import (
        favor_learned, forward_batch,
    )
    from lib.value_features import extract_features

    obs_list = _sample_observations()
    feats = np.stack(
        [extract_features(o, me=0, num_seats=2) for o in obs_list],
        axis=0,
    )
    batched = forward_batch(feats)
    scalar = np.array(
        [favor_learned(o, me=0, num_seats=2) for o in obs_list],
        dtype=np.float32,
    )
    np.testing.assert_allclose(batched, scalar, rtol=1e-5, atol=1e-5)
