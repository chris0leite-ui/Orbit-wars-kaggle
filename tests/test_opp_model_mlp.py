"""Sanity tests for the MLP-as-opp-model tier in `lib/opp_model.py`.

Coverage:
  - Weights load cleanly; `ensemble_proba` returns valid probabilities.
  - `mlp_validated_policy` returns a (possibly empty) subset of
    `lite_greedy_policy`'s emits on a real self-play observation.
  - Threshold gate monotonicity: raising `BASELINE_OPP_MLP_THRESHOLD`
    never INCREASES the emit count.
  - `BASELINE_OPP_MODEL=lite_greedy` (or unset) preserves the legacy
    selector path — `_select_opp_policy()` returns `lite_greedy_policy`,
    NOT the MLP policy.
  - Empty obs (no owned planets) returns [], no crash.
"""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _self_play_obs(seed: int = 42, warmup_turns: int = 5, seat: int = 0):
    """Drive the env forward a few turns and return the obs for `seat`."""
    from kaggle_environments import make
    import random

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    rng = random.Random(seed)
    for _ in range(warmup_turns):
        acts = []
        for p in range(2):
            launches = []
            for pl in env.state[0].observation["planets"]:
                if pl[1] == p and pl[5] > 5 and rng.random() < 0.3:
                    launches.append(
                        [pl[0], rng.uniform(0.0, 6.283), int(pl[5] // 2)]
                    )
            acts.append(launches)
        env.step(acts)
    return env.state[seat].observation


def _reload_opp_model():
    """Re-import lib.opp_model so the module-level `OPP_MLP_THRESHOLD`
    constant picks up the current `BASELINE_OPP_MLP_THRESHOLD` env var.
    Returns the freshly-loaded module."""
    import lib.opp_model
    return importlib.reload(lib.opp_model)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_validator_mlp_loads_and_proba_in_unit_interval():
    """Weights blob parses; ensemble forward produces sigmoid outputs."""
    from lib._validator_mlp import ensemble_proba, is_ready
    assert is_ready(), "validator weights failed to load"
    X = np.zeros((8, 25), dtype=np.float32)
    X[0, 24] = 0.8
    X[1, 24] = -0.8
    X[2, 21] = 0.5  # ship_diff
    p = ensemble_proba(X)
    assert p.shape == (8,)
    assert p.dtype == np.float32
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_mlp_policy_is_subset_of_lite_greedy_emits():
    """The MLP filter can only DROP candidates, never invent new ones."""
    from lib.opp_model import lite_greedy_policy, mlp_validated_policy

    obs = _self_play_obs(seed=7, warmup_turns=6)
    lite_emits = lite_greedy_policy(obs)
    mlp_emits = mlp_validated_policy(obs)

    lite_set = {(int(m[0]), round(float(m[1]), 6), int(m[2])) for m in lite_emits}
    mlp_set = {(int(m[0]), round(float(m[1]), 6), int(m[2])) for m in mlp_emits}
    assert mlp_set.issubset(lite_set), (
        f"MLP policy produced emits not in lite_greedy candidate set: "
        f"extra={mlp_set - lite_set}"
    )


def test_threshold_monotonicity_lowers_or_keeps_emit_count():
    """Raising the threshold can only suppress more launches, not fewer."""
    obs = _self_play_obs(seed=11, warmup_turns=8)
    prior = os.environ.get("BASELINE_OPP_MLP_THRESHOLD")
    try:
        counts = []
        for tau in (0.1, 0.3, 0.5, 0.7, 0.9):
            os.environ["BASELINE_OPP_MLP_THRESHOLD"] = str(tau)
            mod = _reload_opp_model()
            emits = mod.mlp_validated_policy(obs)
            counts.append(len(emits))
        assert counts == sorted(counts, reverse=True), (
            f"non-monotone emit counts vs threshold: {counts}"
        )
    finally:
        if prior is None:
            os.environ.pop("BASELINE_OPP_MLP_THRESHOLD", None)
        else:
            os.environ["BASELINE_OPP_MLP_THRESHOLD"] = prior
        _reload_opp_model()


def test_selector_default_is_lite_greedy_not_mlp():
    """`_select_opp_policy()` must NOT pick the MLP tier unless explicitly
    opted in via `BASELINE_OPP_MODEL=mlp`. Byte-parity for default
    `baseline_leaf_pv_2p` stack."""
    from agents.baseline.chooser import _select_opp_policy
    from lib.opp_model import lite_greedy_policy, mlp_validated_policy

    prior = os.environ.pop("BASELINE_OPP_MODEL", None)
    try:
        sel = _select_opp_policy()
        assert sel is lite_greedy_policy, (
            f"default selector should be lite_greedy, got {sel.__name__}"
        )
        assert sel is not mlp_validated_policy
    finally:
        if prior is not None:
            os.environ["BASELINE_OPP_MODEL"] = prior


def test_selector_opt_in_routes_to_mlp():
    """`BASELINE_OPP_MODEL=mlp` switches the selector to the MLP tier."""
    from agents.baseline.chooser import _select_opp_policy
    from lib.opp_model import mlp_validated_policy

    prior = os.environ.get("BASELINE_OPP_MODEL")
    try:
        os.environ["BASELINE_OPP_MODEL"] = "mlp"
        sel = _select_opp_policy()
        assert sel is mlp_validated_policy
    finally:
        if prior is None:
            os.environ.pop("BASELINE_OPP_MODEL", None)
        else:
            os.environ["BASELINE_OPP_MODEL"] = prior


def test_mlp_policy_handles_empty_obs():
    """No owned planets → []. No crash on the degenerate-obs path."""
    from lib.opp_model import mlp_validated_policy

    obs = {
        "player": 0,
        "planets": [],
        "fleets": [],
        "step": 0,
        "angular_velocity": 0.005,
    }
    out = mlp_validated_policy(obs)
    assert out == []
