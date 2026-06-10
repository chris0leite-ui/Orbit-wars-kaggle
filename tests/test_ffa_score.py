"""Tests for the FFA-aware competitive score (PRODUCER_PLUS_FFA_SCORE).

The mechanism: in 3+ player games, competitive_score's opponent term becomes
a strength-weighted average over rivals (weights sum to 1) instead of the
equal-weight sum. 2P must be byte-identical (weights are only built when
player_count >= 3).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCER_PLUS_DIR = os.path.join(REPO_ROOT, "agents", "producer_plus")
PRODUCER_DIR = os.path.join(REPO_ROOT, "agents", "producer")


@pytest.fixture(scope="module")
def pp_main():
    for p in (PRODUCER_DIR, PRODUCER_PLUS_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        "producer_plus_main_test_ffa",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_ffa"] = module
    spec.loader.exec_module(module)
    return module


def test_ffa_default_off(monkeypatch, pp_main):
    monkeypatch.delenv("PRODUCER_PLUS_FFA_SCORE", raising=False)
    assert pp_main._ffa_score_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_ffa_env_on(monkeypatch, pp_main, value):
    monkeypatch.setenv("PRODUCER_PLUS_FFA_SCORE", value)
    assert pp_main._ffa_score_enabled() is True


def _obs_tensors(planet_rows, fleet_rows):
    return {
        "planets": torch.tensor(planet_rows, dtype=torch.float32),
        "fleets": torch.tensor(fleet_rows, dtype=torch.float32).reshape(-1, 7),
    }


def test_opp_weights_strength_proportional(pp_main):
    # 4P: us=0; opp 1 has 30 planet ships, opp 2 has 10, opp 3 dead (0).
    ot = _obs_tensors(
        [
            [0, 0, 10, 10, 1, 99, 2],   # ours — must be excluded
            [1, 1, 20, 20, 1, 30, 2],
            [2, 2, 30, 30, 1, 10, 2],
        ],
        [],
    )
    w = pp_main._ffa_opp_weights(ot, player_id=0, player_count=4)
    assert w.shape == (4,)
    assert float(w[0]) == 0.0
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert abs(float(w[1]) - 0.75) < 1e-6
    assert abs(float(w[2]) - 0.25) < 1e-6
    assert float(w[3]) == 0.0


def test_opp_weights_include_fleets(pp_main):
    # opp 1: 10 on planet; opp 2: 10 in a fleet — equal weights.
    ot = _obs_tensors(
        [[1, 1, 20, 20, 1, 10, 2]],
        [[0, 2, 5, 5, 0.0, 1, 10]],
    )
    w = pp_main._ffa_opp_weights(ot, player_id=0, player_count=4)
    assert abs(float(w[1]) - 0.5) < 1e-6
    assert abs(float(w[2]) - 0.5) < 1e-6


def test_opp_weights_uniform_mode(monkeypatch, pp_main):
    # Same strengths as the proportional test; uniform mode gives the two
    # living rivals equal weight and the dead one zero.
    monkeypatch.setenv("PRODUCER_PLUS_FFA_WEIGHTS", "uniform")
    ot = _obs_tensors(
        [
            [0, 0, 10, 10, 1, 99, 2],
            [1, 1, 20, 20, 1, 30, 2],
            [2, 2, 30, 30, 1, 10, 2],
        ],
        [],
    )
    w = pp_main._ffa_opp_weights(ot, player_id=0, player_count=4)
    assert abs(float(w[1]) - 0.5) < 1e-6
    assert abs(float(w[2]) - 0.5) < 1e-6
    assert float(w[3]) == 0.0


def test_opp_weights_all_dead(pp_main):
    ot = _obs_tensors([[0, 0, 10, 10, 1, 99, 2]], [])
    w = pp_main._ffa_opp_weights(ot, player_id=0, player_count=4)
    assert float(w.sum()) == 0.0


def test_competitive_score_weighted_semantics():
    """The objective fix itself: mutual-damage trades devalued, profitable
    captures still strongly valued, leader-damage worth more than runt-damage.
    """
    sys.path.insert(0, PRODUCER_DIR)
    from orbit_lite.planner_core import competitive_score

    class Diff:
        def __init__(self, net):
            self.net_ship_delta = torch.tensor(net, dtype=torch.float32)

    # Mutual-damage trade in 4P: I lose 30, opp1 loses 40, opp2/3 unaffected.
    trade = Diff([-30.0, -40.0, 0.0, 0.0])
    legacy = competitive_score(trade, player_id=0)
    assert float(legacy) == pytest.approx(10.0)        # legacy calls this GOOD
    w = torch.tensor([0.0, 1 / 3, 1 / 3, 1 / 3])
    ffa = competitive_score(trade, player_id=0, opp_weights=w)
    assert float(ffa) == pytest.approx(-30.0 + 40.0 / 3)   # ≈ -16.7: BAD
    # Profitable capture: I gain 20, victim loses 25.
    capture = Diff([20.0, -25.0, 0.0, 0.0])
    assert float(competitive_score(capture, player_id=0, opp_weights=w)) > 20.0
    # Leader-weighting: same damage to a strong (w=0.8) vs weak (w=0.1) rival.
    dmg_strong = Diff([0.0, -10.0, 0.0, 0.0])
    dmg_weak = Diff([0.0, 0.0, 0.0, -10.0])
    w2 = torch.tensor([0.0, 0.8, 0.1, 0.1])
    assert float(competitive_score(dmg_strong, player_id=0, opp_weights=w2)) > float(
        competitive_score(dmg_weak, player_id=0, opp_weights=w2)
    )
    # 2P with the single opponent at weight 1 == legacy.
    two_p = Diff([5.0, -7.0])
    assert float(competitive_score(two_p, player_id=0)) == pytest.approx(
        float(competitive_score(two_p, player_id=0, opp_weights=torch.tensor([0.0, 1.0])))
    )


def _play_one_game(focal_path, opp_path, seed):
    """One 2P game; clears PRODUCER_PLUS_* first (clean_ab env-pollution rule)."""
    for _k in [k for k in os.environ if k.startswith("PRODUCER_PLUS_")]:
        os.environ.pop(_k)
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": int(seed)}, debug=False)
    env.run([focal_path, opp_path])
    return env.state, env.steps


def _focal_action_stream(steps):
    import json
    return json.dumps([s[0].get("action") for s in steps], sort_keys=True)


@pytest.mark.slow
def test_ffa_score_2p_byte_identical():
    """2P guard: FFA_SCORE ON must not change a single action in a 2P game
    (weights are only built when player_count >= 3)."""
    producer_path = os.path.join(PRODUCER_DIR, "producer_agent.py")
    plus = os.path.join(PRODUCER_PLUS_DIR, "producer_agent.py")
    state_off, steps_off = _play_one_game(plus, producer_path, 13)
    os.environ["PRODUCER_PLUS_FFA_SCORE"] = "1"
    try:
        from kaggle_environments import make
        env = make("orbit_wars", configuration={"seed": 13}, debug=False)
        env.run([plus, producer_path])
        state_on, steps_on = env.state, env.steps
    finally:
        os.environ.pop("PRODUCER_PLUS_FFA_SCORE", None)
    assert _focal_action_stream(steps_off) == _focal_action_stream(steps_on)
    assert state_off[0]["reward"] == state_on[0]["reward"]
