"""Guard: the LR_DEEP_OPP knob (Phase 1 — swappable deep-search opponent).

The deep rollout (`_deep_pick`) calls an opponent model at every node. Default
(LR_DEEP_OPP unset / 0) is the producer mirror (`_producer_move_obs`) — this MUST
stay the shipped behaviour. LR_DEEP_OPP=1 swaps in the cheap `lite_greedy_policy`
(~1-2 ms vs the mirror's ~10-50 ms) so the rollout can afford more depth under the
1000 ms wall.

These tests are torch-AGNOSTIC: the mode-0 path is checked by routing (monkeypatch
the mirror to a sentinel — no torch needed), and the mode-1 path is the pure-Python
lite policy. So they pass whether or not torch is installed.
"""
import importlib.util
import os

_MAIN = os.path.join(os.path.dirname(__file__), "..", "agents",
                     "least_resistance", "main.py")


def _load_main():
    spec = importlib.util.spec_from_file_location("lr_main_for_dispatch_test", _MAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _obs():
    """Minimal obs: one owned planet (seat 0) that can capture one enemy planet.
    Planet tuple order = (id, owner, x, y, radius, ships, production)."""
    return {
        "player": 0,
        "planets": [
            [0, 0, 0.0, 0.0, 1.0, 50, 5],    # mine, 50 ships
            [1, 1, 12.0, 0.0, 1.0, 5, 1],    # enemy, 5 defenders
        ],
        "fleets": [],
    }


def test_deep_opp_default_is_zero(monkeypatch):
    mod = _load_main()
    monkeypatch.delenv("LR_DEEP_OPP", raising=False)
    assert mod._deep_opp() == 0
    monkeypatch.setenv("LR_DEEP_OPP", "1")
    assert mod._deep_opp() == 1


def test_mode0_dispatches_to_producer_mirror(monkeypatch):
    """Default (mode 0) must route to the producer mirror, unchanged — proven by
    routing, so the assertion holds even without torch."""
    mod = _load_main()
    sentinel = [[99, 0.0, 7]]
    called = {}

    def fake_mirror(obs_any, seat):
        called["seat"] = seat
        return sentinel

    monkeypatch.setattr(mod, "_producer_move_obs", fake_mirror)
    # If mode 0 ever called lite_greedy instead, this sentinel would not return.
    out = mod._deep_opp_move(_obs(), 0, 0)
    assert out is sentinel
    assert called.get("seat") == 0


def test_mode1_dispatches_to_lite_greedy():
    """Mode 1 must return exactly lite_greedy_policy(obs) (pure Python)."""
    mod = _load_main()
    obs = _obs()
    expected = mod.lite_greedy_policy(obs)
    out = mod._deep_opp_move(obs, 0, 1)
    assert out == expected
    # The synthetic board is winnable, so the cheap policy should actually launch
    # (guards against a silently-idle opponent model).
    assert out, "lite_greedy should emit a launch on a capturable board"


def test_mode1_never_calls_the_mirror(monkeypatch):
    """Mode 1 must NOT touch the expensive mirror — that's the whole point."""
    mod = _load_main()

    def boom(*a, **k):
        raise AssertionError("mode 1 must not call the producer mirror")

    monkeypatch.setattr(mod, "_producer_move_obs", boom)
    mod._deep_opp_move(_obs(), 0, 1)  # must not raise
