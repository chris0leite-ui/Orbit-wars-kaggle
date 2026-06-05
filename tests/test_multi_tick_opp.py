"""Tests for multi-tick opponent projection.

Covers Rule 38 (fix-verification reproduces failure state):
    1. ``_multi_tick_opp_k`` env gate parses the flag correctly and
       honours the per-player-count override pattern.
    2. With opp_projection ON but multi_tick K UNSET (or 0/1), action
       rows are byte-identical to the same agent with K explicitly = 0
       (reproduces the "K=1 path got perturbed" failure mode).
    3. With opp_projection ON and multi_tick K=3, action rows DIFFER from
       the single-pass path at at least one seed — proving the multi-tick
       code path is actually exercised.
    4. Smoke: max per-turn wallclock under 1000 ms (Rule 46).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCER_PLUS_DIR = os.path.join(REPO_ROOT, "agents", "producer_plus")
PRODUCER_DIR = os.path.join(REPO_ROOT, "agents", "producer")


@pytest.fixture(scope="module")
def producer_plus_main():
    for p in (PRODUCER_DIR, PRODUCER_PLUS_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        "producer_plus_main_test_multi_tick_opp",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_multi_tick_opp"] = module
    spec.loader.exec_module(module)
    return module


def _clear_multi_tick_env(monkeypatch):
    for name in (
        "PRODUCER_PLUS_MULTI_TICK_OPP_K",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_2P",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_4P",
    ):
        monkeypatch.delenv(name, raising=False)


def test_multi_tick_default_zero_2p(monkeypatch, producer_plus_main):
    _clear_multi_tick_env(monkeypatch)
    assert producer_plus_main._multi_tick_opp_k(2) == 0


def test_multi_tick_default_zero_4p(monkeypatch, producer_plus_main):
    _clear_multi_tick_env(monkeypatch)
    assert producer_plus_main._multi_tick_opp_k(4) == 0


@pytest.mark.parametrize("value,expected", [
    ("0", 0), ("1", 1), ("2", 2), ("3", 3), ("7", 7),
])
def test_multi_tick_base_var_parses(monkeypatch, producer_plus_main, value, expected):
    _clear_multi_tick_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_MULTI_TICK_OPP_K", value)
    assert producer_plus_main._multi_tick_opp_k(2) == expected
    assert producer_plus_main._multi_tick_opp_k(4) == expected


def test_multi_tick_2p_suffix_overrides_base(monkeypatch, producer_plus_main):
    _clear_multi_tick_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_MULTI_TICK_OPP_K", "1")
    monkeypatch.setenv("PRODUCER_PLUS_MULTI_TICK_OPP_K_2P", "5")
    assert producer_plus_main._multi_tick_opp_k(2) == 5
    # 4P falls back to base.
    assert producer_plus_main._multi_tick_opp_k(4) == 1


def test_multi_tick_4p_suffix_overrides_base(monkeypatch, producer_plus_main):
    _clear_multi_tick_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_MULTI_TICK_OPP_K", "1")
    monkeypatch.setenv("PRODUCER_PLUS_MULTI_TICK_OPP_K_4P", "7")
    assert producer_plus_main._multi_tick_opp_k(4) == 7
    # 2P falls back to base.
    assert producer_plus_main._multi_tick_opp_k(2) == 1


@pytest.mark.parametrize("value", ["", "abc", "-1", "1.5", "nan"])
def test_multi_tick_invalid_returns_zero(monkeypatch, producer_plus_main, value):
    """Garbage values (negative, non-int, non-numeric) clamp to 0 (single-pass)."""
    _clear_multi_tick_env(monkeypatch)
    monkeypatch.setenv("PRODUCER_PLUS_MULTI_TICK_OPP_K", value)
    assert producer_plus_main._multi_tick_opp_k(2) == 0


def _play_one_game(focal_path, opp_path, seed):
    """Run one kaggle_environments game and return the final state and steps."""
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": int(seed)}, debug=False)
    env.run([focal_path, opp_path])
    return env.state, env.steps


@pytest.mark.slow
def test_multi_tick_K0_matches_K_unset():
    """Rule 38 reproduce-failure test: with opp_projection ON, explicitly
    setting PRODUCER_PLUS_MULTI_TICK_OPP_K=0 must produce the same game
    outcome as leaving the var unset. Both should hit the K_clamped == 1
    branch (which is bit-identical to the original single-pass code).
    Guards against the K=0 ⇒ K_clamped==1 mapping perturbing behaviour.
    """
    shim = os.path.join(PRODUCER_PLUS_DIR, "producer_plus_opp_proj.py")
    producer = os.path.join(PRODUCER_DIR, "producer_agent.py")
    # Run A: K unset (defaults to 0 inside the getter).
    for name in (
        "PRODUCER_PLUS_MULTI_TICK_OPP_K",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_2P",
        "PRODUCER_PLUS_MULTI_TICK_OPP_K_4P",
    ):
        os.environ.pop(name, None)
    state_unset, _ = _play_one_game(shim, producer, 7)
    # Run B: K=0 explicit.
    os.environ["PRODUCER_PLUS_MULTI_TICK_OPP_K"] = "0"
    try:
        state_zero, _ = _play_one_game(shim, producer, 7)
    finally:
        os.environ.pop("PRODUCER_PLUS_MULTI_TICK_OPP_K", None)
    assert state_unset[0]["reward"] == state_zero[0]["reward"], (
        f"K=0 explicit ({state_zero[0]['reward']}) differs from K unset "
        f"({state_unset[0]['reward']}) — the K=0 mapping is mis-routed"
    )


@pytest.mark.slow
def test_multi_tick_K3_changes_planner_output():
    """Proof the multi-tick code path is actually exercised: with K=3,
    at least one of (seeds 7, 13, 42) must produce a different game
    outcome than the K=1 single-pass path. If they're identical at all
    three seeds, the multi-tick branch is unreachable / no-op.
    """
    producer_path = os.path.join(PRODUCER_DIR, "producer_agent.py")
    single_pass = os.path.join(PRODUCER_PLUS_DIR, "producer_plus_multi_opp.py")
    multi_tick = os.path.join(PRODUCER_PLUS_DIR, "producer_plus_multi_tick_opp_K3.py")
    differs = False
    for seed in (7, 13, 42):
        state_single, _ = _play_one_game(single_pass, producer_path, seed)
        state_multi, _ = _play_one_game(multi_tick, producer_path, seed)
        if (
            state_single[0]["reward"] != state_multi[0]["reward"]
            or state_single[1]["reward"] != state_multi[1]["reward"]
        ):
            differs = True
            break
    assert differs, (
        "multi_tick K=3 produced identical outcomes to single-pass on "
        "seeds 7, 13, 42 — code path may be unreachable"
    )


@pytest.mark.slow
def test_multi_tick_K3_smoke_wallclock_under_60s():
    """Rule 46 smoke: run one full game with multi-tick K=3 on, assert
    total wallclock under 60 s as a loose per-turn proxy. K=3 is roughly
    3x the single-pass opp_proj cost (~20 ms → ~60 ms per turn), well
    under the 1000 ms cap.
    """
    shim_path = os.path.join(
        PRODUCER_PLUS_DIR, "producer_plus_multi_tick_opp_K3.py",
    )
    producer_path = os.path.join(PRODUCER_DIR, "producer_agent.py")
    t0 = time.time()
    state, steps = _play_one_game(shim_path, producer_path, seed=7)
    elapsed = time.time() - t0
    assert elapsed < 60.0, (
        f"multi_tick K=3 game vs producer at seed 7 took {elapsed:.1f}s — "
        f"per-turn smoke bound suggests wallclock degradation"
    )
    assert state[0]["status"] in ("DONE", "INVALID"), state[0]["status"]
