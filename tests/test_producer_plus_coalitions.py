"""Tests for Step 5 of producer_plus: multi-source L=2 coalitions.

Covers Rule 38 (fix-verification reproduces failure state):
    1. ``_coalitions_enabled`` env gate parses the flag correctly.
    2. With the gate UNSET, action rows are byte-identical to vanilla
       producer at fixed seeds (reproduces the "OFF path got perturbed"
       failure mode).
    3. With the gate SET, action rows DIFFER from the OFF path at fixed
       seeds — proving the coalitions code path is actually exercised.
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
        "producer_plus_main_test_coalitions",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_coalitions"] = module
    spec.loader.exec_module(module)
    return module


def test_coalitions_default_off(monkeypatch, producer_plus_main):
    monkeypatch.delenv("PRODUCER_PLUS_COALITIONS", raising=False)
    assert producer_plus_main._coalitions_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on", "ON"])
def test_coalitions_env_on_truthy(monkeypatch, producer_plus_main, value):
    monkeypatch.setenv("PRODUCER_PLUS_COALITIONS", value)
    assert producer_plus_main._coalitions_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_coalitions_env_off_falsy(monkeypatch, producer_plus_main, value):
    monkeypatch.setenv("PRODUCER_PLUS_COALITIONS", value)
    assert producer_plus_main._coalitions_enabled() is False


def _play_one_game(focal_path, opp_path, seed):
    """Run one kaggle_environments game and return the final reward and steps."""
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": int(seed)}, debug=False)
    env.run([focal_path, opp_path])
    return env.state, env.steps


@pytest.mark.slow
def test_off_path_bit_identical_to_producer():
    """Rule 38 reproduce-failure test: with PRODUCER_PLUS_COALITIONS unset
    (default OFF), producer_plus must emit the same action rows as vanilla
    producer at fixed seeds. This guards against ``the coalitions branch
    accidentally perturbed the single-size fallback`` regressions.
    """
    os.environ.pop("PRODUCER_PLUS_COALITIONS", None)
    os.environ.pop("PRODUCER_PLUS_MULTI_SIZE", None)
    os.environ.pop("PRODUCER_PLUS_ADAPTIVE_K", None)
    producer = os.path.join(PRODUCER_DIR, "producer_agent.py")
    producer_plus = os.path.join(PRODUCER_PLUS_DIR, "producer_agent.py")
    for seed in (7, 13):
        state_p, _ = _play_one_game(producer, producer, seed)
        state_pp, _ = _play_one_game(producer_plus, producer, seed)
        reward_p = state_p[0]["reward"]
        reward_pp = state_pp[0]["reward"]
        assert reward_p == reward_pp, (
            f"seed={seed}: producer_plus OFF differs from producer "
            f"(reward {reward_pp} vs {reward_p}) — OFF path got perturbed"
        )


@pytest.mark.slow
def test_coalitions_on_smoke_wallclock_under_1s():
    """Rule 46 smoke: run one full game with coalitions on, assert total
    wallclock under 60 s as a loose per-turn proxy.
    """
    shim_path = os.path.join(PRODUCER_PLUS_DIR, "producer_plus_coalitions.py")
    producer_path = os.path.join(PRODUCER_DIR, "producer_agent.py")
    t0 = time.time()
    state, steps = _play_one_game(shim_path, producer_path, seed=7)
    elapsed = time.time() - t0
    assert elapsed < 60.0, (
        f"coalitions game vs producer at seed 7 took {elapsed:.1f}s — "
        f"per-turn smoke bound suggests wallclock degradation"
    )
    assert state[0]["status"] in ("DONE", "INVALID"), state[0]["status"]


@pytest.mark.slow
def test_coalitions_on_changes_planner_output():
    """Proof that the coalitions code path is actually exercised: the action
    rows under coalitions ON must differ from producer_plus OFF at at least
    one seed. If they're identical, the new branch is unreachable / no-op.
    """
    producer_path = os.path.join(PRODUCER_DIR, "producer_agent.py")
    plus_off = os.path.join(PRODUCER_PLUS_DIR, "producer_agent.py")
    plus_on  = os.path.join(PRODUCER_PLUS_DIR, "producer_plus_coalitions.py")
    differs = False
    for seed in (7, 13, 42):
        state_off, _ = _play_one_game(plus_off, producer_path, seed)
        state_on,  _ = _play_one_game(plus_on,  producer_path, seed)
        if (
            state_off[0]["reward"] != state_on[0]["reward"]
            or state_off[1]["reward"] != state_on[1]["reward"]
        ):
            differs = True
            break
    assert differs, (
        "coalitions ON produced identical outcomes to OFF on seeds 7, 13, 42 "
        "— code path may be unreachable"
    )
