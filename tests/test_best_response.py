"""Smoke + invariants for the best_response agent.

Cheap guards (no statistical claims): the agent loads, plays a full game
inside the per-turn budget, emits well-formed actions, and degrades to the
Producer's move rather than crashing. Heavier A/B lives in fast.py eval.
"""
from __future__ import annotations

import importlib.util
import os
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch = pytest.importorskip("torch")  # producer dependency


def _load_agent():
    path = os.path.join(REPO, "agents", "best_response", "main.py")
    spec = importlib.util.spec_from_file_location("br_test_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _nearest_path():
    return os.path.join(REPO, "agents", "simple", "nearest.py")


def _load_callable(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def test_imports_and_decides():
    BR = _load_agent()
    assert hasattr(BR, "agent")


def test_full_game_under_budget():
    """One full game vs a cheap opponent; every turn under the 1s cap and
    actions are well-formed [src, angle, ships] launches the agent owns."""
    from kaggle_environments import make

    BR = _load_agent()
    nearest = _load_callable(_nearest_path(), "nearest_test")

    env = make("orbit_wars", configuration={"episodeSteps": 120, "seed": 3})
    env.reset(num_agents=2)
    max_ms = 0.0
    steps = 0
    while not env.done and env.state[0].observation["step"] < 120:
        obs0 = env.state[0].observation
        t = time.perf_counter()
        a0 = BR.agent(obs0, env.configuration)
        max_ms = max(max_ms, (time.perf_counter() - t) * 1000.0)
        # well-formed action
        assert isinstance(a0, list)
        owned = {int(p[0]) for p in obs0["planets"] if int(p[1]) == 0}
        for mv in a0:
            assert len(mv) == 3
            assert int(mv[0]) in owned
            assert int(mv[2]) >= 1
        a1 = nearest(env.state[1].observation)
        env.step([a0, a1])
        steps += 1
    assert steps > 0
    # Generous ceiling: real budget is 1000ms; flag if we ever approach it.
    assert max_ms < 1000.0, f"max turn {max_ms:.0f}ms exceeded budget"


def test_fallback_on_producer_failure(monkeypatch):
    """If the inner decision path raises, agent() must still return a list
    (the Producer move, or [] as the last resort) — never propagate."""
    BR = _load_agent()

    def _boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(BR, "_decide", _boom)
    # Minimal valid obs: one owned planet, no fleets.
    obs = {
        "player": 0, "step": 5,
        "planets": [[0, 0, 20.0, 20.0, 2.0, 10.0, 3.0],
                    [1, -1, 60.0, 60.0, 2.0, 8.0, 2.0]],
        "fleets": [], "comets": [], "comet_planet_ids": [],
        "angular_velocity": 0.0,
    }
    out = BR.agent(obs, None)
    assert isinstance(out, list)
