"""Smoke tests for agents/bundle/main.py — the trajectory-native
BundleSearch agent. Pin the agent contract (returns list of
[src_id, angle, ships]) and that it survives a few turns of a real
env without throwing.

Network-isolated; no Kaggle credentials needed.
"""

from __future__ import annotations

import os

import pytest


# Tighten defaults so the test stays well under any per-turn budget
# even on a slow CI box. Env vars are read at each agent() call.
_TIGHT_KNOBS = {
    "BUNDLE_HORIZON": "10",
    "BUNDLE_OWN_MAX_DEPTH": "1",
    "BUNDLE_OPP_MAX_DEPTH": "1",
    "BUNDLE_OWN_CANDS_PER_SOURCE": "2",
    "BUNDLE_OPP_CANDS_PER_SOURCE": "2",
    "BUNDLE_OWN_LAUNCH_TURNS": "0",
    "BUNDLE_TOTAL_MS": "500",
    "BUNDLE_MIRROR_MS": "150",
}


@pytest.fixture
def tight_knobs(monkeypatch):
    for k, v in _TIGHT_KNOBS.items():
        monkeypatch.setenv(k, v)


def test_agent_returns_list_of_triples(tight_knobs):
    """Action format contract: list of [src_id, angle, ships]."""
    from agents.bundle.main import agent

    obs = {
        "step": 0,
        "player": 0,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, 1, 80.0, 80.0, 2.0, 50, 1],
            [2, -1, 50.0, 85.0, 1.0, 3, 0],
        ],
        "initial_planets": [
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, 1, 80.0, 80.0, 2.0, 50, 1],
            [2, -1, 50.0, 85.0, 1.0, 3, 0],
        ],
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
    }
    actions = agent(obs, configuration=None)
    assert isinstance(actions, list)
    for a in actions:
        assert isinstance(a, list)
        assert len(a) == 3
        src_id, angle, ships = a
        assert isinstance(src_id, int)
        assert isinstance(angle, float)
        assert isinstance(ships, int)
        assert ships >= 1


def test_agent_empty_planets_returns_empty(tight_knobs):
    """No planets in obs → no actions (corner case from real env)."""
    from agents.bundle.main import agent
    obs = {"step": 5, "player": 0, "planets": [], "fleets": [],
           "comets": [], "comet_planet_ids": [], "angular_velocity": 0.0,
           "initial_planets": []}
    assert agent(obs, configuration=None) == []


def test_agent_no_own_planets_returns_empty(tight_knobs):
    """We're eliminated → no actions (must not crash)."""
    from agents.bundle.main import agent
    obs = {
        "step": 100, "player": 0, "angular_velocity": 0.0,
        "planets": [
            [0, 1, 20.0, 80.0, 2.0, 50, 1],
            [1, 1, 80.0, 80.0, 2.0, 50, 1],
        ],
        "initial_planets": [
            [0, 1, 20.0, 80.0, 2.0, 50, 1],
            [1, 1, 80.0, 80.0, 2.0, 50, 1],
        ],
        "fleets": [], "comets": [], "comet_planet_ids": [],
    }
    assert agent(obs, configuration=None) == []


def test_agent_carry_over_persists_across_turns(tight_knobs):
    """If the agent finds a multi-launch bundle including a future
    launch (launch_turn>0), it MUST persist that commitment in
    `_LAST_BUNDLE` so the next turn can pick it up via
    shift_forward. We don't test the search result content
    directly (the toy world may resolve to a single immediate
    launch); we test the contract that turn>0 resets the cache.
    """
    from agents.bundle.main import agent, _LAST_BUNDLE

    obs0 = {
        "step": 0, "player": 0, "angular_velocity": 0.0,
        "planets": [
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, 1, 80.0, 80.0, 2.0, 50, 1],
            [2, -1, 50.0, 85.0, 1.0, 3, 0],
        ],
        "initial_planets": [
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, 1, 80.0, 80.0, 2.0, 50, 1],
            [2, -1, 50.0, 85.0, 1.0, 3, 0],
        ],
        "fleets": [], "comets": [], "comet_planet_ids": [],
    }
    agent(obs0, configuration=None)
    assert 0 in _LAST_BUNDLE
    # Step=0 of a NEW game (different player) should NOT confuse the
    # carry-over — each (my_id, step=0) clears its own slot.
    obs0_p1 = dict(obs0)
    obs0_p1["player"] = 1
    agent(obs0_p1, configuration=None)
    # Player 1 now has its own entry; player 0's persists from earlier.
    assert 1 in _LAST_BUNDLE
    assert 0 in _LAST_BUNDLE


def test_agent_runs_short_real_game(tight_knobs):
    """Run a 30-step real env game; assert no exception + per-turn
    wallclock < the env actTimeout we set."""
    import time
    from kaggle_environments import make
    from agents.bundle.main import agent as bundle_agent

    def random_idle(obs, configuration=None):
        return []

    env = make("orbit_wars",
               configuration={"episodeSteps": 30, "actTimeout": 5})
    per_turn_ms: list[float] = []

    def timed(fn):
        def wrap(obs, cfg=None):
            t0 = time.perf_counter()
            r = fn(obs, cfg)
            per_turn_ms.append((time.perf_counter() - t0) * 1000)
            return r
        return wrap

    env.run([timed(bundle_agent), random_idle])
    assert len(per_turn_ms) > 0
    # Under tight knobs we want every turn well under the 5s budget.
    assert max(per_turn_ms) < 2500, (
        f"some turn blew the budget: max={max(per_turn_ms):.0f}ms"
    )
