"""Smoke test for the reach_frontier agent.

Plays one full game vs the kaggle starter agent on a fixed seed and
asserts: (a) the game runs to DONE without crashing, (b) every emitted
move is env-shape `[src_id, angle_rad, ships]`, (c) the agent's max
per-turn time stays under the env's 1 s/turn cap.

Doctrine §8.4 lists "fleets dying mid-flight" as a critical failure; if
this test grows flaky, that's the first place to look.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def reach_frontier_agent():
    from agents.reach_frontier.main import agent
    return agent


def test_reach_frontier_runs_full_game_no_crash(reach_frontier_agent):
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": 7}, debug=False)
    env.run([reach_frontier_agent, "starter"])
    final = env.steps[-1]
    statuses = [s.status for s in final]
    assert all(s == "DONE" for s in statuses), f"statuses={statuses}"


def test_reach_frontier_emits_env_shape_moves(reach_frontier_agent):
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": 11}, debug=False)
    env.reset(num_agents=2)
    obs0 = env.steps[0][0].observation
    out = reach_frontier_agent(obs0)
    assert isinstance(out, list), type(out)
    for move in out:
        assert isinstance(move, list), move
        assert len(move) == 3, move
        src_id, angle, ships = move
        assert isinstance(src_id, int) and src_id >= 0, move
        assert isinstance(angle, float), move
        assert isinstance(ships, int) and ships > 0, move


def test_reach_frontier_per_turn_within_budget(reach_frontier_agent):
    """Per-turn cost should comfortably clear the env's 1 s timeout.

    Exercises reach.py's pre-filter optimisations and the design's §9
    budgeting on the plain closed-form trajectory path. A regression past
    the 800 ms soft gate signals a substrate change (Rule 47-adjacent).
    """
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": 13}, debug=False)
    env.reset(num_agents=2)
    max_ms = 0.0
    # Sample the first 50 turns — enough to cover early-game build-up
    # without doubling test wallclock.
    for _ in range(50):
        obs0 = env.steps[-1][0].observation
        obs1 = env.steps[-1][1].observation
        t0 = time.perf_counter()
        a0 = reach_frontier_agent(obs0)
        a1 = reach_frontier_agent(obs1)
        max_ms = max(max_ms, (time.perf_counter() - t0) * 1000.0 / 2)
        env.step([a0, a1])
        if env.steps[-1][0].status != "ACTIVE":
            break
    assert max_ms < 800, f"max turn-ms={max_ms:.0f} exceeds 800 ms soft gate"
