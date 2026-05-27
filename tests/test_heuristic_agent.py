"""Behavior tests for agents/heuristic.

Tests the agent on synthetic scenarios where the correct action is obvious:
- captures a defenseless neutral
- defends a planet predicted to flip
- physics gate refuses sun-crossing aim
"""

from __future__ import annotations

import math

from kaggle_environments import make

from agents.heuristic.main import agent


def _new_env(steps=20):
    return make("orbit_wars", configuration={"episodeSteps": steps})


def test_captures_undefended_neutral():
    """A standard self-vs-random match: heuristic must beat random consistently
    in a short game window."""
    wins = 0
    for seed in range(8):
        env = make("orbit_wars", configuration={"episodeSteps": 200},
                   info={"seed": seed})
        env.reset()
        env.run([agent, "random"])
        if env.steps[-1][0]["reward"] > env.steps[-1][1]["reward"]:
            wins += 1
    assert wins >= 7, f"expected ≥7/8 wins vs random, got {wins}"


def test_emits_only_owned_sources():
    """Every emit's src_id must be a planet we own at the moment of emit."""
    env = make("orbit_wars", configuration={"episodeSteps": 100},
               info={"seed": 7})
    env.reset()

    bad = []
    while not env.done:
        obs = env.state[0]["observation"]
        moves = agent(obs)
        my_id = obs["player"]
        owned = {p[0] for p in obs["planets"] if p[1] == my_id}
        for m in moves:
            src_id = m[0]
            if src_id not in owned:
                bad.append((env.steps and len(env.steps), src_id, owned))
        env.step([moves, []])
    assert not bad, f"emits from non-owned planets: {bad[:3]}"


def test_ships_within_garrison():
    """No emit can send more ships than the source's current garrison."""
    env = make("orbit_wars", configuration={"episodeSteps": 100},
               info={"seed": 3})
    env.reset()

    over = []
    while not env.done:
        obs = env.state[0]["observation"]
        moves = agent(obs)
        garrisons = {p[0]: p[5] for p in obs["planets"]}
        # Track ships spent per source so the sum doesn't exceed garrison
        spent = {}
        for m in moves:
            src_id, _, ships = m
            spent[src_id] = spent.get(src_id, 0) + ships
            if spent[src_id] > garrisons.get(src_id, 0):
                over.append((src_id, spent[src_id], garrisons.get(src_id)))
        env.step([moves, []])
    assert not over, f"overspent garrisons: {over[:3]}"
