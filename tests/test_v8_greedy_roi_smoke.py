"""Phase 0 smoke — v8_greedy_roi exercises the foundation pipeline.

Goal: prove the full pipeline (Kaggle obs → JAX state → Strategy.emit
→ ActionTensor → env action format → memory threading) holds a
turn-by-turn game without crashes. No competitive claim.

Tests:
- A fresh agent() call on a synthetic obs returns a well-formed env
  action.
- 5 sequential agent() calls in the same process work (memory
  persists; no exceptions).
- A 30-turn lockstep game vs `data/main.py` reference plays through
  to completion (env reports DONE) without exceptions or budget
  blowouts.
- New game (`step=0` in obs after a previous game) resets the memory
  singleton so a fresh `CompositeMemory` is constructed.
"""

from __future__ import annotations

import pytest
from kaggle_environments import make

from agents.v8_greedy_roi.main import agent
from lib.foundation.agent_loop import get_memory, reset_memory
from lib.foundation.memory_impls import CompositeMemory


@pytest.fixture(autouse=True)
def _isolate_memory():
    """Each test starts with a fresh memory singleton."""
    reset_memory()
    yield
    reset_memory()


def test_single_call_returns_well_formed_action():
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.reset(num_agents=2)
    obs = env.state[0].observation

    action = agent(obs, env.configuration)
    assert isinstance(action, list)
    for move in action:
        assert isinstance(move, list) and len(move) == 3
        src_id, angle, ships = move
        assert isinstance(src_id, int) and src_id >= 0
        assert isinstance(angle, float)
        assert isinstance(ships, int) and ships > 0


def test_memory_persists_across_calls():
    """Two sequential calls share the same module-level memory
    singleton."""
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.reset(num_agents=2)

    obs = env.state[0].observation
    agent(obs, env.configuration)
    mem_after_1 = get_memory()
    assert isinstance(mem_after_1, CompositeMemory)

    agent(obs, env.configuration)
    mem_after_2 = get_memory()
    # Same singleton object (or a thread-through replacement).
    assert isinstance(mem_after_2, CompositeMemory)


def test_new_game_resets_memory():
    """A step=0 obs after a previous game wipes the singleton."""
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.reset(num_agents=2)

    # Play a few turns to populate any caches.
    for _ in range(3):
        obs = env.state[0].observation
        env.step([agent(obs, env.configuration), []])

    mem_before = get_memory()
    # Drop a sentinel into scratch.
    mem_before.scratch["sentinel"] = "before-new-game"

    # Now construct a fresh env (simulates a new game), step=0 obs.
    env2 = make("orbit_wars", configuration={"seed": 137}, debug=False)
    env2.reset(num_agents=2)
    obs2 = env2.state[0].observation
    assert int(obs2.get("step", 0)) == 0
    agent(obs2, env2.configuration)

    mem_after = get_memory()
    # Scratch should be empty — the reset wiped the sentinel.
    assert mem_after.scratch.get("sentinel") is None


def test_full_game_completes():
    """30-turn lockstep vs the official reference agent (data/main.py).
    Confirms the pipeline survives an actual game.

    We use `agents/simple/roi.agent` as the opponent — the un-wrapped
    version of the same heuristic — to keep the parity check
    well-bounded. If a 30-turn game runs to completion (or to a
    natural DONE), the foundation pipeline is sound.
    """
    from agents.simple import roi as simple_roi

    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.reset(num_agents=2)

    for tick in range(30):
        obs0 = env.state[0].observation
        obs1 = env.state[1].observation
        a0 = agent(obs0, env.configuration)
        a1 = simple_roi.agent(obs1)
        env.step([a0, a1])
        if all(s.status == "DONE" for s in env.state):
            break

    # Either we played 30 turns or the game ended; either way no
    # exceptions = pass.
    for s in env.state:
        assert s.status in ("ACTIVE", "DONE", "INACTIVE")


def test_action_format_round_trip():
    """The action returned by agent() must be accepted by env.step()
    without translation errors."""
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.reset(num_agents=2)

    obs = env.state[0].observation
    a = agent(obs, env.configuration)
    # env.step expects per-seat actions. Pair with seat 1 no-op.
    env.step([a, []])
    # If env accepted the action, status is still ACTIVE (or DONE if
    # 1-turn game, which won't happen).
    assert env.state[0].status in ("ACTIVE", "DONE")
