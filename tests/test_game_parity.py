"""Shadow-parity harness for lib.game.interpreter vs kaggle_environments.

Drives our interpreter and Kaggle's on identical (seed, action_sequence)
pairs and asserts byte-exact state match after every step. Tolerance is
zero — any divergence is a real bug.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Callable

import pytest

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    interpreter as kaggle_interpreter,
)
from kaggle_environments.utils import Struct

from lib.fast_sim import _FakeEnv
from lib.game.interpreter import interpreter as ours_interpreter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_state_and_env(num_agents: int, seed: int):
    """Build a fresh `(state, fake_env)` pair for an uninitialized game.

    Matches the shape kaggle_environments creates internally before the
    first interpreter call (one Struct per seat, each with an empty
    observation Struct).
    """
    state = []
    for i in range(num_agents):
        s = Struct(
            observation=Struct(),
            action=[],
            status="ACTIVE",
            reward=0,
            info={},
        )
        state.append(s)
    configuration = Struct(
        episodeSteps=500,
        shipSpeed=6.0,
        cometSpeed=4.0,
        actTimeout=1.0,
        agentTimeout=60.0,
        runTimeout=1200.0,
        seed=seed,
    )
    fake_env = _FakeEnv(configuration=configuration, episode_seed=seed)
    fake_env.info = {}  # interpreter populates info["seed"] in init
    fake_env.done = False
    return state, fake_env


def _bookkeeping(state, fake_env):
    """Per-step counter + done propagation that env.step would normally do."""
    obs0 = state[0].observation
    new_step = int(obs0.get("step", 0)) + 1
    obs0.step = new_step
    for i in range(1, len(state)):
        state[i].observation.step = new_step
    if any(s.status == "DONE" for s in state):
        fake_env.done = True


def _random_actions(state, rng: random.Random, num_agents: int):
    """Random per-seat actions matching the orbit_wars action schema."""
    actions = []
    for i in range(num_agents):
        obs = state[i].observation
        moves = []
        for p in obs.planets:
            if p[1] == i and p[5] > 0:
                if rng.random() < 0.4:  # 40% chance to launch
                    angle = rng.uniform(0, 2 * math.pi)
                    ships = max(1, int(p[5] * rng.uniform(0.1, 0.7)))
                    if 0 < ships <= p[5]:
                        moves.append([p[0], angle, ships])
        actions.append(moves)
    return actions


def _state_diff(state_a, state_b) -> str:
    """Return empty string if state_a == state_b (byte-exact); otherwise a
    one-line diff describing the first divergence found."""
    if len(state_a) != len(state_b):
        return f"len mismatch: {len(state_a)} vs {len(state_b)}"

    for i in range(len(state_a)):
        sa, sb = state_a[i], state_b[i]
        if sa.status != sb.status:
            return f"seat[{i}].status: {sa.status!r} vs {sb.status!r}"
        if sa.reward != sb.reward:
            return f"seat[{i}].reward: {sa.reward} vs {sb.reward}"

    obs_a = state_a[0].observation
    obs_b = state_b[0].observation

    if obs_a.planets != obs_b.planets:
        for j, (pa, pb) in enumerate(zip(obs_a.planets, obs_b.planets)):
            if pa != pb:
                return f"planets[{j}]: {pa} vs {pb}"
        return f"planets length: {len(obs_a.planets)} vs {len(obs_b.planets)}"

    if obs_a.fleets != obs_b.fleets:
        for j, (fa, fb) in enumerate(zip(obs_a.fleets, obs_b.fleets)):
            if fa != fb:
                return f"fleets[{j}]: {fa} vs {fb}"
        return f"fleets length: {len(obs_a.fleets)} vs {len(obs_b.fleets)}"

    if list(obs_a.comet_planet_ids) != list(obs_b.comet_planet_ids):
        return (
            f"comet_planet_ids: {list(obs_a.comet_planet_ids)} vs "
            f"{list(obs_b.comet_planet_ids)}"
        )

    if len(obs_a.comets) != len(obs_b.comets):
        return f"comets length: {len(obs_a.comets)} vs {len(obs_b.comets)}"

    for j, (ga, gb) in enumerate(zip(obs_a.comets, obs_b.comets)):
        if ga["path_index"] != gb["path_index"]:
            return f"comets[{j}].path_index: {ga['path_index']} vs {gb['path_index']}"
        if ga["planet_ids"] != gb["planet_ids"]:
            return f"comets[{j}].planet_ids: {ga['planet_ids']} vs {gb['planet_ids']}"
        if ga["paths"] != gb["paths"]:
            return f"comets[{j}].paths diverged"

    return ""


# ---------------------------------------------------------------------------
# Init parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 137, 999, 2026, 31415])
@pytest.mark.parametrize("num_agents", [2, 4])
def test_init_parity(seed, num_agents):
    """Both interpreters in init mode produce identical state for the same seed."""
    state_k, env_k = _fresh_state_and_env(num_agents, seed)
    state_o, env_o = _fresh_state_and_env(num_agents, seed)

    kaggle_interpreter(state_k, env_k)
    ours_interpreter(state_o, env_o)

    diff = _state_diff(state_k, state_o)
    assert not diff, f"init parity broken (seed={seed}, n={num_agents}): {diff}"
    assert env_k.info["seed"] == env_o.info["seed"], "info[seed] mismatch"


# ---------------------------------------------------------------------------
# Per-step parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 137])
@pytest.mark.parametrize("num_agents", [2, 4])
def test_shadow_parity_60_steps(seed, num_agents):
    """60 steps random-policy — stays under step-50 comet spawn boundary
    plus a few steps to cover the first spawn."""
    _run_shadow_parity(seed, num_agents, num_steps=60)


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("num_agents", [2, 4])
@pytest.mark.slow
def test_shadow_parity_500_steps(seed, num_agents):
    """Full 500-step episode random-policy — exercises all 5 comet spawn
    boundaries (50, 150, 250, 350, 450) and termination."""
    _run_shadow_parity(seed, num_agents, num_steps=500)


def _run_shadow_parity(seed: int, num_agents: int, num_steps: int):
    # Init both
    state_k, env_k = _fresh_state_and_env(num_agents, seed)
    state_o, env_o = _fresh_state_and_env(num_agents, seed)
    kaggle_interpreter(state_k, env_k)
    ours_interpreter(state_o, env_o)

    diff = _state_diff(state_k, state_o)
    assert not diff, f"init divergence (seed={seed}): {diff}"

    action_rng = random.Random(seed * 7919 + 1)

    for step_idx in range(num_steps):
        # Generate actions from one observation (both states are identical
        # post-init); use the kaggle-side state as the source of truth.
        actions = _random_actions(state_k, action_rng, num_agents)
        actions_copy = copy.deepcopy(actions)

        for i, a in enumerate(actions):
            state_k[i].action = a
        for i, a in enumerate(actions_copy):
            state_o[i].action = a

        kaggle_interpreter(state_k, env_k)
        ours_interpreter(state_o, env_o)

        _bookkeeping(state_k, env_k)
        _bookkeeping(state_o, env_o)

        diff = _state_diff(state_k, state_o)
        assert not diff, (
            f"divergence at step {step_idx} (seed={seed}, n={num_agents}): {diff}"
        )

        if env_k.done or env_o.done:
            assert env_k.done == env_o.done, (
                f"done mismatch at step {step_idx}: kaggle={env_k.done} ours={env_o.done}"
            )
            break
