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


# ---------------------------------------------------------------------------
# Planet-position cache HIT parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 137])
@pytest.mark.parametrize("num_agents", [2, 4])
def test_planet_position_cache_hit_parity(seed, num_agents):
    """A state with a populated planet_position_cache (the from_obs path)
    must produce byte-identical step output to a state with an empty
    cache (the fresh-init path). This validates the interpreter's
    cache-vs-fallthrough branch.
    """
    import math as _math
    from lib.game.interpreter import (
        BOARD_SIZE as _BS, CENTER as _CTR,
        ROTATION_RADIUS_LIMIT as _RRL,
    )

    state_a, env_a = _fresh_state_and_env(num_agents, seed)
    state_b, env_b = _fresh_state_and_env(num_agents, seed)
    ours_interpreter(state_a, env_a)
    ours_interpreter(state_b, env_b)

    # Populate state B's planet_position_cache from initial_planets,
    # exactly mimicking lib/fast_sim._populate_planet_position_cache.
    obs0_b = state_b[0].observation
    angular_velocity = float(obs0_b.angular_velocity)
    episode_steps = 500
    comet_pid_set = set(obs0_b.comet_planet_ids)
    for ip in obs0_b.initial_planets:
        pid = ip[0]
        if pid in comet_pid_set:
            continue
        dx = ip[2] - _CTR
        dy = ip[3] - _CTR
        r = _math.sqrt(dx * dx + dy * dy)
        if r + ip[4] >= _RRL:
            continue
        initial_angle = _math.atan2(dy, dx)
        positions = []
        for s in range(episode_steps + 1):
            theta = initial_angle + angular_velocity * s
            positions.append(
                (_CTR + r * _math.cos(theta), _CTR + r * _math.sin(theta))
            )
        env_b.planet_position_cache[pid] = positions

    # Now step both with identical empty actions and verify byte-exact parity.
    action_rng = random.Random(seed * 65537 + 3)
    for step_idx in range(80):
        actions = _random_actions(state_a, action_rng, num_agents)
        for i, a in enumerate(actions):
            state_a[i].action = a
            state_b[i].action = list(a)
        ours_interpreter(state_a, env_a)
        ours_interpreter(state_b, env_b)
        _bookkeeping(state_a, env_a)
        _bookkeeping(state_b, env_b)
        diff = _state_diff(state_a, state_b)
        assert not diff, (
            f"planet-cache-hit divergence at step {step_idx} "
            f"(seed={seed}, n={num_agents}): {diff}"
        )
        if env_a.done or env_b.done:
            break


# ---------------------------------------------------------------------------
# Comet-path cache HIT parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 137])
@pytest.mark.parametrize("num_agents", [2, 4])
def test_comet_cache_hit_parity(seed, num_agents):
    """Two parallel game state branches sharing one comet_path_cache must
    produce byte-identical state across a spawn boundary — even though
    only one of them does the expensive generate_comet_paths call.

    This validates that the cached value is correctly reused on cache hit.
    """
    import copy as _copy

    # Build state A and state B with a SHARED cache dict
    state_a, env_a = _fresh_state_and_env(num_agents, seed)
    state_b, env_b = _fresh_state_and_env(num_agents, seed)
    shared_cache = {}
    env_a.comet_path_cache = shared_cache
    env_b.comet_path_cache = shared_cache

    # Init both (no comet spawn here; step 0)
    ours_interpreter(state_a, env_a)
    ours_interpreter(state_b, env_b)

    # Fast-forward both to step 48 with empty actions
    for _ in range(49):
        for s in state_a + state_b:
            s.action = []
        ours_interpreter(state_a, env_a)
        ours_interpreter(state_b, env_b)
        _bookkeeping(state_a, env_a)
        _bookkeeping(state_b, env_b)
        if env_a.done or env_b.done:
            pytest.skip("episode ended before reaching spawn boundary")

    # Both states should still be identical (no actions, deterministic init)
    diff = _state_diff(state_a, state_b)
    assert not diff, f"pre-spawn divergence (seed={seed}): {diff}"
    assert len(shared_cache) == 0, "cache should be empty before first spawn"

    # Cross step 49→50: state A computes (cache miss → store), state B then reads (hit)
    for s in state_a:
        s.action = []
    ours_interpreter(state_a, env_a)
    _bookkeeping(state_a, env_a)
    assert len(shared_cache) == 1, "cache miss should have populated"

    for s in state_b:
        s.action = []
    ours_interpreter(state_b, env_b)
    _bookkeeping(state_b, env_b)

    diff = _state_diff(state_a, state_b)
    assert not diff, (
        f"cache-hit divergence at spawn (seed={seed}, n={num_agents}): {diff}"
    )


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
