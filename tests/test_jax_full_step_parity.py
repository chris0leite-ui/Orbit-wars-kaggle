"""End-to-end parity: full `jax_step_jit` vs scalar interpreter.

Runs N seeds × K random-policy steps through both implementations,
compares state-by-state. The agents/missions/mechanism stacks aren't
ported yet (sub-phases 2-6), so we use empty actions or random raw
launches and rely on the engine alone.

Comparison strategy: compare PLANETS BY PID and FLEETS BY ID (the
canonical identity) — JAX may pack them into different slot indices
than scalar after captures/comets, so slot-based comparison would
falsely diverge.

Float tolerance: 1e-3 absolute for positions (JAX float32 + jit
fma may shift by an ULP). Combat outcomes (owner, ships, alive
counts) must be EXACT.
"""

from __future__ import annotations

import math
import random
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    interpreter as scalar_interpreter,
)
from kaggle_environments.utils import Struct

from lib.fast_sim import _FakeEnv
from lib.game.jax import (
    GameState, scalar_to_jax, actions_to_jax, jax_step_jit,
)


def _make_paired_states(seed: int, num_agents: int = 2):
    """Build a scalar env state AND its JAX twin from the same seed."""
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=num_agents)
    gs = scalar_to_jax(env.state, env.info["seed"])
    return env, gs


def _rand_actions_for_scalar(state, rng: random.Random, num_agents: int):
    """Per-agent random actions from a scalar state (list of [pid, angle, ships])."""
    obs = state[0].observation
    actions_per_agent = []
    for pid_seat in range(num_agents):
        moves = []
        for p in obs.planets:
            if p[1] == pid_seat and p[5] > 0 and rng.random() < 0.4:
                angle = rng.uniform(0, 2 * math.pi)
                ships = max(1, int(p[5] * rng.uniform(0.1, 0.7)))
                if 0 < ships <= p[5]:
                    moves.append([p[0], angle, ships])
        actions_per_agent.append(moves)
    return actions_per_agent


def _compare_planets(scalar_obs, gs: GameState, step_idx: int, seed: int):
    """Compare scalar obs.planets to JAX state by PID."""
    scalar_planets = {int(p[0]): tuple(p) for p in scalar_obs.planets}
    jax_alive = np.asarray(gs.planets_alive)
    jax_ids = np.asarray(gs.planets_id)
    jax_planets = {}
    for i in range(len(jax_alive)):
        if not jax_alive[i]:
            continue
        pid = int(jax_ids[i])
        jax_planets[pid] = (
            pid,
            int(gs.planets_owner[i]),
            float(gs.planets_x[i]),
            float(gs.planets_y[i]),
            float(gs.planets_radius[i]),
            int(gs.planets_ships[i]),
            int(gs.planets_prod[i]),
        )

    # PID sets should match.
    scalar_pids = set(scalar_planets.keys())
    jax_pids = set(jax_planets.keys())
    if scalar_pids != jax_pids:
        missing = scalar_pids - jax_pids
        extra = jax_pids - scalar_pids
        return (
            f"seed={seed} step={step_idx} planet PIDs differ: "
            f"scalar-only={sorted(missing)[:5]} jax-only={sorted(extra)[:5]}"
        )

    # Per-pid field comparison.
    for pid in scalar_pids:
        sp = scalar_planets[pid]
        jp = jax_planets[pid]
        # owner, ships, prod, radius — exact
        if sp[1] != jp[1]:
            return f"seed={seed} step={step_idx} pid={pid} owner: {sp[1]} vs {jp[1]}"
        if sp[5] != jp[5]:
            return f"seed={seed} step={step_idx} pid={pid} ships: {sp[5]} vs {jp[5]}"
        if sp[6] != jp[6]:
            return f"seed={seed} step={step_idx} pid={pid} prod: {sp[6]} vs {jp[6]}"
        if abs(sp[4] - jp[4]) > 1e-4:
            return f"seed={seed} step={step_idx} pid={pid} radius: {sp[4]} vs {jp[4]}"
        # x, y — 1e-3 tolerance (float32 + fma)
        if abs(sp[2] - jp[2]) > 1e-3:
            return f"seed={seed} step={step_idx} pid={pid} x: {sp[2]} vs {jp[2]}"
        if abs(sp[3] - jp[3]) > 1e-3:
            return f"seed={seed} step={step_idx} pid={pid} y: {sp[3]} vs {jp[3]}"
    return ""


def _compare_fleets(scalar_obs, gs: GameState, step_idx: int, seed: int):
    """Compare scalar obs.fleets to JAX state by fleet ID."""
    scalar_fleets = {int(f[0]): tuple(f) for f in scalar_obs.fleets}
    jax_alive = np.asarray(gs.fleets_alive)
    jax_ids = np.asarray(gs.fleets_id)
    jax_fleets = {}
    for i in range(len(jax_alive)):
        if not jax_alive[i]:
            continue
        fid = int(jax_ids[i])
        jax_fleets[fid] = (
            fid,
            int(gs.fleets_owner[i]),
            float(gs.fleets_x[i]),
            float(gs.fleets_y[i]),
            float(gs.fleets_angle[i]),
            int(gs.fleets_from_planet[i]),
            int(gs.fleets_ships[i]),
        )
    scalar_ids = set(scalar_fleets.keys())
    jax_ids_set = set(jax_fleets.keys())
    if scalar_ids != jax_ids_set:
        missing = scalar_ids - jax_ids_set
        extra = jax_ids_set - scalar_ids
        return (
            f"seed={seed} step={step_idx} fleet IDs differ: "
            f"scalar-only={sorted(missing)[:5]} jax-only={sorted(extra)[:5]}"
        )
    for fid in scalar_ids:
        sf = scalar_fleets[fid]
        jf = jax_fleets[fid]
        if sf[1] != jf[1]:
            return f"seed={seed} step={step_idx} fleet_id={fid} owner: {sf[1]} vs {jf[1]}"
        if sf[6] != jf[6]:
            return f"seed={seed} step={step_idx} fleet_id={fid} ships: {sf[6]} vs {jf[6]}"
        if sf[5] != jf[5]:
            return f"seed={seed} step={step_idx} fleet_id={fid} from_planet: {sf[5]} vs {jf[5]}"
        if abs(sf[4] - jf[4]) > 1e-4:
            return f"seed={seed} step={step_idx} fleet_id={fid} angle: {sf[4]} vs {jf[4]}"
        if abs(sf[2] - jf[2]) > 1e-3:
            return f"seed={seed} step={step_idx} fleet_id={fid} x: {sf[2]} vs {jf[2]}"
        if abs(sf[3] - jf[3]) > 1e-3:
            return f"seed={seed} step={step_idx} fleet_id={fid} y: {sf[3]} vs {jf[3]}"
    return ""


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 137])
def test_jax_step_vs_scalar_60_random(seed):
    """JAX `jax_step_jit` matches scalar interpreter over 60 random
    steps. Validates the full chain end-to-end."""
    num_agents = 2
    env, gs = _make_paired_states(seed, num_agents)

    # Sanity: initial states should match.
    diff = _compare_planets(env.state[0].observation, gs, -1, seed)
    assert not diff, f"init planet divergence: {diff}"

    rng = random.Random(seed * 7919 + 11)
    for step_idx in range(60):
        if env.state[0].status != "ACTIVE":
            break
        actions = _rand_actions_for_scalar(env.state, rng, num_agents)

        # Scalar: copy actions onto state, run interpreter, bump step.
        for i, a in enumerate(actions):
            env.state[i].action = a
        scalar_interpreter(env.state, env)
        # Bookkeeping: env.step doesn't run interpreter directly here;
        # we manually advance the obs.step counter (mirror what
        # `_bookkeep` does in earlier parity tests).
        obs0 = env.state[0].observation
        new_step = int(obs0.get("step", 0)) + 1
        obs0.step = new_step
        for i in range(1, num_agents):
            env.state[i].observation.step = new_step

        # JAX: convert actions, jit'd step.
        pid_arr, ang_arr, ship_arr = actions_to_jax(actions, num_agents)
        gs = jax_step_jit(gs, pid_arr, ang_arr, ship_arr)

        # Compare.
        diff = _compare_planets(env.state[0].observation, gs, step_idx, seed)
        assert not diff, diff
        diff = _compare_fleets(env.state[0].observation, gs, step_idx, seed)
        assert not diff, diff
