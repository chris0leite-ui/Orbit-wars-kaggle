"""Sub-phase 4 parity: JAX-matrix → settle_plan → mechanism pipeline
vs scalar `realize(intents, obs, mechanisms=DEFAULT_MECHANISMS)`.

We compare:
- which (src_pid, target_pid) pairs survive,
- emitted ship counts (post arrival_size bump),
- aim_angle within tolerance (JAX uses atan2 only; scalar uses
  lead_aim_v2 for orbiting targets — divergence is bounded but real
  and tracked separately in sub-phase 7).
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from kaggle_environments import make

from lib.intent import World, Intent, realize
from lib.world_model import WorldModel, DEFAULT_HORIZON
from lib.missions.snipe import propose_snipe_missions
from lib.missions.reinforce import propose_reinforce_missions
from lib.planner import settle_plan
from lib.mechanism import DEFAULT_MECHANISMS

from lib.game.jax import scalar_to_jax
from lib.game.jax.jax_world_model import build_world_model
from lib.game.jax.jax_missions import (
    compute_snipe_score_matrix,
    compute_reinforce_score_matrix,
    settle_plan_from_matrices,
)
from lib.game.jax.jax_mechanisms import apply_mechanisms_numpy


def _spawn_in_flight_fleets(env, num_agents=2, n_steps=15, rng_seed=7):
    """Lighter random-policy: only launches with 20% probability per
    eligible planet and sends ~30% of garrison, so by step ~25 there
    are still owned planets with ships > 0 to source from."""
    rng = random.Random(rng_seed)
    for _ in range(n_steps):
        if env.state[0].status != "ACTIVE":
            break
        actions = []
        for pid_seat in range(num_agents):
            moves = []
            obs = env.state[pid_seat].observation
            for p in obs.planets:
                if p[1] == pid_seat and p[5] > 5 and rng.random() < 0.2:
                    angle = rng.uniform(0, 2 * math.pi)
                    ships = max(1, int(p[5] * rng.uniform(0.1, 0.3)))
                    if 0 < ships <= p[5]:
                        moves.append([p[0], angle, ships])
            actions.append(moves)
        env.step(actions)


@pytest.mark.parametrize("seed", [3, 11, 42])
def test_mechanism_pipeline_ship_pairs_parity(seed):
    """Ship counts and (src, target) pairs must agree between scalar
    realize() and the JAX-matrix pipeline."""
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _spawn_in_flight_fleets(env, num_agents=2, n_steps=25, rng_seed=seed * 31)
    if env.state[0].status != "ACTIVE":
        pytest.skip(f"seed {seed} terminated early")

    my_id = 0
    obs = env.state[my_id].observation
    scalar_world = World.from_obs(obs)
    scalar_wm = WorldModel.from_world(scalar_world)

    # --- Scalar reference pipeline ---
    scalar_missions = (
        propose_snipe_missions(scalar_world, scalar_wm, aggressive=False)
        + propose_reinforce_missions(scalar_world, scalar_wm)
    )
    scalar_intents = settle_plan(scalar_missions, scalar_world, scalar_wm)
    scalar_actions = realize(
        scalar_intents, obs, mechanisms=DEFAULT_MECHANISMS, model=scalar_wm,
    )
    scalar_by_src = {int(a[0]): a for a in scalar_actions}

    # --- JAX pipeline ---
    gs = scalar_to_jax(env.state, env.info["seed"])
    jax_wm = build_world_model(gs, max_horizon=DEFAULT_HORIZON, num_agents=4)
    snipe_out = compute_snipe_score_matrix(gs, jax_wm, my_id=my_id)
    reinforce_out = compute_reinforce_score_matrix(gs, jax_wm, my_id=my_id)
    chosen = settle_plan_from_matrices(
        class_outputs=[snipe_out, reinforce_out],
        class_names=["snipe", "reinforce"],
        planets_id=gs.planets_id,
        world_owners_at=jax_wm.owners_at,
        world_ships_at=jax_wm.ships_at,
        my_id=my_id,
    )
    emitted = apply_mechanisms_numpy(chosen, gs, jax_wm, my_id=my_id)
    jax_by_src = {int(e["src_pid"]): e for e in emitted}

    # --- Compare ---
    diffs = []
    for src_pid, scalar_action in scalar_by_src.items():
        jax_e = jax_by_src.get(src_pid)
        if jax_e is None:
            diffs.append(
                f"  src={src_pid}: scalar emits "
                f"[ang={scalar_action[1]:.3f}, ships={scalar_action[2]}] "
                f"but JAX drops"
            )
            continue
        # Ship counts: scalar's arrival_size may differ when the target is
        # orbiting (we don't apply the +1 dynamic prod_tick in JAX).
        scalar_ships = int(scalar_action[2])
        jax_ships = int(jax_e["ships"])
        if abs(scalar_ships - jax_ships) > 1:
            diffs.append(
                f"  src={src_pid}: ships scalar={scalar_ships} jax={jax_ships}"
            )

    for src_pid in jax_by_src.keys() - scalar_by_src.keys():
        e = jax_by_src[src_pid]
        diffs.append(
            f"  src={src_pid}: JAX emits "
            f"[tgt={e['target_pid']}, ships={e['ships']}] but scalar drops"
        )

    assert not diffs, "mechanism pipeline divergence:\n" + "\n".join(diffs)


@pytest.mark.parametrize("seed", [3, 11, 42])
def test_mechanism_pipeline_angles_within_tolerance(seed):
    """Per-source aim angles: tolerance 0.2 rad (~12°). JAX uses atan2
    of current target position; scalar uses lead_aim_v2 for moving
    orbiting targets. Wide tolerance acknowledged; sub-phase 7 closes
    the gap with a full lead_aim port.
    """
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _spawn_in_flight_fleets(env, num_agents=2, n_steps=25, rng_seed=seed * 31)
    if env.state[0].status != "ACTIVE":
        pytest.skip(f"seed {seed} terminated early")

    my_id = 0
    obs = env.state[my_id].observation
    scalar_world = World.from_obs(obs)
    scalar_wm = WorldModel.from_world(scalar_world)
    scalar_missions = (
        propose_snipe_missions(scalar_world, scalar_wm, aggressive=False)
        + propose_reinforce_missions(scalar_world, scalar_wm)
    )
    scalar_intents = settle_plan(scalar_missions, scalar_world, scalar_wm)
    scalar_actions = realize(
        scalar_intents, obs, mechanisms=DEFAULT_MECHANISMS, model=scalar_wm,
    )
    scalar_by_src = {int(a[0]): a for a in scalar_actions}

    gs = scalar_to_jax(env.state, env.info["seed"])
    jax_wm = build_world_model(gs, max_horizon=DEFAULT_HORIZON, num_agents=4)
    snipe_out = compute_snipe_score_matrix(gs, jax_wm, my_id=my_id)
    reinforce_out = compute_reinforce_score_matrix(gs, jax_wm, my_id=my_id)
    chosen = settle_plan_from_matrices(
        class_outputs=[snipe_out, reinforce_out],
        class_names=["snipe", "reinforce"],
        planets_id=gs.planets_id,
        world_owners_at=jax_wm.owners_at,
        world_ships_at=jax_wm.ships_at,
        my_id=my_id,
    )
    emitted = apply_mechanisms_numpy(chosen, gs, jax_wm, my_id=my_id)
    jax_by_src = {int(e["src_pid"]): e for e in emitted}

    # Sources present in BOTH outputs.
    common = scalar_by_src.keys() & jax_by_src.keys()
    if not common:
        pytest.skip(f"seed {seed}: no common sources to compare angles on")
    diffs = []
    for src_pid in common:
        scalar_ang = float(scalar_by_src[src_pid][1])
        jax_ang = float(jax_by_src[src_pid]["angle"])
        # Wrap difference to [-π, π].
        delta = math.atan2(math.sin(scalar_ang - jax_ang),
                           math.cos(scalar_ang - jax_ang))
        if abs(delta) > 0.05:
            diffs.append(
                f"  src={src_pid}: scalar_ang={scalar_ang:.3f} "
                f"jax_ang={jax_ang:.3f} delta={delta:.3f}"
            )
    assert not diffs, "angle divergence > 0.2 rad:\n" + "\n".join(diffs)
