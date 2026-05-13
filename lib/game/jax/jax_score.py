"""JAX-driven K-step rollout for candidate scoring.

Sub-phase 5 of the JAX sprint. Composes the per-step pipeline:

    state → build_world_model → score matrices (snipe + reinforce) →
    settle_plan (numpy) → mechanism stack (numpy) → action tensors →
    jax_step → next state.

Opp policy = same pipeline applied to opponent's seat. Tier-1 (v3.5.1)
mirror is the v7_0_drop_one default opp.

Performance notes:
- WorldModel + score-matrix calls are jit'd; their hot-path cost is
  amortised across rollout depth.
- settle_plan + apply_mechanisms_numpy run in pure numpy/Python. For
  N=1 (one game) this is fine; for vmap'd N=64 we'd need a JAX-scan
  port of both, which is sub-phase 7's perf-tune scope.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from lib.game.jax.jax_interpreter import jax_step_jit
from lib.game.jax.jax_world_model import build_world_model_jit, DEFAULT_HORIZON
from lib.game.jax.jax_missions import (
    compute_snipe_score_matrix_jit,
    compute_reinforce_score_matrix_jit,
    settle_plan_from_matrices,
)
from lib.game.jax.jax_mechanisms import (
    apply_mechanisms_numpy,
    emitted_to_jax_action_tensors,
)


def policy_step_jax(state, my_id: int, num_agents: int = 2):
    """Compute one agent's per-turn emitted intent list from a JAX state.

    Returns (emitted: list[dict], world_model). The caller composes
    emitted-per-agent lists across both agents and packs to action
    tensors before stepping the env.
    """
    wm = build_world_model_jit(state, max_horizon=DEFAULT_HORIZON, num_agents=4)
    snipe = compute_snipe_score_matrix_jit(
        state, wm, my_id=my_id, aggressive=False, num_agents=num_agents,
    )
    reinforce = compute_reinforce_score_matrix_jit(state, wm, my_id=my_id)
    chosen = settle_plan_from_matrices(
        class_outputs=[snipe, reinforce],
        class_names=["snipe", "reinforce"],
        planets_id=state.planets_id,
        world_owners_at=wm.owners_at,
        world_ships_at=wm.ships_at,
        my_id=my_id,
    )
    emitted = apply_mechanisms_numpy(chosen, state, wm, my_id=my_id)
    return emitted, wm


def rollout_step_jax(state, my_id: int, num_agents: int = 2):
    """One full env tick: self + opp policies, action pack, jax_step."""
    my_emit, _ = policy_step_jax(state, my_id=my_id, num_agents=num_agents)
    per_agent = [[] for _ in range(num_agents)]
    per_agent[my_id] = my_emit
    for opp_id in range(num_agents):
        if opp_id == my_id:
            continue
        opp_emit, _ = policy_step_jax(state, my_id=opp_id, num_agents=num_agents)
        per_agent[opp_id] = opp_emit
    pids, angles, ships = emitted_to_jax_action_tensors(per_agent, num_agents=num_agents)
    new_state = jax_step_jit(
        state, jnp.asarray(pids), jnp.asarray(angles), jnp.asarray(ships),
    )
    return new_state


def value_delta_ships(state, my_id: int):
    """Total `my_id`'s ships (planets + alive fleets) minus opponents'.

    Mirrors `lib.value_heads.delta_us_minus_them_obs`. Neutral (owner=-1)
    is excluded from both sides.
    """
    my_id_jnp = jnp.int32(my_id)
    planet_my = jnp.sum(
        jnp.where(
            (state.planets_owner == my_id_jnp) & state.planets_alive,
            state.planets_ships, jnp.int32(0),
        )
    )
    fleet_my = jnp.sum(
        jnp.where(
            (state.fleets_owner == my_id_jnp) & state.fleets_alive,
            state.fleets_ships, jnp.int32(0),
        )
    )
    planet_opp = jnp.sum(
        jnp.where(
            (state.planets_owner != my_id_jnp)
            & (state.planets_owner != jnp.int32(-1))
            & state.planets_alive,
            state.planets_ships, jnp.int32(0),
        )
    )
    fleet_opp = jnp.sum(
        jnp.where(
            (state.fleets_owner != my_id_jnp)
            & (state.fleets_owner != jnp.int32(-1))
            & state.fleets_alive,
            state.fleets_ships, jnp.int32(0),
        )
    )
    return (planet_my + fleet_my) - (planet_opp + fleet_opp)


def score_candidate_jax(
    state,
    candidate_emit: list[dict],          # output of apply_mechanisms_numpy
    K: int,
    my_id: int,
    num_agents: int = 2,
):
    """Apply `candidate_emit` as our turn-1 action, simulate K-1 more
    turns of self vs opp self-play, return ship-delta at the end.

    Returns a Python float (value_delta_ships scalarised).
    """
    # Step 1: apply the candidate's action together with opp's policy step.
    opp_emit_all = []
    per_agent = [[] for _ in range(num_agents)]
    per_agent[my_id] = candidate_emit
    for opp_id in range(num_agents):
        if opp_id == my_id:
            continue
        opp_emit, _ = policy_step_jax(state, my_id=opp_id, num_agents=num_agents)
        per_agent[opp_id] = opp_emit
        opp_emit_all.append(opp_emit)
    pids, angles, ships = emitted_to_jax_action_tensors(per_agent, num_agents=num_agents)
    s = jax_step_jit(
        state, jnp.asarray(pids), jnp.asarray(angles), jnp.asarray(ships),
    )
    # Steps 2..K: self vs opp rollout.
    for _ in range(K - 1):
        if bool(s.done):
            break
        s = rollout_step_jax(s, my_id=my_id, num_agents=num_agents)
    return float(value_delta_ships(s, my_id=my_id))
