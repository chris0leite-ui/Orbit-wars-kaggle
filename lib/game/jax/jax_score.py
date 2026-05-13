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

from lib.game.jax.jax_interpreter import jax_step, jax_step_jit
from lib.game.jax.jax_world_model import (
    build_world_model, build_world_model_jit, DEFAULT_HORIZON,
)
from lib.game.jax.jax_missions import (
    compute_snipe_score_matrix,
    compute_snipe_score_matrix_jit,
    compute_reinforce_score_matrix,
    compute_reinforce_score_matrix_jit,
    settle_plan_from_matrices,
    merge_class_matrices,
    settle_plan_jax,
)
from lib.game.jax.jax_mechanisms import (
    apply_mechanisms_numpy,
    apply_mechanisms_jax,
    pack_per_agent_actions_jax,
    emitted_to_jax_action_tensors,
)
from lib.game.jax.jax_types import MAX_AGENTS, MAX_LAUNCH_PER_AGENT


def policy_step_jax(
    state, my_id: int, num_agents: int = 2, aggressive: bool = False,
):
    """Compute one agent's per-turn emitted intent list from a JAX state.

    `aggressive=False` (default) mirrors v7_0's self-side (Tier-0 /
    v3_snipe-style sizing). `aggressive=True` mirrors v3.5.1 / Tier-1
    opp policy (top_tier_mirror_policy) — used for the rollout's opp.

    Returns (emitted: list[dict], world_model). The caller composes
    emitted-per-agent lists across both agents and packs to action
    tensors before stepping the env.
    """
    wm = build_world_model_jit(state, max_horizon=DEFAULT_HORIZON, num_agents=4)
    snipe = compute_snipe_score_matrix_jit(
        state, wm, my_id=my_id, aggressive=aggressive, num_agents=num_agents,
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


def rollout_step_jax(
    state, my_id: int, num_agents: int = 2, opp_aggressive: bool = True,
):
    """One full env tick: self + opp policies, action pack, jax_step.

    `opp_aggressive=True` matches v7_0's default Tier-1 opp model
    (top_tier_mirror_policy with aggressive=True snipe sizing).
    """
    my_emit, _ = policy_step_jax(
        state, my_id=my_id, num_agents=num_agents, aggressive=False,
    )
    per_agent = [[] for _ in range(num_agents)]
    per_agent[my_id] = my_emit
    for opp_id in range(num_agents):
        if opp_id == my_id:
            continue
        opp_emit, _ = policy_step_jax(
            state, my_id=opp_id, num_agents=num_agents,
            aggressive=opp_aggressive,
        )
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
    opp_aggressive: bool = True,
):
    """Apply `candidate_emit` as our turn-1 action, simulate K-1 more
    turns of self vs opp self-play, return ship-delta at the end.

    `opp_aggressive=True` matches v7_0's default Tier-1 mirror.
    Returns a Python float (value_delta_ships scalarised).
    """
    per_agent = [[] for _ in range(num_agents)]
    per_agent[my_id] = candidate_emit
    for opp_id in range(num_agents):
        if opp_id == my_id:
            continue
        opp_emit, _ = policy_step_jax(
            state, my_id=opp_id, num_agents=num_agents,
            aggressive=opp_aggressive,
        )
        per_agent[opp_id] = opp_emit
    pids, angles, ships = emitted_to_jax_action_tensors(per_agent, num_agents=num_agents)
    s = jax_step_jit(
        state, jnp.asarray(pids), jnp.asarray(angles), jnp.asarray(ships),
    )
    for _ in range(K - 1):
        if bool(s.done):
            break
        s = rollout_step_jax(
            s, my_id=my_id, num_agents=num_agents,
            opp_aggressive=opp_aggressive,
        )
    return float(value_delta_ships(s, my_id=my_id))


# ---------------------------------------------------------------------------
# Sub-phase 8b: fully-JAX policy + rollout (vmap-able)
# ---------------------------------------------------------------------------


def policy_emit_jax_pure(
    state,
    world_model,
    my_id: int,
    aggressive: bool,
    num_agents: int,
):
    """Pure JAX: state + WorldModel → packed per-agent action tensors.

    No Python control flow on traced values; jit/vmap compatible.
    Returns three `(MAX_LAUNCH_PER_AGENT,)` arrays: pids, angles, ships.
    """
    snipe = compute_snipe_score_matrix(
        state, world_model, my_id=my_id,
        aggressive=aggressive, num_agents=num_agents,
    )
    reinforce = compute_reinforce_score_matrix(state, world_model, my_id=my_id)
    merged = merge_class_matrices([snipe, reinforce])
    src, tgt, ships, eta = settle_plan_jax(
        merged["score"], merged["ships"], merged["eta"], merged["valid"],
        world_model.ships_at,
    )
    final_src, final_angle, final_ships = apply_mechanisms_jax(
        state, world_model, src, tgt, ships, eta, my_id=my_id,
    )
    return pack_per_agent_actions_jax(
        final_src, final_angle, final_ships, state.planets_id,
    )


def rollout_step_jax_pure(
    state,
    my_id: int,
    num_agents: int = 2,
    opp_aggressive: bool = True,
):
    """One env tick, fully JAX (no numpy / no Python control flow).

    Builds the WorldModel once, runs policy_emit_jax_pure for both
    seats, packs the per-agent action tensors, then jax_step.
    """
    wm = build_world_model(state, max_horizon=DEFAULT_HORIZON, num_agents=4)
    pids_my, ang_my, sh_my = policy_emit_jax_pure(
        state, wm, my_id=my_id, aggressive=False, num_agents=num_agents,
    )
    # For 2P games only: opp_id = 1 - my_id.
    opp_id = 1 - my_id
    pids_op, ang_op, sh_op = policy_emit_jax_pure(
        state, wm, my_id=opp_id, aggressive=opp_aggressive,
        num_agents=num_agents,
    )
    # Pack into (MAX_AGENTS, MAX_LAUNCH_PER_AGENT) tensors.
    pids_full = jnp.full((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), -1, dtype=jnp.int32)
    ang_full = jnp.zeros((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=jnp.float32)
    sh_full = jnp.zeros((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=jnp.int32)
    pids_full = pids_full.at[my_id].set(pids_my)
    pids_full = pids_full.at[opp_id].set(pids_op)
    ang_full = ang_full.at[my_id].set(ang_my)
    ang_full = ang_full.at[opp_id].set(ang_op)
    sh_full = sh_full.at[my_id].set(sh_my)
    sh_full = sh_full.at[opp_id].set(sh_op)
    return jax_step(state, pids_full, ang_full, sh_full)


def score_candidate_jax_pure(
    state,
    K: int,
    my_id: int,
    num_agents: int = 2,
    opp_aggressive: bool = True,
):
    """Fully-JAX K-step rollout from `state` returning ship-delta.

    Self plays its own policy (not a candidate override — for the
    drop-one chooser, the caller mutates state's planets_ships /
    settle_plan output before calling this). Mostly used as the
    vmap-able primitive for the GPU A/B path.
    """
    def step_fn(s, _):
        new_s = rollout_step_jax_pure(
            s, my_id=my_id, num_agents=num_agents,
            opp_aggressive=opp_aggressive,
        )
        return new_s, None

    final_s, _ = jax.lax.scan(step_fn, state, None, length=K)
    return value_delta_ships(final_s, my_id=my_id)


# JIT-compile entry points.
rollout_step_jax_pure_jit = jax.jit(
    rollout_step_jax_pure, static_argnames=("my_id", "num_agents", "opp_aggressive"),
)
score_candidate_jax_pure_jit = jax.jit(
    score_candidate_jax_pure,
    static_argnames=("K", "my_id", "num_agents", "opp_aggressive"),
)
