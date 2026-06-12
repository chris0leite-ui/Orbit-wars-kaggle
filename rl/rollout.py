"""Vectorised self-play rollout: policy actions -> engine launch tensors,
vmapped jax_step, potential-shaped rewards, auto-reset from an init pool.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from lib.game.jax.jax_interpreter import jax_step
from lib.game.jax.jax_types import (
    GameState, MAX_AGENTS, MAX_LAUNCH_PER_AGENT, MAX_PLANETS,
)
from rl import net
from rl.aim import solve_intercept_rows
from rl.features import FRACS, material_potential, seat_features, state_tables

GAMMA = 0.999
SHAPING_COEF = 1.0


def actions_to_launch_tensors(state: GameState, tgt, frac, src_mask):
    """Map per-seat per-planet (target, fraction) to engine launch tensors.

    tgt  (A, P) int32 — chosen target slot, P = hold
    frac (A, P) int32 — fraction index into FRACS
    src_mask (A, P) bool — launch-eligible sources for that seat

    Returns (pids, angles, ships) each (A, MAX_LAUNCH_PER_AGENT).
    Top-`MAX_LAUNCH_PER_AGENT` launches per seat by ship count.
    """
    garrison = state.planets_ships.astype(jnp.float32)  # (P,)

    def per_seat(tgt_s, frac_s, mask_s):
        is_launch = (tgt_s < MAX_PLANETS) & mask_s
        frac_val = FRACS[frac_s]
        ships = jnp.floor(frac_val * garrison)
        ships = jnp.clip(ships, 1.0, garrison).astype(jnp.int32)
        ships = jnp.where(is_launch & (garrison >= 1.0), ships, 0)
        is_launch = is_launch & (ships > 0)

        angle = solve_intercept_rows(state, tgt_s, ships)  # (P,)

        # Rank launching sources by ships desc, take top slots.
        key = jnp.where(is_launch, ships, -1)
        order = jnp.argsort(-key)  # descending
        top = order[:MAX_LAUNCH_PER_AGENT]
        top_launch = is_launch[top]
        pids = jnp.where(top_launch, state.planets_id[top], -1)
        angles = jnp.where(top_launch, angle[top], 0.0)
        ships_out = jnp.where(top_launch, ships[top], 0)
        return pids, angles, ships_out

    return jax.vmap(per_seat)(tgt, frac, src_mask)


def policy_act(key, params, state: GameState):
    """Sample actions for all 4 seats of ONE env. Returns action pytree."""
    tables = state_tables(state)
    seat_keys = jax.random.split(key, MAX_AGENTS)

    def per_seat(seat, k):
        nodes, edges, globals_, src_mask, tgt_mask = seat_features(
            state, tables, seat)
        out = net.sample_actions(
            k, params, nodes, edges, globals_, state.planets_alive,
            src_mask, tgt_mask)
        return out, src_mask

    outs, src_masks = jax.vmap(per_seat)(jnp.arange(MAX_AGENTS), seat_keys)
    return outs, src_masks  # leaves have leading (A,) axis


def policy_act_vs_opp(key, params, opp_params, learner_seat,
                      state: GameState, opp_kind: str):
    """Like policy_act, but seats != learner_seat act under a frozen
    opponent: `opp_kind` = "net" (opp_params) or a scripted policy name
    from rl.scripted.SCRIPTED_POLICIES.

    Learner-seat outputs are the only on-policy ones; the caller masks
    the rest out of the PPO loss.
    """
    from rl.scripted import SCRIPTED_POLICIES

    tables = state_tables(state)
    seat_keys = jax.random.split(key, MAX_AGENTS)

    def per_seat(seat, k):
        nodes, edges, globals_, src_mask, tgt_mask = seat_features(
            state, tables, seat)
        out_cur = net.sample_actions(
            k, params, nodes, edges, globals_, state.planets_alive,
            src_mask, tgt_mask)
        if opp_kind == "net":
            out_opp = net.sample_actions(
                k, opp_params, nodes, edges, globals_, state.planets_alive,
                src_mask, tgt_mask)
        else:  # scripted opponent
            g_tgt, g_frac, _ = SCRIPTED_POLICIES[opp_kind](
                state, tables, seat)
            out_opp = {
                "tgt": g_tgt, "frac": g_frac,
                "logp": jnp.float32(0.0), "value": jnp.float32(0.0),
                "entropy": jnp.float32(0.0),
            }
        is_learner = seat == learner_seat
        out = jax.tree.map(
            lambda c, o: jnp.where(is_learner, c, o), out_cur, out_opp)
        return out, src_mask

    outs, src_masks = jax.vmap(per_seat)(jnp.arange(MAX_AGENTS), seat_keys)
    return outs, src_masks


def rollout_chunk(key, params, state: GameState, pool: GameState,
                  n_steps: int, opp_params=None, learner_seats=None,
                  opp_kind: str = "mirror"):
    """Run `n_steps` of batched self-play. state/pool are batched
    GameStates (leading axes B / N_pool).

    opp_kind:
      "mirror" — all seats act under `params`; all active seats learn.
      "net"    — seats != learner_seats[b] act under `opp_params`;
                 only the learner seat is marked active for PPO.
      "greedy" — same, opponents are the scripted greedy bot.

    Returns (final_state, traj) where traj fields have leading (T, B).
    """
    B = state.step.shape[0]
    n_pool = pool.step.shape[0]

    def step_fn(carry, k):
        st = carry
        k_act, k_reset = jax.random.split(k)
        act_keys = jax.random.split(k_act, B)
        if opp_kind == "mirror":
            outs, src_masks = jax.vmap(policy_act, in_axes=(0, None, 0))(
                act_keys, params, st)
        else:
            outs, src_masks = jax.vmap(
                policy_act_vs_opp,
                in_axes=(0, None, None, 0, 0, None),
            )(act_keys, params, opp_params, learner_seats, st, opp_kind)

        pids, angles, ships = jax.vmap(actions_to_launch_tensors)(
            st, outs["tgt"], outs["frac"], src_masks)

        nxt = jax.vmap(jax_step)(st, pids, angles, ships)

        phi_now = jax.vmap(material_potential)(st)    # (B, A)
        phi_nxt = jax.vmap(material_potential)(nxt)   # (B, A)
        done = nxt.done                                # (B,)
        # Terminal reward straight from the engine (+1 winner / −1 else,
        # active seats only).
        terminal = jnp.where(done[:, None],
                             nxt.rewards.astype(jnp.float32), 0.0)
        shaping = SHAPING_COEF * (
            GAMMA * jnp.where(done[:, None], 0.0, phi_nxt) - phi_now)
        # At terminal, next potential is 0 (episode over).
        reward = shaping + terminal                    # (B, A)

        seat_ids = jnp.arange(MAX_AGENTS)
        active = seat_ids[None, :] < st.num_agents[:, None]  # (B, A)
        if opp_kind != "mirror":
            # Only the learner seat's experience is on-policy.
            active = active & (seat_ids[None, :] == learner_seats[:, None])

        # Auto-reset done envs from the pool.
        ridx = jax.random.randint(k_reset, (B,), 0, n_pool)
        fresh = jax.tree.map(lambda x: x[ridx], pool)
        nxt = jax.tree.map(
            lambda a, b: jnp.where(
                done.reshape((-1,) + (1,) * (a.ndim - 1)), b, a),
            nxt, fresh)

        out = {
            "state": st,
            "tgt": outs["tgt"], "frac": outs["frac"],
            "logp": outs["logp"], "value": outs["value"],
            "reward": reward, "done": done, "active": active,
        }
        return nxt, out

    keys = jax.random.split(key, n_steps)
    final_state, traj = jax.lax.scan(step_fn, state, keys)
    return final_state, traj


def bootstrap_values(params, state: GameState):
    """V(s) for all seats of the batched current state — bootstrap for GAE."""
    def one_env(st):
        tables = state_tables(st)

        def per_seat(seat):
            nodes, edges, globals_, src_mask, tgt_mask = seat_features(
                st, tables, seat)
            v, _, _ = net.forward(params, nodes, edges, globals_,
                                  st.planets_alive, tgt_mask)
            return v
        return jax.vmap(per_seat)(jnp.arange(MAX_AGENTS))
    return jax.vmap(one_env)(state)  # (B, A)
