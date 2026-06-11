"""Tiny scripted JAX opponents for in-training progress evals.

greedy_policy: every turn, each owned planet with a decent garrison
sends 50% at the best-ROI target it can plausibly beat (nearest
neutral/enemy with projected defense < committed ships). Pure-JAX,
same (tgt, frac) action format as the learned policy.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from lib.game.jax.jax_types import GameState, MAX_PLANETS
from rl.features import FRACS, N_FRACS


def greedy_policy(state: GameState, tables, seat):
    """Returns (tgt (P,), frac (P,), src_mask (P,)) for one seat."""
    alive = state.planets_alive
    mine = (state.planets_owner == seat) & alive
    ships = state.planets_ships.astype(jnp.float32)

    aim = tables["aim"]
    eta = aim["eta"]
    ok = aim["valid"] & ~aim["sun_hit"] & ~jnp.eye(MAX_PLANETS, dtype=bool)

    # Send 50% of garrison; commit = what we send.
    commit = jnp.floor(0.5 * ships)  # (P,) per source
    # Defense proj: target ships + prod*eta for owned targets.
    owned_t = (state.planets_owner >= 0) & alive
    def_proj = (ships[None, :]
                + jnp.where(owned_t, state.planets_prod, 0)[None, :]
                .astype(jnp.float32) * eta)
    not_mine_t = alive & (state.planets_owner != seat)
    beatable = (def_proj < commit[:, None] * 0.9) & not_mine_t[None, :] & ok

    # Score: production per eta (ROI-ish); pick argmax per source.
    prod_t = jnp.maximum(state.planets_prod.astype(jnp.float32), 0.5)
    score = prod_t[None, :] / (eta + 5.0)
    score = jnp.where(beatable, score, -1e9)
    best = jnp.argmax(score, axis=1)                       # (P,)
    has_target = jnp.max(score, axis=1) > -1e8

    launch = mine & (ships >= 12.0) & has_target
    tgt = jnp.where(launch, best, MAX_PLANETS)             # hold otherwise
    frac = jnp.full((MAX_PLANETS,), 1, dtype=jnp.int32)    # 50%
    src_mask = mine & (state.planets_ships >= 1)
    return tgt, frac, src_mask


def idle_policy(state: GameState, tables, seat):
    """Never launches."""
    P = MAX_PLANETS
    tgt = jnp.full((P,), P, dtype=jnp.int32)
    frac = jnp.zeros((P,), dtype=jnp.int32)
    mine = (state.planets_owner == seat) & state.planets_alive
    return tgt, frac, mine & (state.planets_ships >= 1)
