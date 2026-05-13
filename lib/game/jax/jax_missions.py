"""JAX port of `lib/missions/snipe.py::propose_snipe_missions`.

Sub-phase 3 of the JAX sprint. Computes the (P_src × P_tgt) score
matrix vectorised over all source/target pairs.

For v7_0_drop_one, snipe is the primary mission builder; reinforce
is layered on top via `lib/missions/reinforce.py`. We port snipe
first, then reinforce, then settle_plan (the per-source greedy
chooser).

Status:
- ✅ Sub-phase 3a: `compute_snipe_score_matrix` base form.
- ✅ Sub-phase 3b: `aggressive=True` sizing variant.
- ⏳ Sub-phase 3c: leader-spoiler + neutral/comet bonus + comet lifetime
- ⏳ Sub-phase 3d: reinforce + recapture
- ⏳ Sub-phase 3e: settle_plan (per-source greedy with arrival ledger)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from lib.game.jax.jax_world_model import (
    JaxWorldModel, fleet_speed_batch, DEFAULT_HORIZON,
)


EPISODE_STEPS = 500

# Mirror lib/missions/snipe.py aggressive-sizing constants.
AGGRESSIVE_FRACTION = 0.7
AGGRESSIVE_RESERVE = 5
AGGRESSIVE_MIN_GARRISON = 12


def compute_snipe_score_matrix(
    state,                            # GameState
    world_model: JaxWorldModel,
    my_id: int,
    aggressive: bool = False,
):
    """Vectorised `propose_snipe_missions` over (src_planet, tgt_planet).

    Returns a dict of `(P_max, P_max)` arrays:
        - "score" float32 — score per (src, tgt). `-inf` for invalid.
        - "ships" int32 — ship count.
        - "eta"   int32 — ceil(distance / fleet_speed(ships)).
        - "valid" bool  — pair is a usable snipe candidate.

    Mirror of scalar `propose_snipe_missions` with:
      - `aggressive=False` (default): `base_ships = max(1, target_ships + 1)`.
      - `aggressive=True`: src-conditioned top-10 sizing — when source
        garrison > AGGRESSIVE_MIN_GARRISON, base_ships scales with
        AGGRESSIVE_FRACTION × src.ships, capped above by src.ships -
        AGGRESSIVE_RESERVE, capped below by target_min.
      - NEUTRAL_BONUS = COMET_BONUS = 1.0 (default identity multipliers)
      - LEADER_MULTIPLIER skipped (2P games only for now)
      - non-comet `time_to_hold = max(1, EPISODE_STEPS - step - eta)`

    Bonuses + comet lifetime handling land in sub-phase 3c.
    """
    P = state.planets_x.shape[0]

    # Source mask: alive + owned by my_id + has ships > 0.
    src_mask = (
        (state.planets_owner == my_id)
        & state.planets_alive
        & (state.planets_ships > 0)
    )
    # Target mask: alive + NOT owned by my_id.
    tgt_mask = state.planets_alive & (state.planets_owner != my_id)

    # Pairwise distance d[src, tgt].
    dx = state.planets_x[None, :] - state.planets_x[:, None]
    dy = state.planets_y[None, :] - state.planets_y[:, None]
    d = jnp.sqrt(dx * dx + dy * dy)  # (P, P)

    # Base ships.
    target_ships_row = state.planets_ships[None, :].astype(jnp.int32)  # (1, P)
    target_min_row = jnp.maximum(target_ships_row + 1, jnp.int32(1))   # (1, P)
    if aggressive:
        # int(src.ships * 0.7) and int(src.ships) - 5, per src.
        src_ships_col = state.planets_ships[:, None].astype(jnp.int32)  # (P, 1)
        fraction_size = jnp.maximum(
            jnp.int32(1),
            (src_ships_col.astype(jnp.float32) * jnp.float32(AGGRESSIVE_FRACTION)).astype(jnp.int32),
        )
        cap = jnp.maximum(jnp.int32(1), src_ships_col - jnp.int32(AGGRESSIVE_RESERVE))
        aggressive_size = jnp.maximum(
            target_min_row,
            jnp.minimum(fraction_size, cap),
        )                                                                # (P, P)
        eligible = src_ships_col > jnp.int32(AGGRESSIVE_MIN_GARRISON)     # (P, 1)
        base_ships = jnp.where(eligible, aggressive_size, target_min_row) # (P, P)
    else:
        base_ships = jnp.broadcast_to(target_min_row, (P, P))             # (P, P)

    # Fleet speed per (src, tgt). speed depends on ship count.
    speed_flat = fleet_speed_batch(base_ships.reshape(-1))
    speed = speed_flat.reshape(P, P)
    eta = jnp.ceil(d / jnp.maximum(speed, jnp.float32(1e-6))).astype(jnp.int32)

    # WorldModel filter: if target predicted ours with surplus at arrival,
    # the mission is redundant — drop.
    H = world_model.horizon
    safe_eta = jnp.clip(eta, jnp.int32(0), H)
    # Gather: for each target index t (column), look up owners_at[t, safe_eta[s, t]]
    tgt_idx_col = jnp.arange(P)[None, :]                            # (1, P)
    tgt_idx_grid = jnp.broadcast_to(tgt_idx_col, eta.shape)         # (P, P)
    pred_owner = world_model.owners_at[tgt_idx_grid, safe_eta]      # (P, P)
    pred_ships = world_model.ships_at[tgt_idx_grid, safe_eta]
    redundant = (pred_owner == jnp.int32(my_id)) & (pred_ships >= base_ships)

    # Value: production × time_to_hold.
    # Non-comet form (comet lifetime handling in sub-phase 3c).
    step_now = state.step
    time_to_hold = jnp.maximum(
        jnp.int32(1),
        jnp.int32(EPISODE_STEPS) - step_now - eta,
    )
    value = state.planets_prod[None, :].astype(jnp.float32) * time_to_hold.astype(jnp.float32)

    # Score = priority × value / (base_ships + d + 1).
    # priority = 1.0 (base form; bonuses in sub-phase 3c).
    denom = base_ships.astype(jnp.float32) + d + jnp.float32(1.0)
    score = value / denom

    # Final validity + masking.
    valid = src_mask[:, None] & tgt_mask[None, :] & ~redundant
    score = jnp.where(valid, score, jnp.float32(-jnp.inf))

    return {
        "score": score,         # (P, P) float32
        "ships": base_ships,    # (P, P) int32
        "eta": eta,             # (P, P) int32
        "valid": valid,         # (P, P) bool
    }


compute_snipe_score_matrix_jit = jax.jit(
    compute_snipe_score_matrix, static_argnames=("my_id", "aggressive")
)
