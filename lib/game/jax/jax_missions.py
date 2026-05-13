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
- ✅ Sub-phase 3c: comet-lifetime correction + neutral/comet/endgame
  bonus + leader-spoiler (4P, rank ≥ 2).
- ✅ Sub-phase 3d: `compute_reinforce_score_matrix` +
  `compute_recapture_score_matrix` (recapture-state tracking stays in
  Python; only the per-pair score math is JAX-vectorised).
- ✅ Sub-phase 3e (numpy form): `settle_plan_from_matrices` —
  per-source greedy with same-turn arrival ledger, operating on the
  stacked score/ships/eta/valid matrices from 3a-3d. JAX-vectorised
  scan form is deferred to sub-phase 5 (when score_candidate composes
  the full per-step decision and we know whether the numpy form fits
  in the per-game budget).
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

# Priority modifiers (mirror lib/missions/snipe.py).
NEUTRAL_BONUS = 1.0
COMET_BONUS = 1.0
ENDGAME_STEP = 470
ENDGAME_NEUTRAL_BONUS = 1.0
LEADER_MULTIPLIER = 1.5


def compute_snipe_score_matrix(
    state,                            # GameState
    world_model: JaxWorldModel,
    my_id: int,
    aggressive: bool = False,
    num_agents: int = 4,
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
    # For non-comet targets: max(1, EPISODE_STEPS - step - eta).
    # For comet targets: max(0, remaining_lifetime - eta).
    step_now = state.step
    non_comet_tth = jnp.maximum(
        jnp.int32(1),
        jnp.int32(EPISODE_STEPS) - step_now - eta,
    )
    # Per-planet comet remaining lifetime: paths_len[g, j] - path_index[g],
    # or 0 for non-comet planets.
    g = jnp.maximum(state.planet_comet_spawn, jnp.int32(0))
    j = jnp.maximum(state.planet_comet_path, jnp.int32(0))
    path_len = state.comet_paths_len[g, j]                # (P_max,)
    path_idx = state.comet_path_index[g]                  # (P_max,)
    rem_lifetime = jnp.where(
        state.is_comet,
        jnp.maximum(jnp.int32(0), path_len - path_idx),
        jnp.int32(0),
    )                                                      # (P_max,)
    comet_tth = jnp.maximum(jnp.int32(0), rem_lifetime[None, :] - eta)
    is_comet_tgt_row = state.is_comet[None, :]            # (1, P)
    time_to_hold = jnp.where(is_comet_tgt_row, comet_tth, non_comet_tth)
    value = state.planets_prod[None, :].astype(jnp.float32) * time_to_hold.astype(jnp.float32)

    # Priority modifiers (per scalar `propose_snipe_missions`).
    # Neutral/comet bonuses are identity (=1.0) by default; leader-spoiler
    # multiplier fires only when we are rank≥2 in a 3+P game.
    priority = jnp.ones((P, P), dtype=jnp.float32)
    # Neutral / comet bonus (currently identity but kept for tuning).
    is_neutral = (state.planets_owner == jnp.int32(-1))[None, :]
    neutral_mult = jnp.where(
        is_comet_tgt_row,
        jnp.float32(COMET_BONUS),
        jnp.float32(NEUTRAL_BONUS),
    )
    priority = jnp.where(is_neutral, priority * neutral_mult, priority)
    endgame_active = step_now >= jnp.int32(ENDGAME_STEP)
    priority = jnp.where(
        is_neutral & endgame_active,
        priority * jnp.float32(ENDGAME_NEUTRAL_BONUS),
        priority,
    )
    # Leader-spoiler: compute total ships per player (planets + fleets).
    NA = num_agents  # static
    planet_owner_clipped = jnp.maximum(state.planets_owner, jnp.int32(0))
    planet_contrib = jnp.where(
        state.planets_alive & (state.planets_owner >= 0),
        state.planets_ships.astype(jnp.float32),
        jnp.float32(0.0),
    )
    planet_totals = jnp.zeros(NA, dtype=jnp.float32).at[planet_owner_clipped].add(planet_contrib)
    fleet_owner_clipped = jnp.maximum(state.fleets_owner, jnp.int32(0))
    fleet_contrib = jnp.where(
        state.fleets_alive & (state.fleets_owner >= 0),
        state.fleets_ships.astype(jnp.float32),
        jnp.float32(0.0),
    )
    fleet_totals = jnp.zeros(NA, dtype=jnp.float32).at[fleet_owner_clipped].add(fleet_contrib)
    totals = planet_totals + fleet_totals
    leader = jnp.argmax(totals).astype(jnp.int32)
    my_total = totals[my_id]
    our_rank = jnp.sum(totals > my_total).astype(jnp.int32)
    num_active = jnp.sum(totals > 0)
    spoiler_on = (num_active >= 3) & (our_rank >= 2)
    is_leader_tgt = (state.planets_owner == leader)[None, :] & spoiler_on
    priority = jnp.where(is_leader_tgt, priority * jnp.float32(LEADER_MULTIPLIER), priority)

    # Score = priority × value / (base_ships + d + 1).
    denom = base_ships.astype(jnp.float32) + d + jnp.float32(1.0)
    score = priority * value / denom

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
    compute_snipe_score_matrix,
    static_argnames=("my_id", "aggressive", "num_agents"),
)


# ---------------------------------------------------------------------------
# Sub-phase 3d: reinforce score matrix
# ---------------------------------------------------------------------------


def compute_reinforce_score_matrix(
    state,                            # GameState
    world_model: JaxWorldModel,
    my_id: int,
):
    """Vectorised `propose_reinforce_missions` over (src, defended).

    For each of OUR planets D, find the first step `t_loss` in
    `[1, horizon]` at which `owners_at[D, t] != my_id`. For each (src,
    D) pair where src is also ours (src != D) and eta < t_loss, build:

        cost = max(1, ships_at[D, t_loss] + 1)
        value = D.production × max(1, EPISODE_STEPS - step - eta)
        score = value / (cost + d + 1)

    Returns dict of `(P_max, P_max)` arrays — same shape contract as
    `compute_snipe_score_matrix`. `valid=False` if D isn't threatened,
    src isn't ours, src==D, or we can't reach D before t_loss.
    """
    P = state.planets_x.shape[0]
    H = world_model.horizon

    # Threatened-target detection: first t ∈ [1, H] where owner != my_id.
    flip_mask = world_model.owners_at != jnp.int32(my_id)       # (P, H+1) bool
    # We only care about t >= 1. Zero out t=0 column.
    t_idx = jnp.arange(H + 1)[None, :]                          # (1, H+1)
    flip_mask = flip_mask & (t_idx >= 1)
    # argmax on bool returns first True index; if all False returns 0.
    t_loss_raw = jnp.argmax(flip_mask, axis=1).astype(jnp.int32)  # (P,)
    has_loss = jnp.any(flip_mask, axis=1)                        # (P,)
    # If no loss, set t_loss to H+1 (sentinel that always fails eta < t_loss).
    t_loss = jnp.where(has_loss, t_loss_raw, jnp.int32(H + 1))   # (P,)

    # Attacker strength at t_loss: post-flip ships on D.
    safe_t = jnp.clip(t_loss, jnp.int32(0), H)
    p_idx = jnp.arange(P)
    attacker_ships = world_model.ships_at[p_idx, safe_t]         # (P,)

    cost_col = jnp.maximum(jnp.int32(1), attacker_ships + jnp.int32(1))  # (P,)

    # Pairwise distance d[src, tgt].
    dx = state.planets_x[None, :] - state.planets_x[:, None]
    dy = state.planets_y[None, :] - state.planets_y[:, None]
    d = jnp.sqrt(dx * dx + dy * dy)                              # (P, P)

    # Cost per (src, tgt) is set by the TARGET column (defender).
    cost = jnp.broadcast_to(cost_col[None, :], (P, P))           # (P, P)
    speed_flat = fleet_speed_batch(cost.reshape(-1))
    speed = speed_flat.reshape(P, P)
    eta = jnp.ceil(d / jnp.maximum(speed, jnp.float32(1e-6))).astype(jnp.int32)

    # Value: defender's production × time_to_hold (we keep D alive for
    # the rest of the game — non-comet form is fine here, defenders
    # aren't comets).
    step_now = state.step
    time_to_hold = jnp.maximum(
        jnp.int32(1),
        jnp.int32(EPISODE_STEPS) - step_now - eta,
    )
    value = state.planets_prod[None, :].astype(jnp.float32) * time_to_hold.astype(jnp.float32)

    denom = cost.astype(jnp.float32) + d + jnp.float32(1.0)
    score = value / denom

    # Validity:
    src_mask = (state.planets_owner == my_id) & state.planets_alive
    tgt_mask = (state.planets_owner == my_id) & state.planets_alive & has_loss
    eye = jnp.eye(P, dtype=bool)
    can_arrive_in_time = eta < t_loss[None, :]                   # (P, P)
    valid = (
        src_mask[:, None] & tgt_mask[None, :] & ~eye & can_arrive_in_time
    )
    score = jnp.where(valid, score, jnp.float32(-jnp.inf))

    return {
        "score": score,
        "ships": cost,
        "eta": eta,
        "valid": valid,
        "t_loss": t_loss,                                        # (P,) per-target
    }


compute_reinforce_score_matrix_jit = jax.jit(
    compute_reinforce_score_matrix, static_argnames=("my_id",)
)


# ---------------------------------------------------------------------------
# Sub-phase 3d: recapture score matrix
# ---------------------------------------------------------------------------

RECAPTURE_WINDOW = 50
RECAPTURE_BONUS_PEAK = 1.5
RECENTLY_LOST_GARRISON_MAX = 50


def compute_recapture_score_matrix(
    state,                            # GameState
    world_model: JaxWorldModel,
    my_id: int,
    lost_at_step: jnp.ndarray,        # int32 (P_max,) — -1 if not lost
):
    """Vectorised `propose_recapture_missions` over (src, recently-lost).

    The recapture-state tracker (which planets we've recently lost +
    at which step) stays in Python — it's a per-turn ownership-diff
    O(P) dictionary update, no JAX win there. The caller passes
    `lost_at_step[i] = step we lost planet at slot i` (or `-1` if the
    planet isn't a recapture target).

    Targets are valid when:
      - lost_at_step >= 0
      - elapsed = step - lost_at_step ∈ [0, RECAPTURE_WINDOW]
      - target.owner != my_id (still lost)
      - target.ships ≤ RECENTLY_LOST_GARRISON_MAX (not fortified)
      - target alive

    Affordability: base_ships < src.ships (matches scalar).
    Redundancy: target predicted ours at arrival → drop.
    """
    P = state.planets_x.shape[0]

    src_mask = (
        (state.planets_owner == my_id)
        & state.planets_alive
        & (state.planets_ships > 0)
    )

    elapsed = state.step - lost_at_step  # (P,) int32, large if lost_at_step == -1
    recently_lost = (
        (lost_at_step >= jnp.int32(0))
        & (elapsed >= jnp.int32(0))
        & (elapsed <= jnp.int32(RECAPTURE_WINDOW))
    )
    not_fortified = state.planets_ships <= jnp.int32(RECENTLY_LOST_GARRISON_MAX)
    still_lost = state.planets_owner != my_id
    tgt_mask = (
        state.planets_alive & recently_lost & not_fortified & still_lost
    )

    # Pairwise distance.
    dx = state.planets_x[None, :] - state.planets_x[:, None]
    dy = state.planets_y[None, :] - state.planets_y[:, None]
    d = jnp.sqrt(dx * dx + dy * dy)

    target_ships_row = state.planets_ships[None, :].astype(jnp.int32)
    target_min_row = jnp.maximum(target_ships_row + 1, jnp.int32(1))
    base_ships = jnp.broadcast_to(target_min_row, (P, P))

    speed_flat = fleet_speed_batch(base_ships.reshape(-1))
    speed = speed_flat.reshape(P, P)
    eta = jnp.ceil(d / jnp.maximum(speed, jnp.float32(1e-6))).astype(jnp.int32)

    # Affordability: base_ships < src.ships (NOTE: scalar uses strict <).
    src_ships_col = state.planets_ships[:, None].astype(jnp.int32)
    affordable = base_ships < src_ships_col

    # Redundancy: target predicted ours at arrival.
    H = world_model.horizon
    safe_eta = jnp.clip(eta, jnp.int32(0), H)
    tgt_idx_col = jnp.arange(P)[None, :]
    tgt_idx_grid = jnp.broadcast_to(tgt_idx_col, eta.shape)
    pred_owner = world_model.owners_at[tgt_idx_grid, safe_eta]
    redundant = pred_owner == jnp.int32(my_id)

    # Urgency / bonus.
    elapsed_f = elapsed.astype(jnp.float32)
    urgency_per_target = jnp.maximum(
        jnp.float32(0.0),
        jnp.float32(1.0) - elapsed_f / jnp.float32(RECAPTURE_WINDOW),
    )                                                            # (P,)
    bonus_per_target = (
        jnp.float32(1.0) + (jnp.float32(RECAPTURE_BONUS_PEAK) - jnp.float32(1.0))
        * urgency_per_target
    )                                                            # (P,)
    bonus = bonus_per_target[None, :]                            # (1, P) → broadcast

    step_now = state.step
    time_to_hold = jnp.maximum(
        jnp.int32(1),
        jnp.int32(EPISODE_STEPS) - step_now - eta,
    )
    value = state.planets_prod[None, :].astype(jnp.float32) * time_to_hold.astype(jnp.float32)

    # Denominator follows snipe-aligned form (RECAPTURE_SCORE_DENOM_MATCHES_SNIPE=1).
    denom = base_ships.astype(jnp.float32) + d + jnp.float32(1.0)
    score = bonus * value / denom

    valid = (
        src_mask[:, None] & tgt_mask[None, :] & affordable & ~redundant
    )
    score = jnp.where(valid, score, jnp.float32(-jnp.inf))

    return {
        "score": score,
        "ships": base_ships,
        "eta": eta,
        "valid": valid,
    }


compute_recapture_score_matrix_jit = jax.jit(
    compute_recapture_score_matrix, static_argnames=("my_id",)
)


# ---------------------------------------------------------------------------
# Sub-phase 3e: settle_plan (numpy form, operating on JAX matrices)
# ---------------------------------------------------------------------------


import numpy as _np


def settle_plan_from_matrices(
    class_outputs: list,                # list of dicts with score/ships/eta/valid
    class_names: list,                  # parallel list of class name strings
    planets_id,                         # int32 (P_max,) JAX or numpy
    world_owners_at,                    # (P_max, H+1) JAX or numpy
    world_ships_at,                     # (P_max, H+1) JAX or numpy
    my_id: int,
):
    """Per-source greedy with same-turn arrival ledger.

    Pure numpy mirror of `lib.planner.settle_plan` operating on the
    JAX score/ships/eta/valid matrices. Returns a list of dicts:
        [{"src_pid", "target_pid", "mission_class", "ships", "eta", "score"}, ...]

    Algorithm (verbatim port of scalar):
    1. Build candidate list per source from each class's valid cells.
    2. Sort each source's candidates by score descending.
    3. Order sources by their top candidate's score (descending).
    4. Per source in order, walk candidates; accept first whose target
       passes the ledger check (cumulative prior arrivals by step ≤ eta
       < pred_enemy at eta + 1). Update ledger on accept.
    """
    planets_id = _np.asarray(planets_id)
    ships_at = _np.asarray(world_ships_at)
    P = planets_id.shape[0]
    H = ships_at.shape[1] - 1

    # Flatten all classes into a (src_slot -> list of candidate tuples).
    # Each tuple: (score, target_slot, ships, eta, class_name).
    by_src: dict[int, list[tuple]] = {}
    for cls_name, out in zip(class_names, class_outputs):
        score = _np.asarray(out["score"])
        ships_m = _np.asarray(out["ships"])
        eta_m = _np.asarray(out["eta"])
        valid_m = _np.asarray(out["valid"])
        valid_idx = _np.argwhere(valid_m)
        for s, t in valid_idx:
            tup = (
                float(score[s, t]),
                int(t),
                int(ships_m[s, t]),
                int(eta_m[s, t]),
                cls_name,
            )
            by_src.setdefault(int(s), []).append(tup)

    # Sort each source's candidates by score descending.
    for s in by_src:
        by_src[s].sort(key=lambda x: -x[0])

    # Order sources by top-candidate score desc.
    src_order = sorted(by_src.keys(), key=lambda s: -by_src[s][0][0])

    # Ledger: per-target list of (eta, ships).
    pending: dict[int, list[tuple[int, int]]] = {}
    chosen = []
    for src_slot in src_order:
        for (sc, t_slot, ships_v, eta_v, cls_name) in by_src[src_slot]:
            # Sum of prior arrivals at this target with eta_prior ≤ eta_v.
            already = 0
            for (e_prior, s_prior) in pending.get(t_slot, []):
                if e_prior <= eta_v:
                    already += s_prior
            # pred_enemy at the chosen eta. Clip to horizon.
            e_clamp = min(max(eta_v, 0), H)
            pred_enemy = float(ships_at[t_slot, e_clamp])
            if already >= pred_enemy + 1.0:
                continue
            chosen.append({
                "src_pid": int(planets_id[src_slot]),
                "target_pid": int(planets_id[t_slot]),
                "mission_class": cls_name,
                "ships": int(ships_v),
                "eta": int(eta_v),
                "score": float(sc),
            })
            pending.setdefault(t_slot, []).append((eta_v, ships_v))
            break

    return chosen
