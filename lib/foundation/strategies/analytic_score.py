"""Phase A scoring kernel — strategy-agnostic atomic enumeration +
JAX-vmap'd batched candidate scoring.

Two pieces:

1. `enumerate_atomic_launches(state, my_id)` — strategy-agnostic
   action enumeration. For each owned planet × every other alive
   planet × each ship-fraction in `{0.5, 1.0}`, compute the orbit-
   aware aim angle via `lib.aim.aim_orbiting`. Returns ~200-600
   `ActionSpec` per typical mid-game state. No proposer dependency.

2. `score_candidates_vmap_value_prod(state, my_pids_c, my_angles_c,
   my_ships_c, my_id, K)` — batched scorer. For each candidate's
   action arrays (shape `(C, MAX_LAUNCH_PER_AGENT)`):
   - Apply our action at turn 0; opp plays no-op (empty action).
   - Score with `value_with_future_production`: `(my_ships +
     my_production × remaining_steps) - (opp_ships + opp_production
     × remaining_steps)`.
   - Returns shape `(C,)` float32.

The value head differs from `lib.game.jax.jax_score.value_delta_ships`
(used by v7_pv) — it adds `production × remaining_steps` to capture
the long-horizon production value that v7's K=10 rollout window
can't see. v7 sees only the next-10-turn ship delta; we score the
full game tail.

**Phase A deferred scope**: the PI's K-step Tier-1 mirror opp
rollout (plan §"Decisions locked", line 416) was traced and
measured at ~24 s cold compile + 70-200 ms/call warm at C=128 on
CPU JAX. With 16 beam calls/turn that's 1.1-3.2 s/turn — busts the
1000 ms Kaggle budget. Phase A therefore ships the simpler
"our-action-only + value head" kernel; Phase B re-enables the
mirror rollout once we can afford it (T4 GPU, smaller beam, or a
two-tier "shortlist + refine" pattern). The `K` argument is
retained for forward compatibility but currently unused.

This module's `score_candidates_vmap_value_prod` is the primitive
that the beam-search constructor calls once per beam level (vmap'd
over the level's candidate set). 2P-only assumption preserved for
symmetry with the Phase B opp-mirror upgrade.
"""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from lib.aim import aim_orbiting
from lib.foundation.actions import ActionSpec
from lib.game.jax.jax_interpreter import jax_step
from lib.game.jax.jax_types import (
    GameState,
    MAX_AGENTS,
    MAX_LAUNCH_PER_AGENT,
)


EPISODE_STEPS = 500


# ---------------------------------------------------------------------------
# Value head: ship-delta + future production
# ---------------------------------------------------------------------------


def value_with_future_production(state, my_id: int, episode_steps: int = EPISODE_STEPS):
    """`(my_ships + my_prod × remaining) - (opp_ships + opp_prod × remaining)`.

    JAX-pure; differentiable through state's continuous fields if
    needed downstream. `remaining = max(0, episode_steps - state.step)`.

    Improves on `value_delta_ships` by counting *future production* on
    held planets — v7's K=10 rollout sees the ship count at turn 10,
    but a planet captured on turn 10 is worth `production × 490` ships
    by game end, not just its turn-10 garrison.
    """
    my_id_jnp = jnp.int32(my_id)
    remaining = jnp.maximum(
        jnp.int32(0), jnp.int32(episode_steps) - state.step
    ).astype(jnp.float32)

    mine_p = (state.planets_owner == my_id_jnp) & state.planets_alive
    opp_p = (
        (state.planets_owner != my_id_jnp)
        & (state.planets_owner != jnp.int32(-1))
        & state.planets_alive
    )
    mine_f = (state.fleets_owner == my_id_jnp) & state.fleets_alive
    opp_f = (
        (state.fleets_owner != my_id_jnp)
        & (state.fleets_owner != jnp.int32(-1))
        & state.fleets_alive
    )

    ships_my = (
        jnp.sum(jnp.where(mine_p, state.planets_ships, jnp.int32(0)))
        + jnp.sum(jnp.where(mine_f, state.fleets_ships, jnp.int32(0)))
    ).astype(jnp.float32)
    ships_opp = (
        jnp.sum(jnp.where(opp_p, state.planets_ships, jnp.int32(0)))
        + jnp.sum(jnp.where(opp_f, state.fleets_ships, jnp.int32(0)))
    ).astype(jnp.float32)

    prod_my = jnp.sum(jnp.where(mine_p, state.planets_prod, jnp.int32(0))).astype(jnp.float32)
    prod_opp = jnp.sum(jnp.where(opp_p, state.planets_prod, jnp.int32(0))).astype(jnp.float32)

    return (ships_my + prod_my * remaining) - (ships_opp + prod_opp * remaining)


# ---------------------------------------------------------------------------
# Atomic-launch enumeration (strategy-agnostic)
# ---------------------------------------------------------------------------


def enumerate_atomic_launches(
    state: GameState,
    my_id: int,
    *,
    ship_fractions: tuple[float, ...] = (0.5, 1.0),
    max_eta: int = 80,
) -> list[ActionSpec]:
    """Strategy-agnostic action enumeration.

    For each (`src_planet`, `target_planet`) pair where `src.owner ==
    my_id` and `src.ships > 1`, plus each `fraction` in `ship_fractions`:
    compute the orbit-aware aim angle via `lib.aim.aim_orbiting`. Drop
    if no valid intercept exists or ETA exceeds `max_eta`.

    Returns a list of `ActionSpec` (de-duplicated by (src, target,
    fraction)). ~200-600 per typical mid-game state.

    Strategy-agnostic: no mission framework, no proposer ranking. The
    beam search picks among these.
    """
    out: list[ActionSpec] = []

    alive = np.asarray(state.planets_alive)
    ids = np.asarray(state.planets_id)
    owner = np.asarray(state.planets_owner)
    ships = np.asarray(state.planets_ships)
    x = np.asarray(state.planets_x)
    y = np.asarray(state.planets_y)
    radius = np.asarray(state.planets_radius)
    prod = np.asarray(state.planets_prod)
    omega = float(state.angular_velocity)

    P = len(alive)
    my_planets = [
        i for i in range(P)
        if bool(alive[i]) and int(owner[i]) == my_id and int(ships[i]) > 1
    ]
    all_targets = [
        i for i in range(P) if bool(alive[i]) and int(ids[i]) >= 0
    ]

    for src_i in my_planets:
        src_id = int(ids[src_i])
        src_pos = (float(x[src_i]), float(y[src_i]))
        src_radius = float(radius[src_i])
        src_ships = int(ships[src_i])

        for tgt_i in all_targets:
            if tgt_i == src_i:
                continue
            tgt_tuple = (
                int(ids[tgt_i]),
                int(owner[tgt_i]),
                float(x[tgt_i]),
                float(y[tgt_i]),
                float(radius[tgt_i]),
                int(ships[tgt_i]),
                int(prod[tgt_i]),
            )
            tgt_radius = float(radius[tgt_i])

            for fraction in ship_fractions:
                fleet_ships = max(1, int(src_ships * fraction))
                if fleet_ships > src_ships:
                    continue

                aim = aim_orbiting(
                    src_pos, src_radius, tgt_tuple, tgt_radius,
                    fleet_ships, omega,
                )
                if aim is None:
                    continue
                aim_angle, _arrival, eta = aim
                if eta is None or eta > max_eta:
                    continue

                out.append(ActionSpec(
                    from_planet_id=src_id,
                    dir_angle=float(aim_angle),
                    ships=fleet_ships,
                    launch_turn=0,
                    agent_id=my_id,
                ))

    return out


# ---------------------------------------------------------------------------
# Batched candidate scorer — vmap over (C,)
# ---------------------------------------------------------------------------


def score_candidates_vmap_value_prod(
    state: GameState,
    my_pids_c: jnp.ndarray,
    my_angles_c: jnp.ndarray,
    my_ships_c: jnp.ndarray,
    K: int,
    my_id: int,
    num_agents: int = 2,
    opp_aggressive: bool = True,
) -> jnp.ndarray:
    """Score C candidate action sets in one JIT'd vmap.

    Inputs:
      `state` — current `GameState`.
      `my_pids_c, my_angles_c, my_ships_c` — shape `(C, MAX_LAUNCH_
        PER_AGENT)`. Our action for each candidate; sentinel `-1`
        marks no-launch slots.
      `K` — RESERVED. Currently ignored; Phase B will use it for the
        mirror-rollout depth. See module docstring.
      `my_id` — our seat (0 or 1; 2P-only).
      `opp_aggressive` — RESERVED for Phase B; currently ignored
        (opp plays no-op at turn 0).

    Returns shape `(C,)` float32 scores from
    `value_with_future_production`.

    Performance (Phase A scope): one `jax_step` per candidate + one
    value-head eval per candidate. Cold compile ~18 s on CPU,
    warm ~30-50 ms per call at C=128.
    """
    if num_agents != 2:
        raise ValueError(
            f"score_candidates_vmap_value_prod is 2P-only "
            f"(got num_agents={num_agents}); 4P support follows the "
            f"Phase B opp-mirror generalisation."
        )

    def score_one(my_pids, my_angles, my_ships):
        pids_full = jnp.full(
            (MAX_AGENTS, MAX_LAUNCH_PER_AGENT), -1, dtype=jnp.int32,
        )
        ang_full = jnp.zeros(
            (MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=jnp.float32,
        )
        sh_full = jnp.zeros(
            (MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=jnp.int32,
        )
        pids_full = pids_full.at[my_id].set(my_pids)
        ang_full = ang_full.at[my_id].set(my_angles)
        sh_full = sh_full.at[my_id].set(my_ships)

        s = jax_step(state, pids_full, ang_full, sh_full)
        return value_with_future_production(s, my_id=my_id)

    return jax.vmap(score_one, in_axes=(0, 0, 0))(
        my_pids_c, my_angles_c, my_ships_c,
    )


score_candidates_vmap_value_prod_jit = jax.jit(
    score_candidates_vmap_value_prod,
    static_argnames=("K", "my_id", "num_agents", "opp_aggressive"),
)


def action_specs_to_candidate_arrays(
    candidates: list[list[ActionSpec]],
    *,
    max_launch: int = MAX_LAUNCH_PER_AGENT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack a list of C action sets into `(pids, angles, ships)` of
    shape `(C, max_launch)` each.

    Slots beyond a candidate's launch count are filled with sentinels.
    """
    C = len(candidates)
    pids = -np.ones((C, max_launch), dtype=np.int32)
    angles = np.zeros((C, max_launch), dtype=np.float32)
    ships = np.zeros((C, max_launch), dtype=np.int32)
    for c, specs in enumerate(candidates):
        for k, spec in enumerate(specs[:max_launch]):
            pids[c, k] = int(spec.from_planet_id)
            angles[c, k] = float(spec.dir_angle)
            ships[c, k] = int(spec.ships)
    return pids, angles, ships
