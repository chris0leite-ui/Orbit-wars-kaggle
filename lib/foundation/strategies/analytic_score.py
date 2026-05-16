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
from lib.game.jax.jax_interpreter import jax_step, jax_step_no_launch
from lib.game.jax.jax_types import (
    GameState,
    MAX_AGENTS,
    MAX_LAUNCH_PER_AGENT,
)
from lib.intent import World
from lib.world_model import WorldModel


EPISODE_STEPS = 500

# Geometric production-value discount (Piece B.2 of the v8_scavenge port).
# Matches `lib/scoring.py:PV_GAMMA` convention. The closed-form horizon
# factor `gamma_h = (1 - PV_GAMMA**remaining) / (1 - PV_GAMMA)` weights
# early-turn production much more than late-turn production, which adds
# (i) own-planet retention signal (`prod_my` drops when a planet flips
# during the K-step rollout) and (ii) early-capture compounding signal
# (the maruichi01 pattern: grabbing high-prod planets at turn 5 is
# worth materially more than turn 50).
PV_GAMMA = 0.99


# ---------------------------------------------------------------------------
# Value head: ship-balance + γ-discounted production (favor)
# ---------------------------------------------------------------------------


def value_with_future_production(state, my_id: int, episode_steps: int = EPISODE_STEPS):
    """γ-discounted favor: `(my_ships + my_prod × γ_h) - (opp_ships + opp_prod × γ_h)`.

    `γ_h = sum_{t=0..remaining-1} PV_GAMMA**t = (1 - PV_GAMMA**remaining)
    / (1 - PV_GAMMA)`. Mathematically equivalent to v8_scavenge's
    `F1 + F2 × pv_horizon(γ=0.99)` factored across both seats.

    Two signals emerge naturally when paired with a K-step rollout:
    - Defensive: when one of my planets flips at turn t ≤ K, `prod_my`
      drops by that planet's production for the remainder of the
      rollout horizon, and the leaf value drops by
      `prod[tgt] × γ_h × (1 - t/K)`.
    - Compounding: production accrued near `state.step` is worth much
      more than production accrued near `EPISODE_STEPS`. A 3-prod
      capture at step 5 vs step 50 sees a meaningful γ-discount gap,
      where the previous linear-`remaining` head saw only ~10%.

    JAX-pure; JIT-compatible. Name preserved for caller compatibility.
    """
    my_id_jnp = jnp.int32(my_id)
    remaining = jnp.maximum(
        jnp.int32(0), jnp.int32(episode_steps) - state.step
    ).astype(jnp.float32)
    gamma = jnp.float32(PV_GAMMA)
    # Closed-form geometric sum; at γ=0.99 and remaining=500, gamma_h≈
    # (1 − 0.99**500)/0.01 ≈ 99.3, vs the prior linear value of 500.
    gamma_h = (jnp.float32(1.0) - jnp.power(gamma, remaining)) / (jnp.float32(1.0) - gamma)

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

    return (ships_my + prod_my * gamma_h) - (ships_opp + prod_opp * gamma_h)


# ---------------------------------------------------------------------------
# Atomic-launch enumeration (strategy-agnostic)
# ---------------------------------------------------------------------------


def enumerate_atomic_launches(
    state: GameState,
    my_id: int,
    *,
    ship_fractions: tuple[float, ...] = (0.5, 1.0),
    max_eta: int = 80,
    nearest_k_per_source: int = 8,
) -> list[ActionSpec]:
    """Strategy-agnostic action enumeration.

    For each (`src_planet`, `target_planet`) pair where `src.owner ==
    my_id` and `src.ships > 1`, plus each `fraction` in `ship_fractions`:
    compute the orbit-aware aim angle via `lib.aim.aim_orbiting`. Drop
    if no valid intercept exists or ETA exceeds `max_eta`.

    `nearest_k_per_source` caps targets per source to the `K` closest
    by Euclidean distance. Matches v8_scavenge's `NUM_TARGETS_PER_SOURCE
    =8` — keeps the beam budget bounded by reducing total atoms from
    `|sources|×|alive|×|fractions|` to `|sources|×K×|fractions|`. The
    nearest planets are also the ones a fleet can plausibly reach with
    a usable ETA, so this trims mostly atoms that the `max_eta` filter
    would have dropped anyway.

    Returns a list of `ActionSpec` (de-duplicated by (src, target,
    fraction)). ~80-200 per typical mid-game state with the cap.

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

        # Cap targets per source to the K closest by Euclidean distance.
        candidates = [t for t in all_targets if t != src_i]
        candidates.sort(
            key=lambda t: (x[t] - x[src_i]) ** 2 + (y[t] - y[src_i]) ** 2
        )
        candidates = candidates[:nearest_k_per_source]

        for tgt_i in candidates:
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
# Defensive-reinforce enumeration (Commit 1 of v8_scavenge port)
# ---------------------------------------------------------------------------

# Sized to match v8_scavenge's `MAX_HORIZON=30` minus a small margin; any
# threat further out than this gets handled on a later turn (the chooser
# revisits every step). Tighter than 30 to avoid generating reinforce
# atoms for far-future threats that are noise more than signal.
_K_THREAT_HORIZON = 25

# A 1-ship "reinforce" can't survive combat resolution; matches v8_scavenge.
_MIN_REINFORCE_SHIPS = 2


def enumerate_defensive_reinforce(
    state: GameState,
    my_id: int,
    world_model: Optional[WorldModel] = None,
    raw_obs: Optional[object] = None,
    *,
    max_eta: int = 80,
) -> list[ActionSpec]:
    """Generate `ActionSpec` atoms aimed at my own threatened planets.

    Ports v8_scavenge's M2 fix (commit 82b5526): the standard offensive
    enumerator never proposes "send a fleet to defend my own planet,"
    so the beam can't choose defence even though the action space
    allows it. Without this, opp's continuous fleet pressure flips
    captured planets back and eliminates us mid-game — observed as the
    0/32 smoke loss vs `nearest`.

    Threat is detected via `WorldModel.incoming_enemy_eta`. For each
    own planet with at least one inbound enemy fleet arriving within
    `_K_THREAT_HORIZON` turns, we (a) sum the incoming enemy ships
    that land at or before that ETA, (b) predict our garrison at the
    same ETA (`current + prod × eta`), (c) compute the shortfall, and
    (d) for every OTHER owned planet with a usable garrison, propose a
    reinforce fleet sized to close the gap. The fleet must be able to
    arrive at-or-before the threat lands (`reinforce_eta ≤ enemy_eta`).

    `world_model` is preferred (caller-precomputed). When `None`, we
    fall back to building one from `raw_obs` — slightly slower but
    keeps the entry point usable from any agent that passes obs.

    Returns the new atoms; the caller concatenates them with the
    offensive output of `enumerate_atomic_launches`.
    """
    if world_model is None:
        if raw_obs is None:
            return []
        world_model = WorldModel.from_world(World.from_obs(raw_obs))

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
        if bool(alive[i]) and int(owner[i]) == my_id
    ]
    if len(my_planets) < 2:
        # Need at least one source and one target; if we own ≤1 planet
        # nothing else can reinforce it.
        return []

    out: list[ActionSpec] = []

    for tgt_i in my_planets:
        tgt_id = int(ids[tgt_i])
        enemy_eta = world_model.incoming_enemy_eta(tgt_id, my_id)
        if enemy_eta is None or enemy_eta > _K_THREAT_HORIZON:
            continue

        arrivals = world_model.ledger.get(tgt_id) or []
        enemy_ships = sum(
            int(a_ships) for (a_eta, a_owner, a_ships) in arrivals
            if a_owner != my_id and a_eta <= enemy_eta + 1 and a_ships > 0
        )
        if enemy_ships <= 0:
            continue

        my_garrison_at_eta = int(ships[tgt_i]) + int(prod[tgt_i]) * int(enemy_eta)
        shortfall = enemy_ships - my_garrison_at_eta + 1
        if shortfall <= 0:
            continue

        tgt_tuple = (
            tgt_id,
            int(owner[tgt_i]),
            float(x[tgt_i]),
            float(y[tgt_i]),
            float(radius[tgt_i]),
            int(ships[tgt_i]),
            int(prod[tgt_i]),
        )
        tgt_radius = float(radius[tgt_i])

        for src_i in my_planets:
            if src_i == tgt_i:
                continue
            src_ships = int(ships[src_i])
            if src_ships <= _MIN_REINFORCE_SHIPS:
                continue

            fleet_ships = min(int(math.ceil(shortfall)), src_ships - 1)
            if fleet_ships < _MIN_REINFORCE_SHIPS:
                continue

            src_pos = (float(x[src_i]), float(y[src_i]))
            src_radius = float(radius[src_i])
            aim = aim_orbiting(
                src_pos, src_radius, tgt_tuple, tgt_radius,
                fleet_ships, omega,
            )
            if aim is None:
                continue
            aim_angle, _arrival, eta = aim
            if eta is None or eta > max_eta or eta > enemy_eta:
                # Must arrive before the threat lands; otherwise the
                # reinforce can't help.
                continue

            out.append(ActionSpec(
                from_planet_id=int(ids[src_i]),
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
      `K` — rollout horizon in turns. Turn 0 applies my candidate
        action; turns 1..K-1 are strict-idle (both seats no-op). This
        lets the value head observe enemy fleet arrivals at ETAs up
        to K-1, which is required for defensive reinforce candidates
        to win argmax over offensive captures (Piece B of the
        v8_scavenge port).
      `my_id` — our seat (0 or 1; 2P-only).
      `opp_aggressive` — RESERVED. Strict-idle in the scan body; the
        v8_scavenge wallclock budget tolerates this and a later
        iteration can swap in a Tier-1 mirror policy.

    Returns shape `(C,)` float32 scores from
    `value_with_future_production` evaluated at the K-step leaf.
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

        # Turn 0: my candidate action via full jax_step.
        s = jax_step(state, pids_full, ang_full, sh_full)
        # Turns 1..K-1: strict-idle, so threats inbound at ETA ≤ K-1
        # land and any planet flips become visible to the leaf head.
        # `jax_step_no_launch` omits `fleet_launch` — the dominant
        # phase (55% of per-step cost) — since no seat launches on
        # idle steps. Cuts the K-scan body roughly in half.
        def idle_body(s_in, _):
            return jax_step_no_launch(s_in), None
        s, _ = jax.lax.scan(idle_body, s, None, length=K - 1)
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
