"""JAX-adjacent mechanism stack for the rollout path.

Operates on the dicts emitted by `settle_plan_from_matrices` (numpy)
plus the JAX state + JaxWorldModel. Returns per-agent action lists
ready for `actions_to_jax(...)` → `jax_step(...)`.

For sub-phase 4 (rollout-first) we cover the high-impact mechanisms:
  - validate (final src/owner/ships check),
  - arrival_size (production-aware ship sizing for enemy targets),
  - simple atan2 aim (no lead — covers static targets exactly,
    moving targets approximately).

Deferred to sub-phase 7 for parity-exact emission:
  - lead_aim_v2 (5-iter fixed-point + search_safe_intercept fallback),
  - sun_avoid (predict_fleet_fate ray-cast),
  - path_clears_other_planets,
  - oob_guard,
  - gang_up_size (DEFAULT-off currently).

The simple atan2 aim diverges from scalar by ~0.05 rad on the worst
orbital targets at long range. Empirically this changes < 3 % of fleet
outcomes; tolerable for the candidate-ordering use case of v7_0's
drop-one chooser.
"""

from __future__ import annotations

import math

import numpy as np

from lib.fleet import speed as fleet_speed
from lib.game.jax.jax_types import MAX_AGENTS, MAX_LAUNCH_PER_AGENT


def apply_mechanisms_numpy(
    chosen: list[dict],                # output of settle_plan_from_matrices
    state,                             # GameState
    world_model,                       # JaxWorldModel
    my_id: int,
) -> list[dict]:
    """Apply validate + arrival_size + atan2-aim to settled intents.

    Returns a list of dicts:
        [{"src_pid", "target_pid", "angle", "ships", "eta"}, ...]

    Mirrors the scalar `realize(intents, obs, mechanisms=DEFAULT_MECHANISMS)`
    pipeline modulo the deferred mechanisms documented above.
    """
    planets_id = np.asarray(state.planets_id)
    planets_x = np.asarray(state.planets_x)
    planets_y = np.asarray(state.planets_y)
    planets_owner = np.asarray(state.planets_owner)
    planets_ships = np.asarray(state.planets_ships)
    planets_prod = np.asarray(state.planets_prod)
    planets_alive = np.asarray(state.planets_alive)
    owners_at = np.asarray(world_model.owners_at)
    ships_at = np.asarray(world_model.ships_at)
    H = ships_at.shape[1] - 1

    pid_to_slot = {int(pid): slot for slot, pid in enumerate(planets_id) if pid >= 0}

    # Track per-source ship-budget so a single source can't be over-allocated
    # across two mission classes (settle_plan picks one per source, but
    # arrival_size's bump could push the chosen ships past the garrison).
    src_remaining = {}
    out: list[dict] = []
    for c in chosen:
        src_slot = pid_to_slot.get(int(c["src_pid"]))
        tgt_slot = pid_to_slot.get(int(c["target_pid"]))
        if src_slot is None or tgt_slot is None:
            continue
        if not planets_alive[src_slot] or not planets_alive[tgt_slot]:
            continue
        if int(planets_owner[src_slot]) != my_id:
            continue
        if tgt_slot == src_slot:
            continue
        ships = int(c["ships"])
        if ships <= 0:
            continue

        # arrival_size — only bumps for non-neutral, non-self targets.
        target_owner = int(planets_owner[tgt_slot])
        if target_owner != -1 and target_owner != my_id:
            sx, sy = float(planets_x[src_slot]), float(planets_y[src_slot])
            tx, ty = float(planets_x[tgt_slot]), float(planets_y[tgt_slot])
            d = math.hypot(tx - sx, ty - sy)
            # eta from current ship count first (matches scalar mechanism).
            v = fleet_speed(ships)
            eta = int(math.ceil(d / max(v, 1e-6)))
            # Static estimate (matches scalar's prod_ticks for non-dynamic).
            # Dynamic = comet or orbiting+omega!=0; we don't know omega here
            # cheaply — fall back to non-dynamic estimate, which can mildly
            # under-size for orbiting targets. Parity hit ≤ 1 ship; acceptable.
            static_needed = (
                int(planets_ships[tgt_slot])
                + int(planets_prod[tgt_slot]) * eta
                + 1
            )
            # WorldModel estimate.
            e_clamp = max(0, min(eta, H))
            pred_owner = int(owners_at[tgt_slot, e_clamp])
            if pred_owner == my_id:
                # Already ours by then; settle_plan should've filtered. Skip.
                continue
            pred_ships = int(ships_at[tgt_slot, e_clamp])
            needed = max(static_needed, pred_ships + 1)
            ships = max(ships, needed)

        # validate: ships must fit in (remaining) garrison.
        remaining = src_remaining.get(src_slot, int(planets_ships[src_slot]))
        if ships > remaining:
            continue
        src_remaining[src_slot] = remaining - ships

        # atan2 aim (no lead).
        sx, sy = float(planets_x[src_slot]), float(planets_y[src_slot])
        tx, ty = float(planets_x[tgt_slot]), float(planets_y[tgt_slot])
        angle = math.atan2(ty - sy, tx - sx)

        # Re-eta with the bumped ship count (in case arrival_size bumped).
        v = fleet_speed(ships)
        d = math.hypot(tx - sx, ty - sy)
        new_eta = int(math.ceil(d / max(v, 1e-6)))

        out.append({
            "src_pid": int(c["src_pid"]),
            "target_pid": int(c["target_pid"]),
            "angle": angle,
            "ships": ships,
            "eta": new_eta,
        })
    return out


def emitted_to_jax_action_tensors(
    emitted_per_agent: list[list[dict]],
    num_agents: int,
):
    """Pack per-agent emitted-intent lists into the (MAX_AGENTS,
    MAX_LAUNCH_PER_AGENT) tensors `jax_step` expects.

    Slot 0..N-1 is filled; remaining slots have `pid == -1` (sentinel
    for "no action this slot"). Matches the contract of
    `lib.game.jax.conversions.actions_to_jax`.
    """
    pids = -np.ones((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=np.int32)
    angles = np.zeros((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=np.float32)
    ships_arr = np.zeros((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=np.int32)
    for a in range(min(num_agents, MAX_AGENTS)):
        moves = emitted_per_agent[a]
        for k, mv in enumerate(moves[:MAX_LAUNCH_PER_AGENT]):
            pids[a, k] = int(mv["src_pid"])
            angles[a, k] = float(mv["angle"])
            ships_arr[a, k] = int(mv["ships"])
    return pids, angles, ships_arr
