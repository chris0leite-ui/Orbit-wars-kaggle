"""JAX per-step physics for orbit_wars — sub-phase 1b (in progress).

This is the JIT'd hot path. Each phase is a pure function returning a
new GameState; `jax_step` chains them.

Status:
- ✅ production_tick — vectorised
- ✅ planet_path_compute — orbital rotation via initial_x/y + step
- ✅ comet_expire — masked alive update for expired comets
- ⏳ comet_path_advance — index increment + position lookup
- ⏳ fleet_launch_from_actions — gather + scatter
- ⏳ fleet_movement + sweep collision — F×P (the hot loop)
- ⏳ combat resolution — segment ops
- ⏳ apply planet movement — write new x/y
- ⏳ termination + rewards

Each phase is parity-tested separately against the scalar interpreter
in `tests/test_jax_phase_parity.py`. Full `jax_step` ships in
sub-phase 1c.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from lib.game.interpreter import (
    BOARD_SIZE, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS,
)
from lib.game.jax.jax_types import (
    GameState, MAX_PLANETS, MAX_FLEETS, NUM_COMET_SPAWNS,
    MAX_COMET_PATH_LEN, MAX_COMET_PATHS_PER_GROUP,
    MAX_LAUNCH_PER_AGENT, MAX_AGENTS,
)


def production_tick(state: GameState) -> GameState:
    """Add per-planet production to ships, but only for owned + alive
    planets. Mirror of `lib/game/interpreter.py:511-514`:

        for planet in obs0.planets:
            if planet[1] != -1:
                planet[5] += planet[6]

    Vectorised: O(P_max) elementwise.
    """
    owned = (state.planets_owner != -1) & state.planets_alive
    new_ships = state.planets_ships + jnp.where(
        owned, state.planets_prod, jnp.int32(0)
    )
    return state._replace(planets_ships=new_ships)


def planet_path_compute(state: GameState) -> GameState:
    """Compute new planet positions from initial orbital state + step.

    Mirror of `lib/game/interpreter.py:530-547`:
        for planet in obs0.planets:
            if planet[0] in comet_pid_set:
                continue
            initial_p = initial_by_id.get(planet[0])
            if initial_p:
                dx = initial_p[2] - CENTER
                dy = initial_p[3] - CENTER
                r = sqrt(dx*dx + dy*dy)
                if r + planet[4] < ROTATION_RADIUS_LIMIT:
                    current_angle = atan2(dy, dx) + angular_velocity * step
                    new_pos = (CENTER + r*cos(current_angle),
                               CENTER + r*sin(current_angle))
            planet_paths[pid] = (old_pos, new_pos, True)

    Vectorised over P_max; comets are skipped (their positions are
    updated by `comet_path_advance` instead).
    """
    dx = state.initial_x - CENTER
    dy = state.initial_y - CENTER
    r = jnp.sqrt(dx * dx + dy * dy)
    initial_angle = jnp.arctan2(dy, dx)
    step_f = state.step.astype(jnp.float32)
    current_angle = initial_angle + state.angular_velocity * step_f

    # Rotating condition: r + radius < ROTATION_RADIUS_LIMIT (and alive,
    # not a comet). For non-rotating planets we keep the OLD x/y
    # (matches scalar interpreter's `new_pos = old_pos` fallthrough).
    is_rotating = (
        (r + state.planets_radius < ROTATION_RADIUS_LIMIT)
        & state.planets_alive
        & ~state.is_comet
    )

    # Write to planets_new_*; planets_x/y stay at OLD positions until
    # apply_planet_movement (so fleet_movement's sweep-pair sees the
    # correct old→new planet segment, mirroring scalar interpreter).
    new_x = jnp.where(
        is_rotating,
        CENTER + r * jnp.cos(current_angle),
        state.planets_x,
    )
    new_y = jnp.where(
        is_rotating,
        CENTER + r * jnp.sin(current_angle),
        state.planets_y,
    )
    return state._replace(planets_new_x=new_x, planets_new_y=new_y)


def comet_expire(state: GameState) -> GameState:
    """Mark expired comet planets as not-alive.

    Mirror of `lib/game/interpreter.py:410-429` (the FIRST expiration
    pass, at top of each step before fleet launch):
        for group in obs0.comets:
            idx = group["path_index"]
            for i, pid in enumerate(group["planet_ids"]):
                if idx >= len(group["paths"][i]):
                    expired.append(pid)

    JAX version: each planet knows its source (spawn, path); look up the
    path length and current index, mark expired ones not-alive.

    Note: in the scalar code there are TWO expiration passes — at the
    top of the step (this one) and at the bottom (after fleet
    movement, in case a comet hits the boundary mid-step). Both use
    the same predicate (path_index >= len(path)). We model the top
    pass here; the bottom pass after fleet movement gets a second
    `comet_expire` call after the path_index has been advanced.
    """
    spawn_k = state.planet_comet_spawn
    path_j = state.planet_comet_path
    is_a_comet = spawn_k >= 0

    # Safe indexing: where not a comet, use index 0 (value unused).
    safe_spawn = jnp.where(is_a_comet, spawn_k, 0)
    safe_path = jnp.where(is_a_comet, path_j, 0)

    # Look up path length for each comet planet's (spawn, path).
    # comet_paths_len shape: (S, 4).
    my_path_len = state.comet_paths_len[safe_spawn, safe_path]
    # Current path index for the spawn. comet_path_index shape: (S,).
    my_path_idx = state.comet_path_index[safe_spawn]

    expired = is_a_comet & (my_path_idx >= my_path_len)
    new_alive = state.planets_alive & ~expired
    return state._replace(planets_alive=new_alive)


def comet_path_advance(state: GameState) -> GameState:
    """Increment comet path indices + look up new positions.

    Mirror of `lib/game/interpreter.py:549-566`:
        for group in obs0.comets:
            group["path_index"] += 1
            idx = group["path_index"]
            for i, pid in enumerate(group["planet_ids"]):
                ...
                if idx >= len(p_path):
                    expired.append(pid)
                    keep old_pos
                else:
                    new_pos = (p_path[idx][0], p_path[idx][1])

    JAX: per-planet gather from `comet_paths_xy[spawn, path, idx, :]`.
    Comets whose new index overruns their path length get marked
    expired (planets_alive set False, position unchanged).
    """
    # Per-spawn index increment (only for spawns that have FIRED, per
    # the `comet_spawned` flag). Newly-spawned comets have
    # `comet_path_index = -1`; first increment puts them at 0 (path[0]).
    spawned = state.comet_spawned
    new_path_index = jnp.where(
        spawned,
        state.comet_path_index + 1,
        state.comet_path_index,
    )

    # Per-planet: is it a (live) comet?
    spawn_k = state.planet_comet_spawn
    path_j = state.planet_comet_path
    is_a_comet = (spawn_k >= 0) & state.planets_alive

    # Safe gather indices.
    safe_spawn = jnp.where(is_a_comet, spawn_k, 0)
    safe_path = jnp.where(is_a_comet, path_j, 0)

    # New path index for this comet (post-increment).
    new_idx = new_path_index[safe_spawn]
    path_len = state.comet_paths_len[safe_spawn, safe_path]
    expired = is_a_comet & (new_idx >= path_len)

    # Look up new positions from comet_paths_xy. Use clipped index for
    # the gather (expired comets read garbage but get masked out).
    safe_idx = jnp.clip(new_idx, 0, MAX_COMET_PATH_LEN - 1)
    looked_up_x = state.comet_paths_xy[safe_spawn, safe_path, safe_idx, 0]
    looked_up_y = state.comet_paths_xy[safe_spawn, safe_path, safe_idx, 1]

    # Update positions: only move non-expired live comets. Write to
    # planets_new_* (deferred application via apply_planet_movement).
    # We seed from the EXISTING planets_new_* (which the scalar-equivalent
    # planet_path_compute writes for non-comets); this way the dict
    # represents "where everything will be after movement".
    movable = is_a_comet & ~expired
    new_x = jnp.where(movable, looked_up_x, state.planets_new_x)
    new_y = jnp.where(movable, looked_up_y, state.planets_new_y)
    new_alive = state.planets_alive & ~expired

    return state._replace(
        planets_new_x=new_x,
        planets_new_y=new_y,
        planets_alive=new_alive,
        comet_path_index=new_path_index,
    )


def fleet_launch(
    state: GameState,
    actions_pid: jnp.ndarray,    # (MAX_AGENTS, MAX_LAUNCH_PER_AGENT) int32
    actions_angle: jnp.ndarray,  # (MAX_AGENTS, MAX_LAUNCH_PER_AGENT) float32
    actions_ships: jnp.ndarray,  # (MAX_AGENTS, MAX_LAUNCH_PER_AGENT) int32
) -> GameState:
    """Process per-seat launch actions; allocate fleets in free slots.

    Mirror of `lib/game/interpreter.py:476-509` (process_moves).

    Action format is THREE parallel padded tensors (not packed) because
    pid + ships are int32 and angle is float32. Sentinel: `pid == -1`
    marks an unused action slot.

    For each potential launch (agent × k):
      1. Find planet slot whose `planets_id == pid`.
      2. Validate: planet alive, owner == agent_id, ships > 0,
         source ships >= requested ships.
      3. If valid: subtract ships from source planet, allocate next
         free fleet slot, write fleet fields.

    Slot allocation uses the cumsum-of-~alive trick: launches numbered
    1..N consume the 1st..Nth free fleet slots in order. JIT-friendly
    via fixed unrolled loop (MAX_AGENTS × MAX_LAUNCH_PER_AGENT = 80).
    """
    # Running state through the unrolled loop.
    planets_ships = state.planets_ships
    fleets_x = state.fleets_x
    fleets_y = state.fleets_y
    fleets_angle = state.fleets_angle
    fleets_owner = state.fleets_owner
    fleets_ships = state.fleets_ships
    fleets_from_planet = state.fleets_from_planet
    fleets_id = state.fleets_id
    fleets_alive = state.fleets_alive
    next_fleet_id = state.next_fleet_id
    launches_so_far = jnp.int32(0)

    for agent_id_py in range(MAX_AGENTS):
        for k in range(MAX_LAUNCH_PER_AGENT):
            pid_action = actions_pid[agent_id_py, k]
            angle_f = actions_angle[agent_id_py, k]
            ships_action = actions_ships[agent_id_py, k]

            # Find planet slot where planets_id == pid_action.
            match = (state.planets_id == pid_action) & state.planets_alive
            any_match = jnp.any(match)
            slot = jnp.argmax(match.astype(jnp.int32))

            is_valid = (
                (pid_action >= 0)
                & (ships_action > 0)
                & any_match
                & (state.planets_owner[slot] == agent_id_py)
                & (planets_ships[slot] >= ships_action)
            )

            # Allocate next free fleet slot from CURRENT fleets_alive
            # (which has been updated by previous iterations of this loop).
            is_free = ~fleets_alive
            cum_free = jnp.cumsum(is_free.astype(jnp.int32))
            target_count = launches_so_far + 1
            slot_mask = (cum_free == target_count) & is_free
            fleet_slot = jnp.argmax(slot_mask.astype(jnp.int32))

            do_launch = is_valid

            # Subtract ships from source planet.
            planets_ships = planets_ships.at[slot].set(
                jnp.where(
                    do_launch,
                    planets_ships[slot] - ships_action,
                    planets_ships[slot],
                )
            )

            # Compute fleet starting position from planet CURRENT pos +
            # r * (cos, sin) of angle, offset by (radius + 0.1).
            r_off = state.planets_radius[slot] + jnp.float32(0.1)
            sx = state.planets_x[slot] + jnp.cos(angle_f) * r_off
            sy = state.planets_y[slot] + jnp.sin(angle_f) * r_off

            fleets_x = fleets_x.at[fleet_slot].set(
                jnp.where(do_launch, sx, fleets_x[fleet_slot])
            )
            fleets_y = fleets_y.at[fleet_slot].set(
                jnp.where(do_launch, sy, fleets_y[fleet_slot])
            )
            fleets_angle = fleets_angle.at[fleet_slot].set(
                jnp.where(do_launch, angle_f, fleets_angle[fleet_slot])
            )
            fleets_owner = fleets_owner.at[fleet_slot].set(
                jnp.where(do_launch, jnp.int32(agent_id_py),
                          fleets_owner[fleet_slot])
            )
            fleets_ships = fleets_ships.at[fleet_slot].set(
                jnp.where(do_launch, ships_action, fleets_ships[fleet_slot])
            )
            fleets_from_planet = fleets_from_planet.at[fleet_slot].set(
                jnp.where(do_launch, pid_action, fleets_from_planet[fleet_slot])
            )
            fleets_id = fleets_id.at[fleet_slot].set(
                jnp.where(do_launch, next_fleet_id, fleets_id[fleet_slot])
            )
            fleets_alive = fleets_alive.at[fleet_slot].set(
                jnp.where(do_launch, True, fleets_alive[fleet_slot])
            )
            next_fleet_id = next_fleet_id + jnp.where(do_launch, 1, 0)
            launches_so_far = launches_so_far + jnp.where(do_launch, 1, 0)

    return state._replace(
        planets_ships=planets_ships,
        fleets_x=fleets_x, fleets_y=fleets_y, fleets_angle=fleets_angle,
        fleets_owner=fleets_owner, fleets_ships=fleets_ships,
        fleets_from_planet=fleets_from_planet, fleets_id=fleets_id,
        fleets_alive=fleets_alive, next_fleet_id=next_fleet_id,
    )


def apply_planet_movement(state: GameState) -> GameState:
    """Copy `planets_new_x/y` → `planets_x/y` for surviving planets.

    Mirror of `lib/game/interpreter.py:611-615`:
        for planet in obs0.planets:
            path = planet_paths.get(planet[0])
            if path is not None:
                planet[2], planet[3] = path[1]

    Vectorised: just write planets_new_* over planets_*. Dead planets
    keep their stale values but planets_alive=False masks them out
    everywhere else, so no harm.
    """
    return state._replace(
        planets_x=state.planets_new_x,
        planets_y=state.planets_new_y,
    )


def comet_spawn(state: GameState) -> GameState:
    """Instantiate 4 comet planets at the appropriate spawn boundary.

    Mirror of `lib/game/interpreter.py:431-474` — the lookup version,
    using the pre-computed `comet_paths_xy` + `comet_ships` from state
    (built in Python at game init by `scalar_to_jax`, which replays
    the scalar `generate_comet_paths` deterministically).

    Triggered when `state.step + 1 == comet_step[k]` for some valid k
    that hasn't fired yet. At most one spawn per step (the 5 spawn
    boundaries are distinct).

    Adds 4 new comet planets in the first 4 `~planets_alive` slots.
    Scalar interpreter starts them at `(-99, -99)`; `comet_path_advance`
    later that same step increments their path index from -1 to 0 and
    moves them to `paths[k, j, 0]`.
    """
    step_plus_1 = state.step + 1

    # Which spawn (if any) fires now? Per-spawn predicate:
    should_spawn_per_k = (
        (state.comet_step == step_plus_1)
        & state.comet_valid
        & ~state.comet_spawned
    )
    any_spawn = jnp.any(should_spawn_per_k)
    active_k = jnp.argmax(should_spawn_per_k.astype(jnp.int32))  # safe: 0 if no True
    safe_k = jnp.where(any_spawn, active_k, jnp.int32(0))

    # Find first 4 free planet slots via cumsum of ~alive mask.
    is_free = ~state.planets_alive
    cum_free = jnp.cumsum(is_free.astype(jnp.int32))

    def _first_idx_where(target: int):
        # First i where cum_free[i] == target and is_free[i].
        target_mask = (cum_free == target) & is_free
        return jnp.argmax(target_mask.astype(jnp.int32))

    slot0 = _first_idx_where(1)
    slot1 = _first_idx_where(2)
    slot2 = _first_idx_where(3)
    slot3 = _first_idx_where(4)
    slots = jnp.stack([slot0, slot1, slot2, slot3])  # (4,)

    new_ships = state.comet_ships[safe_k]

    # Per-planet update: at slot j, if any_spawn, overwrite with comet
    # template; else keep current value.
    planets_alive = state.planets_alive
    planets_x = state.planets_x
    planets_y = state.planets_y
    planets_new_x = state.planets_new_x
    planets_new_y = state.planets_new_y
    planets_id = state.planets_id
    planets_owner = state.planets_owner
    planets_ships = state.planets_ships
    planets_prod = state.planets_prod
    planets_radius = state.planets_radius
    is_comet_arr = state.is_comet
    initial_x = state.initial_x
    initial_y = state.initial_y
    planet_comet_spawn = state.planet_comet_spawn
    planet_comet_path = state.planet_comet_path

    # Synthetic pid for new comet planets: pack (spawn_k, path_j) into
    # a high range so it doesn't collide with the original 0..P_init-1
    # ids. Encoding: 100_000 + 10*k + j. This is internal-only — agents
    # never see these ids; they go through obs.planets[i][0] which we
    # rebuild in jax_to_scalar.
    base_comet_pid = 100_000 + 10 * safe_k

    # Constants matching scalar interpreter's comet template.
    COMET_X_PLACEHOLDER = jnp.float32(-99.0)
    COMET_Y_PLACEHOLDER = jnp.float32(-99.0)
    COMET_RADIUS_F = jnp.float32(1.0)         # COMET_RADIUS
    COMET_PRODUCTION_I = jnp.int32(1)         # COMET_PRODUCTION

    new_comet_planet_idx = state.comet_planet_idx

    for j in range(4):
        slot = slots[j]
        # mask_at_slot is True iff this slot will be written this step.
        planets_alive = planets_alive.at[slot].set(
            jnp.where(any_spawn, True, planets_alive[slot])
        )
        planets_x = planets_x.at[slot].set(
            jnp.where(any_spawn, COMET_X_PLACEHOLDER, planets_x[slot])
        )
        planets_y = planets_y.at[slot].set(
            jnp.where(any_spawn, COMET_Y_PLACEHOLDER, planets_y[slot])
        )
        # Seed planets_new_* to the placeholder too; comet_path_advance
        # will overwrite with the path[0] position later this step.
        planets_new_x = planets_new_x.at[slot].set(
            jnp.where(any_spawn, COMET_X_PLACEHOLDER, planets_new_x[slot])
        )
        planets_new_y = planets_new_y.at[slot].set(
            jnp.where(any_spawn, COMET_Y_PLACEHOLDER, planets_new_y[slot])
        )
        planets_id = planets_id.at[slot].set(
            jnp.where(any_spawn, base_comet_pid + j, planets_id[slot])
        )
        planets_owner = planets_owner.at[slot].set(
            jnp.where(any_spawn, jnp.int32(-1), planets_owner[slot])
        )
        planets_ships = planets_ships.at[slot].set(
            jnp.where(any_spawn, new_ships, planets_ships[slot])
        )
        planets_prod = planets_prod.at[slot].set(
            jnp.where(any_spawn, COMET_PRODUCTION_I, planets_prod[slot])
        )
        planets_radius = planets_radius.at[slot].set(
            jnp.where(any_spawn, COMET_RADIUS_F, planets_radius[slot])
        )
        is_comet_arr = is_comet_arr.at[slot].set(
            jnp.where(any_spawn, True, is_comet_arr[slot])
        )
        initial_x = initial_x.at[slot].set(
            jnp.where(any_spawn, COMET_X_PLACEHOLDER, initial_x[slot])
        )
        initial_y = initial_y.at[slot].set(
            jnp.where(any_spawn, COMET_Y_PLACEHOLDER, initial_y[slot])
        )
        planet_comet_spawn = planet_comet_spawn.at[slot].set(
            jnp.where(any_spawn, safe_k, planet_comet_spawn[slot])
        )
        planet_comet_path = planet_comet_path.at[slot].set(
            jnp.where(any_spawn, jnp.int32(j), planet_comet_path[slot])
        )
        new_comet_planet_idx = new_comet_planet_idx.at[active_k, j].set(
            jnp.where(any_spawn, slot, new_comet_planet_idx[active_k, j])
        )

    # Flip the spawned flag for active_k.
    new_comet_spawned = state.comet_spawned.at[active_k].set(
        jnp.where(any_spawn, True, state.comet_spawned[active_k])
    )

    return state._replace(
        planets_alive=planets_alive,
        planets_x=planets_x, planets_y=planets_y,
        planets_new_x=planets_new_x, planets_new_y=planets_new_y,
        planets_id=planets_id,
        planets_owner=planets_owner, planets_ships=planets_ships,
        planets_prod=planets_prod, planets_radius=planets_radius,
        is_comet=is_comet_arr,
        initial_x=initial_x, initial_y=initial_y,
        planet_comet_spawn=planet_comet_spawn,
        planet_comet_path=planet_comet_path,
        comet_spawned=new_comet_spawned,
        comet_planet_idx=new_comet_planet_idx,
    )


def swept_pair_hit_batch(
    fold: jnp.ndarray,  # (F, 2) fleet old positions
    fnew: jnp.ndarray,  # (F, 2) fleet new positions
    pold: jnp.ndarray,  # (P, 2) planet old positions
    pnew: jnp.ndarray,  # (P, 2) planet new positions
    pr: jnp.ndarray,    # (P,)   planet radii
) -> jnp.ndarray:
    """F×P broadcast swept-pair test — returns `(F, P)` bool of hits.

    Mirrors `lib/game/interpreter.py:swept_pair_hit` line-for-line but
    vectorised over both fleet and planet axes. This is the JAX
    building block for the sub-phase-1c fleet-movement-collision phase.

    Math (per-pair):
        d0 = fold - pold
        dv = (fnew - fold) - (pnew - pold)
        a  = |dv|^2
        b  = 2 (d0·dv)
        c  = |d0|^2 - r^2
        if a < 1e-12:        hit iff c <= 0
        else:
            disc = b^2 - 4ac
            disc < 0 →       hit = False
            else:
                t1, t2 = (-b ± √disc) / (2a)
                hit = t2 ≥ 0 AND t1 ≤ 1
    """
    # Broadcasting shapes:  fold[:, None, :] → (F, 1, 2);
    #                       pold[None, :, :] → (1, P, 2).
    d0 = fold[:, None, :] - pold[None, :, :]         # (F, P, 2)
    dv = (fnew[:, None, :] - fold[:, None, :]) - (
        pnew[None, :, :] - pold[None, :, :]
    )
    a = jnp.sum(dv * dv, axis=-1)                     # (F, P)
    b = 2.0 * jnp.sum(d0 * dv, axis=-1)               # (F, P)
    c = jnp.sum(d0 * d0, axis=-1) - (pr * pr)[None, :]

    a_small = a < 1e-12
    disc = b * b - 4.0 * a * c
    disc_ok = disc >= 0.0
    sq = jnp.sqrt(jnp.where(disc_ok, disc, 0.0))
    denom = jnp.where(a_small, 1.0, 2.0 * a)
    t1 = (-b - sq) / denom
    t2 = (-b + sq) / denom
    hit_full = disc_ok & (t2 >= 0.0) & (t1 <= 1.0)
    hit_degenerate = c <= 0.0
    return jnp.where(a_small, hit_degenerate, hit_full)


# Convenience: a partial `jax_step` covering ONLY the phases ported so
# far. Useful for parity-testing the implemented phases in isolation.
# The full jax_step (sub-phase 1c) will chain ALL phases.
def jax_step_partial(state: GameState) -> GameState:
    """Apply only the phases ported in sub-phase 1b. NOT a complete step.

    Order matches the scalar interpreter's phases that are present:
      1. comet_expire (pre-fleet-launch)
      2. production_tick (post-fleet-launch, but we don't have launch yet)
      3. planet_path_compute (uses current step)
    """
    state = comet_expire(state)
    state = production_tick(state)
    state = planet_path_compute(state)
    return state


# JIT'd versions for benchmarking.
production_tick_jit = jax.jit(production_tick)
planet_path_compute_jit = jax.jit(planet_path_compute)
comet_expire_jit = jax.jit(comet_expire)
comet_path_advance_jit = jax.jit(comet_path_advance)
comet_spawn_jit = jax.jit(comet_spawn)
apply_planet_movement_jit = jax.jit(apply_planet_movement)
fleet_launch_jit = jax.jit(fleet_launch)
swept_pair_hit_batch_jit = jax.jit(swept_pair_hit_batch)
jax_step_partial_jit = jax.jit(jax_step_partial)
