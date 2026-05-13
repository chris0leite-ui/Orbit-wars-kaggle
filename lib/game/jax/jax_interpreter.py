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
    return state._replace(planets_x=new_x, planets_y=new_y)


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
    # Per-spawn index increment (only for spawns that have been
    # spawned, i.e. comet_path_index >= 0).
    spawned = state.comet_path_index >= 0
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

    # Update positions: only move non-expired live comets.
    movable = is_a_comet & ~expired
    new_x = jnp.where(movable, looked_up_x, state.planets_x)
    new_y = jnp.where(movable, looked_up_y, state.planets_y)
    new_alive = state.planets_alive & ~expired

    return state._replace(
        planets_x=new_x,
        planets_y=new_y,
        planets_alive=new_alive,
        comet_path_index=new_path_index,
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
swept_pair_hit_batch_jit = jax.jit(swept_pair_hit_batch)
jax_step_partial_jit = jax.jit(jax_step_partial)
