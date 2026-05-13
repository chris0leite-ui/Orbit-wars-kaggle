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
jax_step_partial_jit = jax.jit(jax_step_partial)
