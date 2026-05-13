"""JAX port of `lib/world_model.py` — arrival ledger + per-planet timeline.

Sub-phase 2 of the JAX sprint. Builds on the engine port (sub-phase 1).

Status:
- ✅ Sub-phase 2a: `fleet_target_planet_batch` (vectorised raycast)
- ⏳ Sub-phase 2b: `build_arrival_ledger_jax` (per-planet arrival lists)
- ⏳ Sub-phase 2c: `simulate_planet_timeline_jax` (step-by-step ownership/garrison)
- ⏳ Sub-phase 2d: `JaxWorldModel` Pytree wrapping the timelines

The agent's mission framework consumes the WorldModel via `owner_at`,
`ships_at`, and `incoming_enemy_eta` queries; sub-phase 3 (JAX missions)
will read from `JaxWorldModel` directly.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


# Mirror lib/world_model.DEFAULT_HORIZON.
DEFAULT_HORIZON = 250

# Mirror lib/fleet.speed (the SAME formula used by orbit_wars.interpreter):
#   speed = 1 + (MAX_SPEED - 1) * (log(ships) / log(1000)) ^ 1.5, capped.
_LOG_1000 = math.log(1000.0)


def fleet_speed_batch(
    ships: jnp.ndarray,  # (F,) int32 or float32
    max_speed: float = 6.0,
) -> jnp.ndarray:
    """Vectorised fleet-speed formula. Returns float32 per fleet."""
    ships_f = ships.astype(jnp.float32)
    # Guard log(0) for dead-slot sentinels.
    safe = jnp.maximum(ships_f, jnp.float32(1.0))
    spd = jnp.float32(1.0) + (jnp.float32(max_speed) - 1.0) * (
        jnp.log(safe) / jnp.float32(_LOG_1000)
    ) ** 1.5
    return jnp.minimum(spd, jnp.float32(max_speed))


def fleet_target_planet_batch(
    fleets_x: jnp.ndarray,       # (F,) float32 — fleet positions
    fleets_y: jnp.ndarray,
    fleets_angle: jnp.ndarray,   # (F,) float32
    fleets_ships: jnp.ndarray,   # (F,) int32
    fleets_alive: jnp.ndarray,   # (F,) bool
    planets_x: jnp.ndarray,      # (P,) float32 — planet centers
    planets_y: jnp.ndarray,
    planets_radius: jnp.ndarray, # (P,) float32
    planets_alive: jnp.ndarray,  # (P,) bool
    max_horizon: int = DEFAULT_HORIZON,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-fleet first-planet-hit raycast. Returns `(target_idx[F],
    eta[F])` — slot index of the first planet each fleet's trajectory
    crosses (or -1 if none within horizon), and integer ETA in turns.

    Mirror of `lib/world_model.fleet_target_planet`. Vectorised F×P;
    a single broadcasted call replaces the scalar nested-loop.

    Math per (f, p):
        dir_x = cos(angle_f), dir_y = sin(angle_f)
        dx = px - fx,         dy = py - fy
        proj = dx*dir_x + dy*dir_y   # signed forward distance
        if proj < 0: no hit (planet behind fleet)
        perp_sq = dx² + dy² - proj²  # squared perpendicular distance
        r_sq = radius²
        if perp_sq >= r_sq: no hit
        hit_d = max(0, proj - sqrt(r_sq - perp_sq))
        turns = ceil(hit_d / speed)

    For each fleet, picks the planet with minimum `turns`.
    """
    F = fleets_x.shape[0]
    P = planets_x.shape[0]

    dir_x = jnp.cos(fleets_angle)
    dir_y = jnp.sin(fleets_angle)
    spd = fleet_speed_batch(fleets_ships)

    # Broadcast to (F, P) grid.
    dx = planets_x[None, :] - fleets_x[:, None]
    dy = planets_y[None, :] - fleets_y[:, None]
    proj = dx * dir_x[:, None] + dy * dir_y[:, None]              # (F, P)
    perp_sq = dx * dx + dy * dy - proj * proj
    r_sq = (planets_radius * planets_radius)[None, :]

    # Hit conditions
    proj_ok = proj >= jnp.float32(0.0)
    perp_ok = perp_sq < r_sq

    inside_sqrt = jnp.maximum(r_sq - perp_sq, jnp.float32(0.0))
    hit_d = jnp.maximum(proj - jnp.sqrt(inside_sqrt), jnp.float32(0.0))
    safe_spd = jnp.maximum(spd[:, None], jnp.float32(1e-9))
    turns = hit_d / safe_spd                                       # (F, P) float

    valid = (
        proj_ok & perp_ok
        & fleets_alive[:, None] & planets_alive[None, :]
        & (turns <= jnp.float32(max_horizon))
    )

    # For each fleet, find planet with min turns among valid candidates.
    inf = jnp.float32(1e18)
    masked_turns = jnp.where(valid, turns, inf)
    target_idx = jnp.argmin(masked_turns, axis=1)                  # (F,) int
    any_hit = jnp.any(valid, axis=1)
    target_out = jnp.where(any_hit, target_idx, jnp.int32(-1))
    # ETA = ceil(min_turns). Use -1 sentinel for no-hit.
    min_turns = jnp.min(masked_turns, axis=1)
    eta_out = jnp.where(
        any_hit,
        jnp.ceil(min_turns).astype(jnp.int32),
        jnp.int32(-1),
    )
    return target_out, eta_out


fleet_target_planet_batch_jit = jax.jit(
    fleet_target_planet_batch, static_argnames=("max_horizon",)
)
fleet_speed_batch_jit = jax.jit(fleet_speed_batch)


# ---------------------------------------------------------------------------
# Sub-phase 2b: build_arrival_grid
# ---------------------------------------------------------------------------


def build_arrival_grid(
    state,                               # GameState (sub-phase 1 type)
    max_horizon: int = DEFAULT_HORIZON,
    num_agents: int = 4,
) -> jnp.ndarray:
    """Per-(planet, step, owner) total incoming ships from in-flight fleets.

    Returns int32 grid shape `(MAX_PLANETS, max_horizon+1, num_agents)`.
    `grid[p, t, a]` = sum of ships owned by player `a` arriving at planet
    slot `p` at step `t` (1-indexed; t=0 unused).

    Equivalent to scalar `lib.world_model.build_arrival_ledger` but
    pre-aggregated by owner (which is what `resolve_arrivals` does as
    its first step in `simulate_planet_timeline`).

    Implementation: per-fleet raycast → 3D scatter-add into the grid.
    Dead / unowned / out-of-horizon / no-hit fleets are masked out.
    """
    from lib.game.jax.jax_types import MAX_PLANETS

    target_idx, eta = fleet_target_planet_batch(
        state.fleets_x, state.fleets_y, state.fleets_angle,
        state.fleets_ships, state.fleets_alive,
        state.planets_x, state.planets_y, state.planets_radius,
        state.planets_alive, max_horizon=max_horizon,
    )
    # Scalar buckets ceil(eta) to max(1, ceil(eta)) before storing.
    bucketed = jnp.maximum(eta, jnp.int32(1))

    valid = (
        (target_idx >= 0)
        & state.fleets_alive
        & (state.fleets_ships > 0)
        & (state.fleets_owner >= 0)
        & (state.fleets_owner < jnp.int32(num_agents))
        & (bucketed <= jnp.int32(max_horizon))
    )

    safe_p = jnp.where(valid, target_idx, jnp.int32(0))
    safe_t = jnp.where(valid, bucketed, jnp.int32(0))
    safe_a = jnp.where(valid, state.fleets_owner, jnp.int32(0))
    safe_ships = jnp.where(valid, state.fleets_ships, jnp.int32(0))

    P = MAX_PLANETS
    H = max_horizon + 1
    grid = jnp.zeros((P, H, num_agents), dtype=jnp.int32)
    grid = grid.at[safe_p, safe_t, safe_a].add(safe_ships)
    return grid


build_arrival_grid_jit = jax.jit(
    build_arrival_grid, static_argnames=("max_horizon", "num_agents")
)
