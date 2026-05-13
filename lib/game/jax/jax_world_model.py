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


# ---------------------------------------------------------------------------
# Sub-phase 2c: simulate_planet_timeline (lax.scan over horizon)
# ---------------------------------------------------------------------------


def _resolve_arrivals_jax(
    garrison_owner: jnp.ndarray,   # () int32
    garrison_ships: jnp.ndarray,   # () int32
    by_owner: jnp.ndarray,         # (A,) int32
):
    """JAX equivalent of `lib.combat.resolve_arrivals` for a single
    (planet, step). Pure-functional, scalar inputs (use vmap to batch).

    Rules (mirror `data/README.md` + scalar implementation):
      1. Same-step arrivals are already grouped by owner (caller's
         responsibility — passed as `by_owner[A]`).
      2. Largest attacker fights second-largest; survivor = top - second.
      3. Two-way tie (top == second) → all attackers destroyed
         (survivor_ships = 0).
      4. survivor vs garrison:
         - same owner → garrison += survivor_ships
         - different owner → garrison -= survivor_ships; if negative,
           owner flips and ships = |remaining|.
    """
    top_owner = jnp.argmax(by_owner)
    top_ships = by_owner[top_owner]
    masked = by_owner.at[top_owner].set(jnp.int32(-1))
    second_owner = jnp.argmax(masked)
    second_ships = jnp.maximum(masked[second_owner], jnp.int32(0))

    has_attackers = top_ships > 0
    tie = (top_ships == second_ships) & has_attackers
    survivor_ships = jnp.where(tie, jnp.int32(0), top_ships - second_ships)
    has_survivor = survivor_ships > 0
    survivor_owner = jnp.where(has_survivor, top_owner, jnp.int32(-1))

    has_combat = has_attackers & has_survivor
    same = garrison_owner == survivor_owner
    diff = garrison_ships - survivor_ships
    flip = diff < 0

    new_ships_same = garrison_ships + survivor_ships
    new_ships_diff_keep = diff
    new_ships_diff_flip = -diff
    new_ships_diff = jnp.where(flip, new_ships_diff_flip, new_ships_diff_keep)
    new_owner_diff = jnp.where(flip, survivor_owner, garrison_owner)

    new_ships = jnp.where(
        has_combat,
        jnp.where(same, new_ships_same, new_ships_diff),
        garrison_ships,
    )
    new_owner = jnp.where(
        has_combat & ~same, new_owner_diff, garrison_owner,
    )
    new_ships = jnp.maximum(new_ships, jnp.int32(0))
    return new_owner, new_ships


def simulate_planet_timeline_jax(
    initial_owner: jnp.ndarray,   # () int32
    initial_ships: jnp.ndarray,   # () int32
    production: jnp.ndarray,      # () int32
    arrivals_by_step: jnp.ndarray,  # (max_horizon+1, num_agents) int32
    max_horizon: int = DEFAULT_HORIZON,
):
    """For ONE planet, step-by-step ownership/garrison simulation over
    `max_horizon` steps. Mirror of `lib.world_model.simulate_planet_timeline`.

    Per step t in [1, horizon]:
      1. If owned (owner != -1): ships += production.
      2. Apply combat with `arrivals_by_step[t, :]` (a per-owner ship
         total) via `_resolve_arrivals_jax`.
      3. Record (owner, ships).

    Returns `(owner_per_step, ships_per_step)` both of shape
    `(max_horizon+1,)`. Index 0 is the initial state.

    Vmap over the P axis to batch across planets.
    """
    def body(carry, t):
        owner, ships = carry
        # Production: only if owned.
        ships = jnp.where(
            owner != jnp.int32(-1),
            ships + production,
            ships,
        )
        by_owner = arrivals_by_step[t, :]
        new_owner, new_ships = _resolve_arrivals_jax(owner, ships, by_owner)
        return (new_owner, new_ships), (new_owner, new_ships)

    steps = jnp.arange(1, max_horizon + 1)
    _, (owners_stream, ships_stream) = jax.lax.scan(
        body, (initial_owner, jnp.maximum(initial_ships, jnp.int32(0))), steps
    )
    # Prepend initial state for owner_per_step[0] / ships_per_step[0].
    owners_per_step = jnp.concatenate(
        [jnp.array([initial_owner], dtype=jnp.int32), owners_stream]
    )
    ships_per_step = jnp.concatenate(
        [jnp.array([jnp.maximum(initial_ships, jnp.int32(0))], dtype=jnp.int32), ships_stream]
    )
    return owners_per_step, ships_per_step


simulate_planet_timeline_jax_jit = jax.jit(
    simulate_planet_timeline_jax, static_argnames=("max_horizon",)
)


def simulate_all_timelines(
    state,                            # GameState
    arrival_grid: jnp.ndarray,        # (P_max, H+1, A) int32
    max_horizon: int = DEFAULT_HORIZON,
):
    """Per-planet vmap'd version. Returns:
      - `owners_per_step[P_max, H+1]` int32
      - `ships_per_step[P_max, H+1]`  int32
    """
    # vmap over P. arrival_grid is (P, H+1, A); take per-planet slice.
    sim_one = lambda owner, ships, prod, arrivals: simulate_planet_timeline_jax(
        owner, ships, prod, arrivals, max_horizon=max_horizon,
    )
    owners_stream, ships_stream = jax.vmap(sim_one)(
        state.planets_owner,
        state.planets_ships,
        state.planets_prod,
        arrival_grid,
    )
    return owners_stream, ships_stream


simulate_all_timelines_jit = jax.jit(
    simulate_all_timelines, static_argnames=("max_horizon",)
)


# ---------------------------------------------------------------------------
# Sub-phase 2d: JaxWorldModel (Pytree wrapping timelines + queries)
# ---------------------------------------------------------------------------


from typing import NamedTuple


class JaxWorldModel(NamedTuple):
    """JAX analogue of `lib.world_model.WorldModel`. Carries the
    pre-computed per-planet timelines + arrival grid; sub-phase 3
    (JAX missions) consumes via the query helpers below.

    Pytree-compatible (jit/vmap'd). Build via `build_world_model()`.
    """

    # (P_max, H+1) int32 — owner per (planet_slot, step).
    owners_at: jnp.ndarray
    # (P_max, H+1) int32 — ships per (planet_slot, step).
    ships_at: jnp.ndarray
    # (P_max, H+1, A) int32 — incoming ships per (planet, step, owner).
    arrival_grid: jnp.ndarray
    # () int32 — the horizon used.
    horizon: jnp.ndarray


def build_world_model(
    state,                              # GameState
    max_horizon: int = DEFAULT_HORIZON,
    num_agents: int = 4,
) -> JaxWorldModel:
    """End-to-end build: scalar_to_jax → grid → timelines → JaxWorldModel.

    Equivalent to `lib.world_model.WorldModel.from_world(world)`. Costs
    one F×P raycast + a 3D scatter + a vmap'd lax.scan over horizon.
    """
    arrival_grid = build_arrival_grid(
        state, max_horizon=max_horizon, num_agents=num_agents
    )
    owners_at, ships_at = simulate_all_timelines(
        state, arrival_grid, max_horizon=max_horizon
    )
    return JaxWorldModel(
        owners_at=owners_at,
        ships_at=ships_at,
        arrival_grid=arrival_grid,
        horizon=jnp.int32(max_horizon),
    )


build_world_model_jit = jax.jit(
    build_world_model, static_argnames=("max_horizon", "num_agents")
)


# ---------------------------------------------------------------------------
# Query helpers — mirror WorldModel.owner_at / ships_at / incoming_enemy_eta
# ---------------------------------------------------------------------------


def owner_at(world: JaxWorldModel, planet_slot: jnp.ndarray, step: jnp.ndarray):
    """Predicted owner of `planet_slot` at `step`. Clamps step to
    [0, horizon]. Returns int32 (-1 = neutral)."""
    t = jnp.clip(step, jnp.int32(0), world.horizon)
    return world.owners_at[planet_slot, t]


def ships_at(world: JaxWorldModel, planet_slot: jnp.ndarray, step: jnp.ndarray):
    """Predicted garrison at `planet_slot` at `step`."""
    t = jnp.clip(step, jnp.int32(0), world.horizon)
    return world.ships_at[planet_slot, t]


def incoming_enemy_eta(
    world: JaxWorldModel,
    planet_slot: jnp.ndarray,    # () int32
    my_id: jnp.ndarray,          # () int32
):
    """Min step at which a non-`my_id` fleet arrives at `planet_slot`.

    Returns `horizon + 1` sentinel if no enemy arrival within horizon.
    """
    # arrival_grid[p, t, a]: ships from `a` arriving at `p` at step `t`.
    # Want: min t > 0 where sum_{a != my_id} arrival_grid[p, t, a] > 0.
    grid_p = world.arrival_grid[planet_slot]  # (H+1, A)
    A = grid_p.shape[1]
    # Mask out our own arrivals.
    other_mask = jnp.arange(A) != my_id
    other_ships = jnp.sum(grid_p * other_mask[None, :].astype(grid_p.dtype), axis=1)
    has = other_ships > 0
    # First True along the step axis.
    any_hit = jnp.any(has)
    first = jnp.argmax(has.astype(jnp.int32))
    return jnp.where(any_hit, first, world.horizon + jnp.int32(1))
