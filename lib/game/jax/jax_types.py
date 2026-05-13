"""GameState Pytree — fixed-shape padded JAX arrays mirroring the scalar
interpreter's mutable state.

A single game is one instance of `GameState`. Batching N games is
achieved by `jax.vmap` over the leading axis of every field.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


# --- Padding bounds (sized for typical orbit_wars game) ---------------------

# Max non-comet planets that can exist in any game state. The initial
# generator builds 20-60; up to 4 comets × 5 spawn boundaries = 20 more
# during the episode. 80 is a comfortable upper bound observed across
# 150+ seeds.
MAX_PLANETS = 80

# Max in-flight fleets at any moment. Peak observed in mid-game heavy
# play is ~150; 256 is the conservative bound to avoid resize.
MAX_FLEETS = 256

# Comet spawn schedule: 5 boundaries per episode (steps 50, 150, 250,
# 350, 450); each spawn creates 4 symmetric paths.
NUM_COMET_SPAWNS = 5
MAX_COMET_GROUPS = NUM_COMET_SPAWNS  # one group per spawn
MAX_COMET_PATHS_PER_GROUP = 4         # 4-fold symmetry — always 4

# Max length of a single comet path (visible-on-board sample count).
# Per `generate_comet_paths`, paths are validated to 5 ≤ len ≤ 40.
MAX_COMET_PATH_LEN = 40

# Max simultaneous launches per agent per turn. v3.5.1 typically
# launches 2-4; v7_0 lookahead candidates can produce up to ~6.
# 20 is a comfortable upper bound.
MAX_LAUNCH_PER_AGENT = 20

# Max number of agents (2P or 4P games).
MAX_AGENTS = 4


class GameState(NamedTuple):
    """Padded JAX game state. Mirrors lib/game/interpreter.py's mutable
    fields. All shapes are static (jit-friendly); `*_alive` masks gate
    valid entries.

    Single-game shapes; batch via `jax.vmap` over the leading axis.
    """

    # --- Planets (initial + comet-spawned, all in one padded array) ---
    # Index 0..P_initial-1 are the initial generated planets; comets
    # added during the episode occupy higher indices.
    planets_x: jnp.ndarray         # (P_max,) float32 — CURRENT (post-applied) x
    planets_y: jnp.ndarray         # (P_max,) float32 — CURRENT (post-applied) y
    # Deferred new positions — written by planet_path_compute and
    # comet_path_advance, read by fleet_movement's sweep-pair check,
    # applied to planets_x/y by apply_planet_movement at end of step.
    # Mirrors the scalar interpreter's `planet_paths` dict.
    planets_new_x: jnp.ndarray     # (P_max,) float32
    planets_new_y: jnp.ndarray     # (P_max,) float32
    # Scalar planet IDs (the int returned by `obs.planets[i][0]`).
    # Lets fleet_launch translate action pid → JAX slot via comparison
    # against this array. -1 in dead slots.
    planets_id: jnp.ndarray        # (P_max,) int32
    planets_owner: jnp.ndarray     # (P_max,) int32 — -1 = neutral
    planets_ships: jnp.ndarray     # (P_max,) int32
    planets_prod: jnp.ndarray      # (P_max,) int32
    planets_radius: jnp.ndarray    # (P_max,) float32
    planets_alive: jnp.ndarray     # (P_max,) bool
    # Initial planet x,y for the orbital rotation formula; doesn't
    # change during the episode (comets fix their entry at spawn).
    initial_x: jnp.ndarray         # (P_max,) float32
    initial_y: jnp.ndarray         # (P_max,) float32
    # Whether each planet is a comet (rotates differently / can expire).
    is_comet: jnp.ndarray          # (P_max,) bool
    # For comet planets, which spawn (0..S-1) and which path (0..3) it
    # came from — used by the per-step expiration check. -1 for
    # non-comet planets.
    planet_comet_spawn: jnp.ndarray  # (P_max,) int32
    planet_comet_path: jnp.ndarray   # (P_max,) int32

    # --- In-flight fleets ---
    fleets_x: jnp.ndarray          # (F_max,) float32
    fleets_y: jnp.ndarray          # (F_max,) float32
    fleets_angle: jnp.ndarray      # (F_max,) float32
    fleets_owner: jnp.ndarray      # (F_max,) int32
    fleets_ships: jnp.ndarray      # (F_max,) int32
    fleets_from_planet: jnp.ndarray  # (F_max,) int32 — source planet id
    fleets_id: jnp.ndarray         # (F_max,) int32 — unique within game
    fleets_alive: jnp.ndarray      # (F_max,) bool

    # --- Comet schedule (pre-computed in Python at game init) ---
    # For each spawn boundary index k (0..NUM_COMET_SPAWNS-1):
    #   comet_step[k]            — the game step at which it spawns (50, 150, ...)
    #   comet_paths_xy[k]        — (4, MAX_COMET_PATH_LEN, 2) — 4 paths per spawn
    #   comet_paths_len[k]       — (4,) actual length of each path
    #   comet_ships[k]           — int32 — initial ship count for the group
    #   comet_valid[k]           — bool — True if `generate_comet_paths` returned a valid set
    # comet groups currently live in `obs0.comets`; for JAX we flatten:
    #   path_index_per_spawn[k]  — int32 — current position along the path
    #                              (-1 if not yet spawned / already expired)
    comet_step: jnp.ndarray              # (S,) int32
    comet_paths_xy: jnp.ndarray          # (S, 4, L, 2) float32
    comet_paths_len: jnp.ndarray         # (S, 4) int32
    comet_ships: jnp.ndarray             # (S,) int32
    comet_valid: jnp.ndarray             # (S,) bool
    # Per spawn: current path index. Starts at -1 (interpretation:
    # "not yet advanced"); first `comet_path_advance` call after spawn
    # increments to 0, which selects `paths[k, j, 0]`. The
    # `comet_spawned` flag distinguishes "has this spawn fired" from
    # "what's the current index": both pre-spawn and just-spawned states
    # have `comet_path_index == -1`, but only the latter has
    # `comet_spawned == True`, so only the latter increments.
    comet_path_index: jnp.ndarray        # (S,) int32
    comet_spawned: jnp.ndarray           # (S,) bool — True after spawn fired
    # Planet indices (into planets_* arrays) for each spawn's 4 comets.
    # -1 if not spawned. Lets us look up "is this planet a comet from
    # spawn k path j" without scanning.
    comet_planet_idx: jnp.ndarray        # (S, 4) int32

    # --- Scalars ---
    step: jnp.ndarray                    # () int32
    angular_velocity: jnp.ndarray        # () float32
    episode_seed: jnp.ndarray            # () int32
    done: jnp.ndarray                    # () bool
    num_agents: jnp.ndarray              # () int32 (2 or 4)
    next_fleet_id: jnp.ndarray           # () int32
    # Per-seat rewards after termination (-1 / 1, 0 if not done).
    rewards: jnp.ndarray                 # (num_agents,) int32 — actually max 4


# Convenience: shapes documentation
SHAPES = {
    "planets_x": (MAX_PLANETS,),
    "planets_y": (MAX_PLANETS,),
    "planets_new_x": (MAX_PLANETS,),
    "planets_new_y": (MAX_PLANETS,),
    "planets_id": (MAX_PLANETS,),
    "planets_owner": (MAX_PLANETS,),
    "planets_ships": (MAX_PLANETS,),
    "planets_prod": (MAX_PLANETS,),
    "planets_radius": (MAX_PLANETS,),
    "planets_alive": (MAX_PLANETS,),
    "initial_x": (MAX_PLANETS,),
    "initial_y": (MAX_PLANETS,),
    "is_comet": (MAX_PLANETS,),
    "planet_comet_spawn": (MAX_PLANETS,),
    "planet_comet_path": (MAX_PLANETS,),
    "fleets_x": (MAX_FLEETS,),
    "fleets_y": (MAX_FLEETS,),
    "fleets_angle": (MAX_FLEETS,),
    "fleets_owner": (MAX_FLEETS,),
    "fleets_ships": (MAX_FLEETS,),
    "fleets_from_planet": (MAX_FLEETS,),
    "fleets_id": (MAX_FLEETS,),
    "fleets_alive": (MAX_FLEETS,),
    "comet_step": (NUM_COMET_SPAWNS,),
    "comet_paths_xy": (NUM_COMET_SPAWNS, MAX_COMET_PATHS_PER_GROUP, MAX_COMET_PATH_LEN, 2),
    "comet_paths_len": (NUM_COMET_SPAWNS, MAX_COMET_PATHS_PER_GROUP),
    "comet_ships": (NUM_COMET_SPAWNS,),
    "comet_valid": (NUM_COMET_SPAWNS,),
    "comet_path_index": (NUM_COMET_SPAWNS,),
    "comet_spawned": (NUM_COMET_SPAWNS,),
    "comet_planet_idx": (NUM_COMET_SPAWNS, MAX_COMET_PATHS_PER_GROUP),
    "step": (),
    "angular_velocity": (),
    "episode_seed": (),
    "done": (),
    "num_agents": (),
    "next_fleet_id": (),
    "rewards": (4,),
}
