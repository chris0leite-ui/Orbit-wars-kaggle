"""Convert between the scalar interpreter's state object and the JAX
`GameState` Pytree.

Why we need this: agents and bundles operate on the scalar `state[i].
observation` Struct (planets/fleets as lists-of-lists). Our JAX port
operates on padded ndarray Pytree. The conversion is the bridge.

`scalar_to_jax(scalar_state, episode_seed)` — used by `jax_init` to
seed a JAX game from a freshly-initialised scalar state. Pre-computes
the comet spawn schedule by replaying `generate_comet_paths` on the
initial planets for each of the 5 spawn boundaries.

`jax_to_scalar(jax_state)` — used by tests + the future JAX agent
wrapper to convert a JAX state back into a scalar-compatible obs for
parity testing.
"""

from __future__ import annotations

import random
from typing import Any

import jax.numpy as jnp
import numpy as np

from lib.game.interpreter import (
    COMET_PRODUCTION, COMET_RADIUS, COMET_SPAWN_STEPS,
    generate_comet_paths,
)
from lib.game.jax.jax_types import (
    GameState,
    MAX_PLANETS, MAX_FLEETS, MAX_COMET_PATH_LEN,
    MAX_COMET_PATHS_PER_GROUP, NUM_COMET_SPAWNS,
)


def scalar_to_jax(state, episode_seed: int) -> GameState:
    """Build a JAX GameState from a scalar `state` (list of Struct, one
    per seat) after the init phase has run.

    Pre-computes the 5 comet-spawn results by calling
    `generate_comet_paths` 5 times with the right RNG keys. This
    eliminates Python's string-keyed `random.Random` from the JAX
    step-time path; spawns become a table lookup.
    """
    obs0 = state[0].observation
    num_agents = len(state)

    # --- Planets (padded) ---
    planets = list(obs0.planets)
    initial_planets = list(obs0.initial_planets)
    comet_planet_ids = set(obs0.comet_planet_ids) if obs0.comet_planet_ids else set()

    P = len(planets)
    assert P <= MAX_PLANETS, f"too many planets ({P} > {MAX_PLANETS})"

    planets_x = np.zeros(MAX_PLANETS, dtype=np.float32)
    planets_y = np.zeros(MAX_PLANETS, dtype=np.float32)
    planets_new_x = np.zeros(MAX_PLANETS, dtype=np.float32)
    planets_new_y = np.zeros(MAX_PLANETS, dtype=np.float32)
    planets_id = -np.ones(MAX_PLANETS, dtype=np.int32)
    planets_owner = -np.ones(MAX_PLANETS, dtype=np.int32)
    planets_ships = np.zeros(MAX_PLANETS, dtype=np.int32)
    planets_prod = np.zeros(MAX_PLANETS, dtype=np.int32)
    planets_radius = np.zeros(MAX_PLANETS, dtype=np.float32)
    planets_alive = np.zeros(MAX_PLANETS, dtype=bool)
    initial_x = np.zeros(MAX_PLANETS, dtype=np.float32)
    initial_y = np.zeros(MAX_PLANETS, dtype=np.float32)
    is_comet = np.zeros(MAX_PLANETS, dtype=bool)
    planet_comet_spawn = -np.ones(MAX_PLANETS, dtype=np.int32)
    planet_comet_path = -np.ones(MAX_PLANETS, dtype=np.int32)

    # Build pid -> array index map (scalar uses arbitrary planet ids; we
    # pack into 0..P-1 for the JAX array). NOTE: this means
    # planets_*[i] != planet with id == i; lookup via planet_id_to_idx.
    pid_to_idx: dict[int, int] = {}
    for i, p in enumerate(planets):
        pid = int(p[0])
        pid_to_idx[pid] = i
        planets_x[i] = p[2]
        planets_y[i] = p[3]
        # Initialise new_x/new_y to current (no movement applied yet).
        planets_new_x[i] = p[2]
        planets_new_y[i] = p[3]
        planets_id[i] = pid
        planets_owner[i] = p[1]
        planets_ships[i] = p[5]
        planets_prod[i] = p[6]
        planets_radius[i] = p[4]
        planets_alive[i] = True
        is_comet[i] = pid in comet_planet_ids

    # Initial positions (for orbital rotation in jax_step's planet path
    # phase). `initial_planets` is a parallel list with same ids.
    for ip in initial_planets:
        pid = int(ip[0])
        idx = pid_to_idx.get(pid)
        if idx is not None:
            initial_x[idx] = ip[2]
            initial_y[idx] = ip[3]

    # --- Fleets (padded) ---
    fleets = list(obs0.fleets)
    F = len(fleets)
    assert F <= MAX_FLEETS, f"too many fleets ({F} > {MAX_FLEETS})"

    fleets_x = np.zeros(MAX_FLEETS, dtype=np.float32)
    fleets_y = np.zeros(MAX_FLEETS, dtype=np.float32)
    fleets_angle = np.zeros(MAX_FLEETS, dtype=np.float32)
    fleets_owner = -np.ones(MAX_FLEETS, dtype=np.int32)
    fleets_ships = np.zeros(MAX_FLEETS, dtype=np.int32)
    fleets_from_planet = -np.ones(MAX_FLEETS, dtype=np.int32)
    fleets_id = -np.ones(MAX_FLEETS, dtype=np.int32)
    fleets_alive = np.zeros(MAX_FLEETS, dtype=bool)
    for i, f in enumerate(fleets):
        fleets_x[i] = f[2]
        fleets_y[i] = f[3]
        fleets_angle[i] = f[4]
        fleets_owner[i] = f[1]
        fleets_ships[i] = f[6]
        fleets_from_planet[i] = f[5]
        fleets_id[i] = f[0]
        fleets_alive[i] = True

    # --- Comet schedule: pre-compute the 5 spawn results -----------
    # Each spawn at step S uses `random.Random(f"orbit_wars-comet-{seed}-{S+1}")`.
    # We replay that exactly here (Python), capture the paths + ship
    # count, and store in JAX state for jax_step to look up.
    comet_step = np.zeros(NUM_COMET_SPAWNS, dtype=np.int32)
    comet_paths_xy = np.zeros(
        (NUM_COMET_SPAWNS, MAX_COMET_PATHS_PER_GROUP, MAX_COMET_PATH_LEN, 2),
        dtype=np.float32,
    )
    comet_paths_len = np.zeros(
        (NUM_COMET_SPAWNS, MAX_COMET_PATHS_PER_GROUP), dtype=np.int32,
    )
    comet_ships_arr = np.zeros(NUM_COMET_SPAWNS, dtype=np.int32)
    comet_valid_arr = np.zeros(NUM_COMET_SPAWNS, dtype=bool)
    comet_path_index = -np.ones(NUM_COMET_SPAWNS, dtype=np.int32)
    comet_spawned_arr = np.zeros(NUM_COMET_SPAWNS, dtype=bool)
    comet_planet_idx = -np.ones(
        (NUM_COMET_SPAWNS, MAX_COMET_PATHS_PER_GROUP), dtype=np.int32,
    )

    comet_speed = 4.0  # default; matches DEFAULT_CONFIG in fast_sim.py
    angular_velocity = float(obs0.angular_velocity)
    for k, spawn_step in enumerate(COMET_SPAWN_STEPS):
        comet_step[k] = spawn_step
        # Replay scalar comet generation deterministically.
        rng = random.Random(f"orbit_wars-comet-{episode_seed}-{spawn_step}")
        paths = generate_comet_paths(
            initial_planets,
            angular_velocity,
            spawn_step,
            list(obs0.comet_planet_ids) if obs0.comet_planet_ids else [],
            comet_speed,
            rng=rng,
        )
        if paths:
            comet_valid_arr[k] = True
            for j, path in enumerate(paths[:MAX_COMET_PATHS_PER_GROUP]):
                L = min(len(path), MAX_COMET_PATH_LEN)
                comet_paths_len[k, j] = L
                for t in range(L):
                    comet_paths_xy[k, j, t, 0] = path[t][0]
                    comet_paths_xy[k, j, t, 1] = path[t][1]
            # 4 randint(1, 99) draws — same as scalar interpreter.
            cs = min(
                rng.randint(1, 99),
                rng.randint(1, 99),
                rng.randint(1, 99),
                rng.randint(1, 99),
            )
            comet_ships_arr[k] = cs
        else:
            comet_valid_arr[k] = False

    # --- In-flight comets: link existing comet planets back to spawn groups ---
    # When converting a mid-game state, comets may already be on the board.
    # The pre-baked spawn tables are populated, but `planet_comet_spawn`,
    # `planet_comet_path`, `comet_planet_idx`, `comet_path_index`, and
    # `comet_spawned` need to be wired up so that downstream code (missions,
    # path advance, expiration) sees the correct group/path mapping.
    current_step = int(obs0.get("step", 0))
    spawn_step_to_k = {int(s): k for k, s in enumerate(COMET_SPAWN_STEPS)}
    for group in (obs0.comets or []):
        if hasattr(group, "keys"):
            g_planet_ids = list(group["planet_ids"])
            g_path_index = int(group["path_index"])
        else:
            g_planet_ids = list(group.planet_ids)
            g_path_index = int(group.path_index)
        # Spawn step = current step - path_index advances. Each path_advance
        # increments by 1, and the first advance happens on the spawn step.
        spawn_step = current_step - g_path_index
        k = spawn_step_to_k.get(int(spawn_step))
        if k is None:
            # Comet group whose spawn step doesn't match the schedule; skip.
            continue
        comet_path_index[k] = g_path_index
        comet_spawned_arr[k] = True
        for j, pid in enumerate(g_planet_ids[:MAX_COMET_PATHS_PER_GROUP]):
            slot = pid_to_idx.get(int(pid))
            if slot is None:
                continue
            planet_comet_spawn[slot] = k
            planet_comet_path[slot] = j
            comet_planet_idx[k, j] = slot

    # --- Scalars + rewards ---
    rewards = np.zeros(4, dtype=np.int32)
    for i in range(min(num_agents, 4)):
        rewards[i] = state[i].reward or 0

    return GameState(
        planets_x=jnp.asarray(planets_x),
        planets_y=jnp.asarray(planets_y),
        planets_new_x=jnp.asarray(planets_new_x),
        planets_new_y=jnp.asarray(planets_new_y),
        planets_id=jnp.asarray(planets_id),
        planets_owner=jnp.asarray(planets_owner),
        planets_ships=jnp.asarray(planets_ships),
        planets_prod=jnp.asarray(planets_prod),
        planets_radius=jnp.asarray(planets_radius),
        planets_alive=jnp.asarray(planets_alive),
        initial_x=jnp.asarray(initial_x),
        initial_y=jnp.asarray(initial_y),
        is_comet=jnp.asarray(is_comet),
        planet_comet_spawn=jnp.asarray(planet_comet_spawn),
        planet_comet_path=jnp.asarray(planet_comet_path),
        fleets_x=jnp.asarray(fleets_x),
        fleets_y=jnp.asarray(fleets_y),
        fleets_angle=jnp.asarray(fleets_angle),
        fleets_owner=jnp.asarray(fleets_owner),
        fleets_ships=jnp.asarray(fleets_ships),
        fleets_from_planet=jnp.asarray(fleets_from_planet),
        fleets_id=jnp.asarray(fleets_id),
        fleets_alive=jnp.asarray(fleets_alive),
        comet_step=jnp.asarray(comet_step),
        comet_paths_xy=jnp.asarray(comet_paths_xy),
        comet_paths_len=jnp.asarray(comet_paths_len),
        comet_ships=jnp.asarray(comet_ships_arr),
        comet_valid=jnp.asarray(comet_valid_arr),
        comet_path_index=jnp.asarray(comet_path_index),
        comet_spawned=jnp.asarray(comet_spawned_arr),
        comet_planet_idx=jnp.asarray(comet_planet_idx),
        step=jnp.asarray(int(obs0.get("step", 0)), dtype=jnp.int32),
        angular_velocity=jnp.asarray(angular_velocity, dtype=jnp.float32),
        episode_seed=jnp.asarray(int(episode_seed), dtype=jnp.int32),
        done=jnp.asarray(False),
        num_agents=jnp.asarray(num_agents, dtype=jnp.int32),
        next_fleet_id=jnp.asarray(int(obs0.next_fleet_id), dtype=jnp.int32),
        rewards=jnp.asarray(rewards),
    )


def actions_to_jax(per_agent_actions: list, num_agents: int = 2):
    """Convert per-agent scalar actions (list of [pid, angle, ships] per
    agent) into the three padded JAX tensors that `jax_step` expects.

    Each tensor shape: `(MAX_AGENTS, MAX_LAUNCH_PER_AGENT)`. Sentinel
    `pid == -1` flags unused slots.
    """
    from lib.game.jax.jax_types import MAX_AGENTS, MAX_LAUNCH_PER_AGENT
    pids = -np.ones((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=np.int32)
    angles = np.zeros((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=np.float32)
    ships = np.zeros((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=np.int32)
    for a, agent_actions in enumerate(per_agent_actions[:MAX_AGENTS]):
        if not agent_actions or not isinstance(agent_actions, list):
            continue
        for k, mv in enumerate(agent_actions[:MAX_LAUNCH_PER_AGENT]):
            if len(mv) != 3:
                continue
            pids[a, k] = int(mv[0])
            angles[a, k] = float(mv[1])
            ships[a, k] = int(mv[2])
    return (
        jnp.asarray(pids),
        jnp.asarray(angles),
        jnp.asarray(ships),
    )


def jax_to_scalar_obs(s: GameState) -> dict:
    """Convert a JAX GameState back to a scalar-style observation dict.

    Returns the shape consumers expect (planets/fleets as list-of-list)
    so the result can be fed to scalar `interpreter()`, mission
    builders, etc. Used by parity tests and the future JAX agent
    wrapper.
    """
    planets_alive = np.asarray(s.planets_alive)
    fleets_alive = np.asarray(s.fleets_alive)

    planets = []
    # Maintain the original order; planet[0] (id) is the array index
    # since we packed sequentially in scalar_to_jax. Round-trip preserves
    # ids only if the scalar side hadn't deleted any planets in the
    # meantime (jax_step's expire/spawn updates the alive mask).
    for i in range(MAX_PLANETS):
        if not planets_alive[i]:
            continue
        planets.append([
            int(i),  # pid placeholder (the array index)
            int(s.planets_owner[i]),
            float(s.planets_x[i]),
            float(s.planets_y[i]),
            float(s.planets_radius[i]),
            int(s.planets_ships[i]),
            int(s.planets_prod[i]),
        ])

    fleets = []
    for i in range(MAX_FLEETS):
        if not fleets_alive[i]:
            continue
        fleets.append([
            int(s.fleets_id[i]),
            int(s.fleets_owner[i]),
            float(s.fleets_x[i]),
            float(s.fleets_y[i]),
            float(s.fleets_angle[i]),
            int(s.fleets_from_planet[i]),
            int(s.fleets_ships[i]),
        ])

    return {
        "planets": planets,
        "fleets": fleets,
        "angular_velocity": float(s.angular_velocity),
        "step": int(s.step),
        "next_fleet_id": int(s.next_fleet_id),
        "done": bool(s.done),
    }


# Alias retained for backward-compat with imports
jax_to_scalar = jax_to_scalar_obs
