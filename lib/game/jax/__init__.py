"""JAX port of the orbit_wars game engine.

Multi-session sprint to enable GPU-batched A/B testing on Kaggle Kernels.
This package builds incrementally; see the sub-phase roadmap in
`/root/.claude/plans/your-task-is-to-unified-hellman.md`.

Status:
- Sub-phase 1 (this session): scaffolding, GameState Pytree, conversions,
  smoke test that scalar↔JAX round-trip preserves state.
- Sub-phases 2-8: per-phase JAX physics, WorldModel, missions, mechanism
  stack, score_candidate, v7_0 wrapper, Kaggle Kernel deployment.

Design:
- Game state is a Pytree of jnp arrays with fixed padding (MAX_PLANETS=80,
  MAX_FLEETS=256). Masks indicate live entries.
- Game INIT stays in Python (planet generation uses Python's
  `random.Random` seeded by a string for comet RNG — can't reproduce
  byte-exact in JAX). Init pre-computes all comet spawn data and
  stores it in the state.
- Per-step PHYSICS will be pure JAX, jit-able and vmap-able over the
  N-game axis. Live ladder submissions stay pure-Python.
"""

from lib.game.jax.jax_types import (
    GameState,
    MAX_PLANETS,
    MAX_FLEETS,
    MAX_COMET_GROUPS,
    MAX_COMET_PATH_LEN,
    NUM_COMET_SPAWNS,
    MAX_LAUNCH_PER_AGENT,
    MAX_AGENTS,
)
from lib.game.jax.conversions import scalar_to_jax, jax_to_scalar
from lib.game.jax.jax_interpreter import (
    production_tick,
    planet_path_compute,
    comet_expire,
    comet_path_advance,
    comet_spawn,
    apply_planet_movement,
    fleet_launch,
    fleet_movement,
    combat_resolution,
    terminate,
    remove_expired_comets_mid_step,
    jax_step,
    swept_pair_hit_batch,
    jax_step_partial,
    production_tick_jit,
    planet_path_compute_jit,
    comet_expire_jit,
    comet_path_advance_jit,
    comet_spawn_jit,
    apply_planet_movement_jit,
    fleet_launch_jit,
    fleet_movement_jit,
    combat_resolution_jit, terminate_jit, jax_step_jit,
    swept_pair_hit_batch_jit,
    jax_step_partial_jit,
)

__all__ = [
    "GameState",
    "MAX_PLANETS", "MAX_FLEETS", "MAX_COMET_GROUPS",
    "MAX_COMET_PATH_LEN", "NUM_COMET_SPAWNS",
    "MAX_LAUNCH_PER_AGENT", "MAX_AGENTS",
    "scalar_to_jax", "jax_to_scalar",
    "production_tick", "planet_path_compute",
    "comet_expire", "comet_path_advance", "comet_spawn",
    "apply_planet_movement",
    "fleet_launch",
    "fleet_movement", "combat_resolution", "terminate",
    "remove_expired_comets_mid_step", "jax_step",
    "swept_pair_hit_batch",
    "jax_step_partial",
    "production_tick_jit", "planet_path_compute_jit",
    "comet_expire_jit", "comet_path_advance_jit", "comet_spawn_jit",
    "apply_planet_movement_jit", "fleet_launch_jit",
    "fleet_movement_jit", "combat_resolution_jit",
    "terminate_jit", "jax_step_jit",
    "swept_pair_hit_batch_jit", "jax_step_partial_jit",
]
