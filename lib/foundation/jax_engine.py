"""Foundation-layer wrapper around the JAX game engine.

The orphan asset diagnosis from the foundation plan: `lib/game/jax/
jax_interpreter.py::jax_step` exists, is parity-tested at float32
tolerance, and is JIT-compiled — but no live agent calls it. This
module provides the thin entry points that bridge the JAX engine to
foundation consumers:

- `step(state, pids, angles, ships)` — one turn, single state.
  Direct delegation to `jax_step` (already JIT'd at module load via
  `jax_step_jit`).
- `step_from_action_tensor(state, action_tensor, candidate, turn)` —
  convenience: extract the `(candidate, turn)` slice of an
  `ActionTensor` and call `step`. Not on the JIT'd hot path
  (Python-level indexing).
- `step_batched(states, pids_b, angles_b, ships_b)` — `jax.vmap` over
  a leading games / candidates batch axis. The primitive that Step 9's
  candidate evaluator builds on.
- `rollout_python(state, action_fn, horizon, memory)` — Python loop
  over turns; suitable for strategies that aren't yet pure JAX
  (heuristic mission frameworks, search trees). The `lax.scan`
  variant for pure-JAX strategies lands in Step 9.
- `clone_state(state)` — identity (Pytrees are immutable in JAX so
  cloning is free); documented for callers migrating from `fast_sim`'s
  `clone(snap)` API.

Float-tolerance contract:
- combat / ownership / ship counts / fleet list: bit-exact vs the
  scalar interpreter (these are integer / boolean fields).
- planet & fleet positions: float32 tolerance ≤ 1e-3 (JAX uses float32
  vs the scalar interpreter's float64; the env's swept-pair collision
  uses tolerances much larger than 1e-3 so this drift is invisible at
  the gameplay level).

See `tests/test_jax_engine_random_action_parity.py` for the random-
action parity assertion.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from lib.foundation.actions import ActionTensor
from lib.foundation.memory import EmptyMemory, Memory
from lib.game.jax.jax_interpreter import jax_step, jax_step_jit
from lib.game.jax.jax_types import GameState


# Re-export the JIT'd primitive so callers can pick whichever entry
# point fits their context.
__all__ = [
    "step",
    "step_jit",
    "step_from_action_tensor",
    "step_batched",
    "step_batched_jit",
    "rollout_python",
    "clone_state",
]


def step(
    state: GameState,
    pids: jnp.ndarray,
    angles: jnp.ndarray,
    ships: jnp.ndarray,
) -> GameState:
    """Apply one full game turn. Thin wrapper over `jax_step`.

    Action arrays must be shape `(MAX_AGENTS, MAX_LAUNCH_PER_AGENT)`:
        pids   — int32, `-1` for no-op slots
        angles — float32, radians
        ships  — int32, `0` for no-op slots

    JIT-friendly if all inputs are `jnp.ndarray`. Returns a new
    `GameState`; the input is not mutated (Pytrees are immutable).
    """
    return jax_step(state, pids, angles, ships)


# JIT'd singleton — reuses the underlying `jax_step_jit` compile cache.
step_jit = jax_step_jit


def step_from_action_tensor(
    state: GameState,
    action_tensor: ActionTensor,
    candidate: int = 0,
    turn: int = 0,
) -> GameState:
    """Extract the `(candidate, turn)` slice of `action_tensor` and
    step `state`. Convenience for non-batched callers.

    NOT on the JIT'd hot path — `candidate` / `turn` are Python ints
    (would need to be static under JIT). Batched evaluation should
    use `step_batched` directly with pre-sliced arrays.
    """
    pids = jnp.asarray(action_tensor.pids[candidate, turn])
    angles = jnp.asarray(action_tensor.angles[candidate, turn])
    ships = jnp.asarray(action_tensor.ships[candidate, turn])
    return step(state, pids, angles, ships)


def step_batched(
    states: GameState,
    pids_b: jnp.ndarray,
    angles_b: jnp.ndarray,
    ships_b: jnp.ndarray,
) -> GameState:
    """`jax.vmap` of `step` over a leading batch axis.

    Inputs must share the leading dimension B (batch / candidates /
    parallel games). Each field of `states` has shape `(B, ...)`;
    `pids_b` / `angles_b` / `ships_b` have shape `(B, MAX_AGENTS,
    MAX_LAUNCH_PER_AGENT)`.

    Returns a `GameState` with the same leading B axis. The primitive
    behind Step 9's batched candidate evaluator and Step 10's parallel
    self-play arena.
    """
    return jax.vmap(step)(states, pids_b, angles_b, ships_b)


step_batched_jit = jax.jit(step_batched)


def rollout_python(
    state: GameState,
    action_fn: Callable[[GameState, int, Memory], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Memory]],
    horizon: int,
    memory: Optional[Memory] = None,
) -> tuple[GameState, Memory]:
    """Run `horizon` turns, calling `action_fn(state, turn_idx,
    memory)` each turn to obtain `(pids, angles, ships, new_memory)`.

    Python loop — suitable for strategies that aren't pure-JAX
    (heuristic mission frameworks, search trees). For pure-JAX
    strategies, prefer the `lax.scan` rollout that Step 9 ships.

    Terminates early on `state.done`. Memory threads through; if the
    caller passes `None`, an `EmptyMemory` is used.
    """
    if memory is None:
        memory = EmptyMemory()
    for t in range(horizon):
        # Top-of-loop check: a done state means the game has already
        # terminated and there's nothing to do. Avoids calling
        # action_fn on a done state and avoids re-stepping through
        # `terminate`, which would re-evaluate the done condition
        # (and could overwrite a manually-set done flag).
        # `bool(state.done)` triggers a device transfer; fine in a
        # Python rollout, but Step 9's `lax.scan` variant uses
        # `jnp.where` masking instead.
        if bool(state.done):
            break
        pids, angles, ships, memory = action_fn(state, t, memory)
        state = step(state, pids, angles, ships)
    return state, memory


def clone_state(state: GameState) -> GameState:
    """Return `state` unchanged. Documented as identity so callers
    migrating from `lib.fast_sim.clone(snap)` know there's no copying
    cost: `GameState` is an immutable Pytree of immutable JAX arrays,
    so any "clone" is a no-op.

    Provided as a stable symbol so the foundation API can substitute
    a real copy in the future (e.g., if mutable scratch state were
    added to `GameState`) without breaking call sites.
    """
    return state
