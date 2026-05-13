"""JAX candidate-axis vmap brute-force scorer.

Phase 3C-2 of the post-handover sprint. Given a state and C candidate
turn-1 action bundles, score each candidate by applying it as our seat's
turn-1 action (with opp playing its v3.5.1 mirror) and rolling K-1
follow-up mirror-mirror steps. Vmap over the candidate axis.

Why vmap-over-candidates instead of vmap-over-games (which the JAX A/B
kernel does today): for an offline brute-search "oracle", we want to
evaluate the full action-candidate set on a SINGLE state. The natural
parallel axis is candidates, not games. JIT-vmap-fold gives a roughly
N×-cheaper search over a fixed state than N serial calls.

The candidate input is the same `(C, MAX_LAUNCH_PER_AGENT)` triple shape
that `jax_step` already expects per-agent: `(pids, angles, ships)`. Each
candidate row is the my-seat action for one variant; opp action is
computed once from `state` and shared across all rows.

Public surface:
- `candidate_emits_to_tensors(emit_list, num_agents=2) -> (pids, angles, ships)`
  Stacks a Python list of emit-lists (each from `apply_mechanisms_numpy`)
  into a `(C, MAX_LAUNCH_PER_AGENT)` triple.
- `score_candidates_vmap(state, pids_c, angles_c, ships_c, K, my_id,
  num_agents=2, opp_aggressive=True) -> jnp.ndarray[C]`
  Vmap'd scoring function. Returns one float per candidate.
- `argmax_candidate_vmap(state, pids_c, angles_c, ships_c, K, my_id,
  ...) -> (best_idx: int, best_score: float)`
  Argmax helper for callers that just want the winning candidate.

Parity gate: `tests/test_jax_brute_search.py` asserts the vmap'd output
matches a serial Python loop over `score_candidate_jax` cell-by-cell
within 1e-3 absolute (the rollout is fully deterministic in float32).
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from lib.game.jax.jax_interpreter import jax_step
from lib.game.jax.jax_score import (
    rollout_step_jax_pure,
    value_delta_ships,
)
from lib.game.jax.jax_world_model import build_world_model, DEFAULT_HORIZON
from lib.game.jax.jax_score import policy_emit_jax_pure
from lib.game.jax.jax_types import MAX_AGENTS, MAX_LAUNCH_PER_AGENT


def candidate_emits_to_tensors(
    emit_list: list[list[dict]],
    num_agents: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack a list of C candidate emit-dicts-lists into stacked tensors.

    Each emit list is `[{"src_pid": int, "angle": float, "ships": int}, ...]`
    as produced by `apply_mechanisms_numpy`. Slots beyond the list length
    or beyond `MAX_LAUNCH_PER_AGENT` are filled with the -1 / 0 / 0
    sentinel triple (no launch).

    Returns `(pids, angles, ships)` of shapes `(C, MAX_LAUNCH_PER_AGENT)`
    each, dtypes (int32, float32, int32).
    """
    C = len(emit_list)
    pids = -np.ones((C, MAX_LAUNCH_PER_AGENT), dtype=np.int32)
    angles = np.zeros((C, MAX_LAUNCH_PER_AGENT), dtype=np.float32)
    ships = np.zeros((C, MAX_LAUNCH_PER_AGENT), dtype=np.int32)
    for c, emits in enumerate(emit_list):
        for k, mv in enumerate(emits[:MAX_LAUNCH_PER_AGENT]):
            pids[c, k] = int(mv["src_pid"])
            angles[c, k] = float(mv["angle"])
            ships[c, k] = int(mv["ships"])
    return pids, angles, ships


def _build_action_tensors_one(
    my_pids: jnp.ndarray, my_angles: jnp.ndarray, my_ships: jnp.ndarray,
    opp_pids: jnp.ndarray, opp_angles: jnp.ndarray, opp_ships: jnp.ndarray,
    my_id: int, opp_id: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pack one (my, opp) action pair into the full `(MAX_AGENTS,
    MAX_LAUNCH_PER_AGENT)` tensors that `jax_step` consumes. JIT-safe.
    """
    pids_full = jnp.full((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), -1, dtype=jnp.int32)
    ang_full = jnp.zeros((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=jnp.float32)
    sh_full = jnp.zeros((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=jnp.int32)
    pids_full = pids_full.at[my_id].set(my_pids)
    pids_full = pids_full.at[opp_id].set(opp_pids)
    ang_full = ang_full.at[my_id].set(my_angles)
    ang_full = ang_full.at[opp_id].set(opp_angles)
    sh_full = sh_full.at[my_id].set(my_ships)
    sh_full = sh_full.at[opp_id].set(opp_ships)
    return pids_full, ang_full, sh_full


def score_candidates_vmap(
    state,
    pids_c: jnp.ndarray,
    angles_c: jnp.ndarray,
    ships_c: jnp.ndarray,
    K: int,
    my_id: int,
    num_agents: int = 2,
    opp_aggressive: bool = True,
) -> jnp.ndarray:
    """Vmap-scored ship-delta for each of C candidate turn-1 actions.

    Inputs:
      state          — JAX game state (any `lib.game.jax.jax_types.GameState`).
      pids_c, angles_c, ships_c — shape `(C, MAX_LAUNCH_PER_AGENT)` each.
      K              — total rollout depth (turn 1 forced + K-1 follow-up).
      my_id          — our seat index (0 or 1 in 2P).
      num_agents     — must be 2 for vmap path (matches rollout_step_jax_pure).
      opp_aggressive — True ≡ Tier-1 v3.5.1 opp mirror (v7_0 default).

    Returns: jnp.ndarray of shape `(C,)`, dtype matches
    `value_delta_ships` return (int32, cast to float32 for argmax safety).

    Implementation detail: the opp's turn-1 action is computed once
    against the input state (it does not depend on our candidate) and
    is broadcast across all C rows. Saves ~C× the WorldModel + mission
    pipeline cost.
    """
    if num_agents != 2:
        raise ValueError(
            f"score_candidates_vmap is 2P-only (got num_agents={num_agents})"
        )
    opp_id = 1 - my_id

    # Compute opp's turn-1 action once. Reused across all C candidates.
    wm = build_world_model(state, max_horizon=DEFAULT_HORIZON, num_agents=4)
    opp_pids, opp_angles, opp_ships = policy_emit_jax_pure(
        state, wm, my_id=opp_id, aggressive=opp_aggressive,
        num_agents=num_agents,
    )

    def score_one(my_pids, my_angles, my_ships):
        pids_full, ang_full, sh_full = _build_action_tensors_one(
            my_pids, my_angles, my_ships,
            opp_pids, opp_angles, opp_ships,
            my_id=my_id, opp_id=opp_id,
        )
        s = jax_step(state, pids_full, ang_full, sh_full)
        # K-1 follow-up steps under mirror-mirror.
        def step_fn(s_, _):
            new_s = rollout_step_jax_pure(
                s_, my_id=my_id, num_agents=num_agents,
                opp_aggressive=opp_aggressive,
                my_aggressive=False,
            )
            return new_s, None
        final, _ = jax.lax.scan(step_fn, s, None, length=max(0, K - 1))
        return value_delta_ships(final, my_id=my_id).astype(jnp.float32)

    return jax.vmap(score_one, in_axes=(0, 0, 0))(pids_c, angles_c, ships_c)


score_candidates_vmap_jit = jax.jit(
    score_candidates_vmap,
    static_argnames=("K", "my_id", "num_agents", "opp_aggressive"),
)


def argmax_candidate_vmap(
    state,
    pids_c: jnp.ndarray,
    angles_c: jnp.ndarray,
    ships_c: jnp.ndarray,
    K: int,
    my_id: int,
    num_agents: int = 2,
    opp_aggressive: bool = True,
) -> Tuple[int, float]:
    """Return `(best_candidate_index, best_score)` from a vmap'd score.

    Convenience wrapper around `score_candidates_vmap` for callers that
    only need the winning index. Scalarises the result to a Python int
    + float so callers don't need to import jnp.
    """
    scores = score_candidates_vmap(
        state, pids_c, angles_c, ships_c,
        K=K, my_id=my_id, num_agents=num_agents,
        opp_aggressive=opp_aggressive,
    )
    best_idx = int(jnp.argmax(scores))
    best_score = float(scores[best_idx])
    return best_idx, best_score
