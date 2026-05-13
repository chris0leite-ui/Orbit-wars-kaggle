"""JAX port of `lib/v7_search.py::choose_depth2` — depth-2 maximin chooser.

What this is: a pure-JAX policy_emit + rollout step that, on a given
state, enumerates drop-one variants for both seats, evaluates the full
payoff matrix via nested vmap (K-2 mirror-mirror tail per cell), and
returns the maximin-best action.

Why we want it: the scalar `choose_depth2` runs at ~500 ms/turn under
the 700 ms wallclock budget — too slow to A/B test in a reasonable
sample size on local CPU. The JAX version runs the full N×M payoff
matrix in one vmap-fold, JIT-compiled once, and game-vmaps 64 games
at a time on a Kaggle T4. Expected wallclock: ~1-2 min/A-B.

Algorithm (matches the scalar `choose_depth2` algorithm with one
deliberate approximation — opp's drop-one set is enumerated from the
INITIAL state, not from the post-our-turn state, so the same M
candidates are scored against every our_i row; this preserves nested
vmap shape and is the price for ~30× speedup):

  1. Build incumbents for both seats from `state`.
  2. Generate drop-one variants for both seats:
     - Row 0 = incumbent.
     - Row k+1 = incumbent with slot k zeroed (sentinel pid=-1, ships=0).
     - Total N = M = MAX_LAUNCH_PER_AGENT + 1.
  3. For each (i, j) cell:
     a. Turn 1: our seat plays our_v[i]; opp plays its INITIAL
        incumbent (NOT opp_v[j] yet — matches scalar choose_depth2
        which evaluates opp's response on turn 2, not turn 1).
     b. Turn 2: our seat passes (no action); opp plays opp_v[j].
     c. Turns 3..K: both seats play mirror-mirror.
     d. Return ship-delta at terminal.
  4. Maximin: argmax_i min_j payoff[i][j]. Ties → row 0 (incumbent).

Public surface:
- `jax_drop_one_variants(pids, angles, ships)`:
    a single agent's `(MAX_LAUNCH,)` triple → `(MAX_LAUNCH+1, MAX_LAUNCH)`
    triple of drop-one variants. Pure JAX, jit/vmap safe.
- `score_depth2_payoff_matrix(state, our_v, opp_v, opp_inc, K_tail,
   my_id, ...)`:
    nested vmap, returns `(MAX_LAUNCH+1, MAX_LAUNCH+1)` payoff matrix.
- `policy_emit_depth2_jax_pure(state, my_id, K_tail, ...)`:
    full chooser. Returns `(pids, angles, ships)` for the winning
    candidate — drop-in replacement for `policy_emit_jax_pure` at sites
    that want depth-2 lookahead.
- `rollout_step_depth2_jax_pure(state, my_id, K_tail, ...)`:
    single env tick where the `my_id` seat uses depth-2 and the other
    seat plays its single-ply mirror. JIT-friendly.

Parity gate: `tests/test_jax_depth2.py` covers shape / determinism /
step-advance invariants. A bit-exact parity test against scalar
`choose_depth2` would require matching cap sizes (scalar caps our_C/opp_C
at 8/4; JAX uses all MAX_LAUNCH+1) and matching K_tail; not implemented.

**Known scale issue (2026-05-13):** the nested vmap × 64-game vmap'd
rollout OOMs on a Kaggle T4 GPU (16 GB) during JIT compile — a single
intermediate tensor wants to allocate ~16 GB. The cell count per step
is 64 games × (MAX_LAUNCH+1)² ≈ 28 000 parallel rollouts, which is too
much memory bandwidth at K_tail≥4. Two paths exist:
  (a) Replace the inner `jax.vmap(jax.vmap(cell))` with a `jax.lax.scan`
      over (i, j) indices — sequentialises cells within a game,
      preserves the outer game-vmap. Memory drops to O(1 cell-state)
      per game; runtime grows ~N×M× but stays vmap'd over games.
  (b) Truncate our_C / opp_C to small fixed caps (e.g., 4 × 2) by
      sorting valid slots first and dropping the rest. Requires a
      pure-JAX top-K selector and breaks parity with the scalar
      "drop-every-slot" semantics.
Path (a) is the cleaner fix; deferred to next session.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from lib.game.jax.jax_interpreter import jax_step
from lib.game.jax.jax_score import (
    policy_emit_jax_pure,
    rollout_step_jax_pure,
    value_delta_ships,
    _build_planet_orbits_jax,
)
from lib.game.jax.jax_world_model import build_world_model, DEFAULT_HORIZON
from lib.game.jax.jax_types import MAX_AGENTS, MAX_LAUNCH_PER_AGENT


def jax_drop_one_variants(
    pids: jnp.ndarray, angles: jnp.ndarray, ships: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build the (incumbent + drop-each-launch) variant set as stacked
    tensors.

    Input shape: `(MAX_LAUNCH_PER_AGENT,)` each.
    Output shape: `(MAX_LAUNCH_PER_AGENT + 1, MAX_LAUNCH_PER_AGENT)` each.

    Row 0 is the incumbent. Row k+1 has slot k zeroed (pid=-1, angle=0,
    ships=0). Variants that "drop" already-sentinel slots are identical
    to the incumbent — harmless duplicates that simply get re-scored.

    Pure JAX: jit/vmap safe; no Python control flow on traced values.
    """
    L = MAX_LAUNCH_PER_AGENT
    incumbent_mask = jnp.ones((1, L), dtype=jnp.bool_)
    drop_masks = ~jnp.eye(L, dtype=jnp.bool_)  # (L, L)
    full_mask = jnp.concatenate([incumbent_mask, drop_masks], axis=0)  # (L+1, L)

    pids_b = jnp.broadcast_to(pids, (L + 1, L))
    angles_b = jnp.broadcast_to(angles, (L + 1, L))
    ships_b = jnp.broadcast_to(ships, (L + 1, L))

    pids_v = jnp.where(full_mask, pids_b, jnp.int32(-1))
    angles_v = jnp.where(full_mask, angles_b, jnp.float32(0.0))
    ships_v = jnp.where(full_mask, ships_b, jnp.int32(0))
    return pids_v, angles_v, ships_v


def _pack_two_seat_action(
    my_pids: jnp.ndarray, my_angles: jnp.ndarray, my_ships: jnp.ndarray,
    opp_pids: jnp.ndarray, opp_angles: jnp.ndarray, opp_ships: jnp.ndarray,
    my_id: int, opp_id: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pack one (my, opp) action pair into full `(MAX_AGENTS,
    MAX_LAUNCH_PER_AGENT)` tensors for `jax_step`. JIT-safe.
    """
    pids_full = jnp.full(
        (MAX_AGENTS, MAX_LAUNCH_PER_AGENT), -1, dtype=jnp.int32,
    )
    ang_full = jnp.zeros(
        (MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=jnp.float32,
    )
    sh_full = jnp.zeros(
        (MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=jnp.int32,
    )
    pids_full = pids_full.at[my_id].set(my_pids).at[opp_id].set(opp_pids)
    ang_full = ang_full.at[my_id].set(my_angles).at[opp_id].set(opp_angles)
    sh_full = sh_full.at[my_id].set(my_ships).at[opp_id].set(opp_ships)
    return pids_full, ang_full, sh_full


def score_depth2_payoff_matrix(
    state,
    our_v_pids: jnp.ndarray, our_v_angles: jnp.ndarray, our_v_ships: jnp.ndarray,
    opp_v_pids: jnp.ndarray, opp_v_angles: jnp.ndarray, opp_v_ships: jnp.ndarray,
    opp_inc_pids: jnp.ndarray, opp_inc_angles: jnp.ndarray, opp_inc_ships: jnp.ndarray,
    K_tail: int,
    my_id: int,
    num_agents: int = 2,
    opp_aggressive: bool = True,
    my_aggressive: bool = True,
    my_use_opening: bool = True,
    opp_use_opening: bool = True,
) -> jnp.ndarray:
    """Compute the (N, M) payoff matrix via nested vmap over our × opp
    drop-one variants.

    Cell (i, j):
      t=1: our seat plays our_v[i]; opp plays opp_inc (turn-1 incumbent).
      t=2: our seat passes; opp plays opp_v[j].
      t=3..K_tail+2: both seats play single-ply mirror.
      leaf: ship-delta from `my_id`'s POV.
    """
    opp_id = (my_id + 1) % num_agents
    L1 = MAX_LAUNCH_PER_AGENT

    def cell(
        u_p: jnp.ndarray, u_a: jnp.ndarray, u_s: jnp.ndarray,
        o_p: jnp.ndarray, o_a: jnp.ndarray, o_s: jnp.ndarray,
    ) -> jnp.ndarray:
        # Turn 1: forced our_i + opp's turn-1 incumbent.
        p1, a1, s1 = _pack_two_seat_action(
            u_p, u_a, u_s,
            opp_inc_pids, opp_inc_angles, opp_inc_ships,
            my_id=my_id, opp_id=opp_id,
        )
        st1 = jax_step(state, p1, a1, s1)

        # Turn 2: us pass (no action), opp plays forced opp_v[j].
        zero_p = jnp.full((L1,), -1, dtype=jnp.int32)
        zero_a = jnp.zeros((L1,), dtype=jnp.float32)
        zero_s = jnp.zeros((L1,), dtype=jnp.int32)
        p2, a2, s2 = _pack_two_seat_action(
            zero_p, zero_a, zero_s,
            o_p, o_a, o_s,
            my_id=my_id, opp_id=opp_id,
        )
        st2 = jax_step(st1, p2, a2, s2)

        # K_tail mirror-mirror tail rollout.
        def step_fn(s_, _):
            new_s = rollout_step_jax_pure(
                s_, my_id=my_id, num_agents=num_agents,
                opp_aggressive=opp_aggressive,
                my_aggressive=my_aggressive,
                my_use_opening=my_use_opening,
                opp_use_opening=opp_use_opening,
            )
            return new_s, None

        final, _ = jax.lax.scan(step_fn, st2, None, length=K_tail)
        return value_delta_ships(final, my_id=my_id).astype(jnp.float32)

    # Inner vmap: vary opp axis (axis 0 of opp_v_*); fix our.
    inner = jax.vmap(cell, in_axes=(None, None, None, 0, 0, 0))
    # Outer vmap: vary our axis; broadcast opp inputs across.
    outer = jax.vmap(inner, in_axes=(0, 0, 0, None, None, None))
    payoff = outer(
        our_v_pids, our_v_angles, our_v_ships,
        opp_v_pids, opp_v_angles, opp_v_ships,
    )
    return payoff  # shape (L+1, L+1)


def policy_emit_depth2_jax_pure(
    state,
    my_id: int,
    K_tail: int = 4,
    num_agents: int = 2,
    opp_aggressive: bool = True,
    my_aggressive: bool = True,
    my_use_opening: bool = True,
    opp_use_opening: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Depth-2 maximin chooser as a pure-JAX policy_emit replacement.

    Builds both seats' incumbents, generates drop-one variant sets,
    scores the full payoff matrix via `score_depth2_payoff_matrix`,
    and returns the maximin-best action for `my_id`. Argmax tie-break:
    `jnp.argmax` returns the FIRST max — row 0 (incumbent) wins ties.
    """
    opp_id = (my_id + 1) % num_agents

    # Shared WorldModel + planet orbits for both incumbents (cheap reuse).
    wm = build_world_model(state, max_horizon=DEFAULT_HORIZON, num_agents=4)
    planet_orbits = _build_planet_orbits_jax(state)

    our_inc_p, our_inc_a, our_inc_s = policy_emit_jax_pure(
        state, wm, my_id=my_id, aggressive=my_aggressive,
        num_agents=num_agents, planet_orbits=planet_orbits,
        use_opening=my_use_opening,
    )
    opp_inc_p, opp_inc_a, opp_inc_s = policy_emit_jax_pure(
        state, wm, my_id=opp_id, aggressive=opp_aggressive,
        num_agents=num_agents, planet_orbits=planet_orbits,
        use_opening=opp_use_opening,
    )

    our_v_p, our_v_a, our_v_s = jax_drop_one_variants(
        our_inc_p, our_inc_a, our_inc_s,
    )
    opp_v_p, opp_v_a, opp_v_s = jax_drop_one_variants(
        opp_inc_p, opp_inc_a, opp_inc_s,
    )

    payoff = score_depth2_payoff_matrix(
        state,
        our_v_p, our_v_a, our_v_s,
        opp_v_p, opp_v_a, opp_v_s,
        opp_inc_p, opp_inc_a, opp_inc_s,
        K_tail=K_tail, my_id=my_id, num_agents=num_agents,
        opp_aggressive=opp_aggressive, my_aggressive=my_aggressive,
        my_use_opening=my_use_opening, opp_use_opening=opp_use_opening,
    )

    # Maximin: argmax over min-per-row. `jnp.argmax` ties → first match,
    # which is row 0 (incumbent) — preserves the scalar choose_depth2
    # parity-floor tie-break.
    worst_per_row = jnp.min(payoff, axis=1)  # (L+1,)
    best_i = jnp.argmax(worst_per_row)

    return our_v_p[best_i], our_v_a[best_i], our_v_s[best_i]


def rollout_step_depth2_jax_pure(
    state,
    my_id: int,
    K_tail: int = 4,
    num_agents: int = 2,
    opp_aggressive: bool = True,
    my_aggressive: bool = True,
    my_use_opening: bool = True,
    opp_use_opening: bool = True,
):
    """Single env tick: `my_id` plays depth-2; opp plays single-ply mirror.

    Drop-in replacement for `rollout_step_jax_pure` at game-vmap sites
    where one seat should use the depth-2 chooser. Both seats' policy
    parameters (`aggressive`, `use_opening`) are exposed so the kernel
    A/B harness can independently configure each.
    """
    assert num_agents == 2, (
        "rollout_step_depth2_jax_pure currently supports only 2P games."
    )
    opp_id = 1 - my_id

    # Our seat: depth-2 maximin.
    my_pids, my_ang, my_sh = policy_emit_depth2_jax_pure(
        state, my_id=my_id, K_tail=K_tail, num_agents=num_agents,
        opp_aggressive=opp_aggressive, my_aggressive=my_aggressive,
        my_use_opening=my_use_opening, opp_use_opening=opp_use_opening,
    )

    # Opp seat: single-ply mirror (matches v7_0 / v7_1 behaviour).
    wm = build_world_model(state, max_horizon=DEFAULT_HORIZON, num_agents=4)
    opp_pids, opp_ang, opp_sh = policy_emit_jax_pure(
        state, wm, my_id=opp_id, aggressive=opp_aggressive,
        num_agents=num_agents, use_opening=opp_use_opening,
    )

    p, a, s = _pack_two_seat_action(
        my_pids, my_ang, my_sh,
        opp_pids, opp_ang, opp_sh,
        my_id=my_id, opp_id=opp_id,
    )
    return jax_step(state, p, a, s)


# JIT-compile entry points.
policy_emit_depth2_jit = jax.jit(
    policy_emit_depth2_jax_pure,
    static_argnames=(
        "my_id", "K_tail", "num_agents",
        "opp_aggressive", "my_aggressive",
        "my_use_opening", "opp_use_opening",
    ),
)
rollout_step_depth2_jit = jax.jit(
    rollout_step_depth2_jax_pure,
    static_argnames=(
        "my_id", "K_tail", "num_agents",
        "opp_aggressive", "my_aggressive",
        "my_use_opening", "opp_use_opening",
    ),
)
