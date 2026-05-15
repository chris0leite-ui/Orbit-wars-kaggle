"""Phase C scoring kernel — multi-turn `jax.lax.scan` rollout under
splicable opp action at turn 0.

Wraps `jax_step` in a `lax.scan` of length `H`, splicing per-turn
actions for both our seat (`my_id`) and the opp seat (`opp_id = 1 - my_id`,
2P-only). Per-candidate cost grows roughly linearly in `H`; warm cost
at C=128, H=2 is expected ~50-80 ms (measured empirically; cf.
Phase A's K=0 baseline ~30 ms warm).

Differs from Phase A's `score_candidates_vmap_value_prod` in three ways:
1. Multi-turn: applies our action over `H` turns (Phase A was 1 turn).
2. Splicable opp: opp plays the SUPPLIED action at turn 0 and no-ops
   for `t >= 1` (the archetype panel injects different opp actions
   here per call).
3. Same JAX-pure value head at the end (`value_with_future_production`).

The 2P-only restriction matches Phase A (`opp_id = 1 - my_id`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from lib.foundation.strategies.analytic_score import value_with_future_production
from lib.game.jax.jax_interpreter import jax_step
from lib.game.jax.jax_types import (
    GameState,
    MAX_AGENTS,
    MAX_LAUNCH_PER_AGENT,
)


def score_candidates_multi_turn_rollout(
    state: GameState,
    my_pids_ch: jnp.ndarray,      # (C, H, MAX_LAUNCH_PER_AGENT)
    my_angles_ch: jnp.ndarray,    # (C, H, MAX_LAUNCH_PER_AGENT)
    my_ships_ch: jnp.ndarray,     # (C, H, MAX_LAUNCH_PER_AGENT)
    opp_pids_h: jnp.ndarray,      # (H, MAX_LAUNCH_PER_AGENT) — single archetype
    opp_angles_h: jnp.ndarray,    # (H, MAX_LAUNCH_PER_AGENT)
    opp_ships_h: jnp.ndarray,     # (H, MAX_LAUNCH_PER_AGENT)
    H: int,
    my_id: int,
    num_agents: int = 2,
) -> jnp.ndarray:
    """Score C candidates over H-turn rollout against a fixed opp action.

    Returns shape `(C,)` float32 — value head at the rolled-out state.

    Inputs already broken out per-turn so JAX scan slicing is direct.
    Opp action broadcasts over candidates: every candidate is scored
    against the SAME opp archetype-action-by-turn sequence supplied
    here. The min-regret aggregator calls this function once per
    archetype.
    """
    if num_agents != 2:
        raise ValueError(
            f"score_candidates_multi_turn_rollout is 2P-only "
            f"(got num_agents={num_agents})."
        )
    opp_id = 1 - my_id

    def score_one(my_pids_h, my_angles_h, my_ships_h):
        # my_*_h shape: (H, MAX_LAUNCH_PER_AGENT)
        def step_fn(carry_state, t_idx):
            pids_full = jnp.full(
                (MAX_AGENTS, MAX_LAUNCH_PER_AGENT), -1, dtype=jnp.int32,
            )
            ang_full = jnp.zeros(
                (MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=jnp.float32,
            )
            sh_full = jnp.zeros(
                (MAX_AGENTS, MAX_LAUNCH_PER_AGENT), dtype=jnp.int32,
            )
            pids_full = pids_full.at[my_id].set(my_pids_h[t_idx])
            ang_full = ang_full.at[my_id].set(my_angles_h[t_idx])
            sh_full = sh_full.at[my_id].set(my_ships_h[t_idx])
            pids_full = pids_full.at[opp_id].set(opp_pids_h[t_idx])
            ang_full = ang_full.at[opp_id].set(opp_angles_h[t_idx])
            sh_full = sh_full.at[opp_id].set(opp_ships_h[t_idx])
            new_state = jax_step(carry_state, pids_full, ang_full, sh_full)
            return new_state, None

        final_state, _ = jax.lax.scan(
            step_fn, state, jnp.arange(H), length=H,
        )
        return value_with_future_production(final_state, my_id=my_id)

    return jax.vmap(score_one, in_axes=(0, 0, 0))(
        my_pids_ch, my_angles_ch, my_ships_ch,
    )


score_candidates_multi_turn_rollout_jit = jax.jit(
    score_candidates_multi_turn_rollout,
    static_argnames=("H", "my_id", "num_agents"),
)
