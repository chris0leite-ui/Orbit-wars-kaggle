"""Beam search over candidate action sets, vmap'd per beam level.

Approach:
- Beam carries `width` partial action sets at each depth.
- At each level, expand every beam node by every compatible atomic
  launch; vmap-score the expansion in one JIT'd call.
- Keep the top-`width` of the expanded set by score.
- Iterate up to `depth` levels OR until no positive-score extension
  exists.
- Adaptive width shrinks under budget pressure (1000 ms turn cap minus
  the cold-JIT amortised cost).

**JIT shape stability.** Every call into
`score_candidates_vmap_value_prod_jit` uses a fixed batch dimension
`FIXED_CANDIDATE_BATCH` and pads the candidate list with
empty-action-set sentinels. Without this, every beam level
recompiled the kernel (variable C → variable trace → ~50 s cold-turn
cost). The padded slots' scores are computed but discarded.

Compatibility filter (Phase A): one launch per source planet per turn.
A source already used by an action in the current set is removed from
the pool when expanding. This caps per-turn launches at
`min(depth, num_owned_planets)` and keeps the beam from picking the
same source twice with different fractions. Phase B can relax to
"track per-source remaining ships."

Cost analysis (target budget 1500 ms cold / 800 ms warm):
- N_atoms typical: 100-400 per turn (post-eta filter).
- One vmap'd batched score call at C=FIXED: ~80-200 ms warm CPU.
- `depth × width` batched calls = 4 × 4 = 16 calls.
- Worst case: 16 × 200 ms = 3200 ms; adaptive width drops if needed.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import jax.numpy as jnp
import numpy as np

from lib.foundation.actions import ActionSpec
from lib.foundation.strategies.analytic_score import (
    action_specs_to_candidate_arrays,
    score_candidates_vmap_value_prod_jit,
)
from lib.game.jax.jax_types import GameState


# Fixed batch dimension for all vmap'd scoring calls. Larger = fewer
# calls per beam expansion but more JIT memory + more padding waste.
# 128 covers ~95% of typical per-node expansion sizes on a 24-planet
# board without needing to chunk.
FIXED_CANDIDATE_BATCH = 128


def beam_search(
    state: GameState,
    atomic_launches: list[ActionSpec],
    my_id: int,
    *,
    width: int = 4,
    depth: int = 4,
    K: int = 5,
    num_agents: int = 2,
    opp_aggressive: bool = True,
    budget_ms: float = 800.0,
    pre_committed: Sequence[ActionSpec] = (),
) -> list[ActionSpec]:
    """Beam search for the best action set; returns the winning set.

    `width` — partial sets carried per level (adaptive shrink).
    `depth` — max launches per action set.
    `K` — rollout depth used inside the scorer (defaults to 5).
    `budget_ms` — wall-clock cap. Beam exits early on budget.
    `pre_committed` — launches already locked in (e.g., from
        `MissionMemory.waves_to_fire`); the beam SEEDS from this set
        rather than the empty set and only extends from non-committed
        sources (the existing `_filter_compatible` enforces
        one-launch-per-source). The returned set is guaranteed to
        contain `pre_committed` as a prefix; `winning_set[len(
        pre_committed):]` are the beam-added launches.
    """
    if num_agents != 2:
        raise ValueError(
            "beam_search currently requires 2P "
            "(score_candidates_vmap_value_prod is 2P-only)."
        )
    seeded = list(pre_committed)
    if not atomic_launches and not seeded:
        return []

    t_start = time.perf_counter()

    def _score_padded(candidate_sets: list[list[ActionSpec]]) -> np.ndarray:
        """Score `len(candidate_sets)` action sets via the JIT'd
        vmap kernel at fixed batch size. Pads or chunks as needed."""
        out = np.empty(len(candidate_sets), dtype=np.float32)
        cursor = 0
        n = len(candidate_sets)
        while cursor < n:
            chunk = candidate_sets[cursor:cursor + FIXED_CANDIDATE_BATCH]
            real_n = len(chunk)
            padded = chunk + [[]] * (FIXED_CANDIDATE_BATCH - real_n)
            pids, angles, ships = action_specs_to_candidate_arrays(padded)
            scores = score_candidates_vmap_value_prod_jit(
                state,
                jnp.asarray(pids),
                jnp.asarray(angles),
                jnp.asarray(ships),
                K=K, my_id=my_id, num_agents=num_agents,
                opp_aggressive=opp_aggressive,
            )
            out[cursor:cursor + real_n] = np.asarray(scores)[:real_n]
            cursor += real_n
        return out

    # Score the seeded (pre-committed) action set as baseline. When
    # `pre_committed=()` this reduces to the empty set — the original
    # Phase A behaviour. Uses the same FIXED batch shape as every
    # subsequent call.
    baseline_score = float(_score_padded([seeded])[0])

    best_set: list[ActionSpec] = list(seeded)
    best_score: float = baseline_score
    beam: list[list[ActionSpec]] = [list(seeded)]

    for d in range(depth):
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        remaining_ms = budget_ms - elapsed_ms
        if remaining_ms < 50.0:
            break

        adapt_width = _adaptive_width(width, remaining_ms)

        new_candidates: list[tuple[float, list[ActionSpec]]] = []
        for current_set in beam:
            valid_atoms = _filter_compatible(current_set, atomic_launches)
            if not valid_atoms:
                continue

            candidate_sets = [current_set + [atom] for atom in valid_atoms]
            scores_np = _score_padded(candidate_sets)

            for atom_idx, atom in enumerate(valid_atoms):
                new_candidates.append(
                    (float(scores_np[atom_idx]), current_set + [atom])
                )

        if not new_candidates:
            break

        # Keep top-adapt_width.
        new_candidates.sort(key=lambda x: -x[0])
        top = new_candidates[:adapt_width]

        # Update global best.
        for score, action_set in top:
            if score > best_score:
                best_score = score
                best_set = action_set

        # Next level's beam.
        beam = [s for _, s in top]

    return best_set


def _adaptive_width(width: int, remaining_ms: float) -> int:
    """Shrink beam width under budget pressure."""
    if remaining_ms < 150.0:
        return max(2, width // 4)
    if remaining_ms < 350.0:
        return max(4, width // 2)
    return width


def _filter_compatible(
    current_set: list[ActionSpec],
    atomic_launches: list[ActionSpec],
) -> list[ActionSpec]:
    """Phase A compatibility: one launch per source per turn.

    Drops atoms whose `from_planet_id` is already used in
    `current_set`. Simplifies the per-turn ship-budget tracking that
    Phase B addresses.
    """
    used_sources = {spec.from_planet_id for spec in current_set}
    return [a for a in atomic_launches if a.from_planet_id not in used_sources]
