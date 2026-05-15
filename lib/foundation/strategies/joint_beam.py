"""Phase C joint two-tier beam search.

Pipeline:

1. **Tier-1 beam** (cheap shortlist): reuse Phase A's
   `score_candidates_vmap_value_prod_jit` (single-step scorer; opp
   plays no-op). Drives a `(W, D)` beam over multi-turn atoms with
   the per-`(source, launch_turn)` compatibility filter. At the end,
   collect the top-W' candidates by Tier-1 score.

2. **Tier-2 min-regret**: re-score the top-W' candidates with the
   H-step joint rollout `score_candidates_multi_turn_rollout_jit`,
   once per opp archetype. Aggregate per-candidate via min (maximin /
   min-regret) across archetypes. Pick argmax.

If `H == 1` the Tier-2 rollout reduces to a single `jax_step` per
candidate and the design degenerates gracefully (still useful — adds
opp-archetype min-regret to Phase A's no-opp scoring).

Compatibility filter is keyed on `(from_planet_id, launch_turn)` so a
source can fire on turn 0 AND turn 1 (the multi-wave-from-one-source
case Phase A's filter blocked).
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
from lib.foundation.strategies.analytic_score_rollout import (
    score_candidates_multi_turn_rollout_jit,
)
from lib.game.jax.jax_types import GameState, MAX_LAUNCH_PER_AGENT


FIXED_CANDIDATE_BATCH = 128
TIER2_BATCH = 16  # top_w_tier2 padded


def joint_beam_search(
    state: GameState,
    atomic_launches: list[ActionSpec],
    my_id: int,
    archetype_panel: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    *,
    width: int = 4,
    depth: int = 4,
    H: int = 2,
    K: int = 5,
    top_w_tier2: int = 8,
    num_agents: int = 2,
    opp_aggressive: bool = True,
    budget_ms: float = 800.0,
    pre_committed: Sequence[ActionSpec] = (),
) -> list[ActionSpec]:
    """Run Phase C joint two-tier beam search; return the winning
    multi-turn action set.

    Args:
        atomic_launches: multi-turn atoms (from
            `enumerate_multi_turn_atoms`); each may have `launch_turn`
            in `range(H)`.
        archetype_panel: output of
            `build_opp_archetype_panel(...)` — list of
            `(name, opp_pids_h, opp_angles_h, opp_ships_h)`.
        H: rollout horizon (matches the panel's H).
        top_w_tier2: Tier-1 shortlist size handed to Tier-2.
    """
    if num_agents != 2:
        raise ValueError(
            "joint_beam_search currently requires 2P "
            "(score_candidates_vmap_value_prod is 2P-only)."
        )
    seeded = list(pre_committed)
    if not atomic_launches and not seeded:
        return []
    if not archetype_panel:
        # Fall back to Phase A behaviour if no archetypes available.
        archetype_panel = []

    t_start = time.perf_counter()

    def _tier1_score(candidate_sets: list[list[ActionSpec]]) -> np.ndarray:
        """Tier-1 scorer (Phase A K=0 kernel, opp no-op)."""
        n = len(candidate_sets)
        out = np.empty(n, dtype=np.float32)
        cursor = 0
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

    baseline_score = float(_tier1_score([seeded])[0])

    best_set: list[ActionSpec] = list(seeded)
    best_score: float = baseline_score
    beam: list[list[ActionSpec]] = [list(seeded)]
    all_visited: list[tuple[float, list[ActionSpec]]] = [(baseline_score, list(seeded))]

    for d in range(depth):
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        remaining_ms = budget_ms - elapsed_ms
        if remaining_ms < 100.0:
            break

        adapt_width = _adaptive_width(width, remaining_ms)

        new_candidates: list[tuple[float, list[ActionSpec]]] = []
        for current_set in beam:
            valid_atoms = _filter_compatible_multi_turn(current_set, atomic_launches)
            if not valid_atoms:
                continue
            candidate_sets = [current_set + [atom] for atom in valid_atoms]
            scores_np = _tier1_score(candidate_sets)
            for atom_idx, _atom in enumerate(valid_atoms):
                new_candidates.append(
                    (float(scores_np[atom_idx]), candidate_sets[atom_idx])
                )

        if not new_candidates:
            break

        new_candidates.sort(key=lambda x: -x[0])
        top = new_candidates[:adapt_width]
        for score, action_set in top:
            if score > best_score:
                best_score = score
                best_set = action_set
        all_visited.extend(new_candidates)
        beam = [s for _, s in top]

    # Tier-2: re-score the top-W' candidates with min-regret across
    # archetype panel. Always include `best_set` (the Tier-1 winner) and
    # the top-W' by Tier-1 score (without duplicates).
    if not archetype_panel:
        return best_set

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    if budget_ms - elapsed_ms < 80.0:
        # Out of budget for Tier-2; ship Tier-1 winner.
        return best_set

    all_visited.sort(key=lambda x: -x[0])
    shortlist_uniq: list[list[ActionSpec]] = []
    seen_keys: set[tuple] = set()
    for _, action_set in all_visited:
        key = tuple(
            (s.from_planet_id, s.launch_turn, round(float(s.dir_angle), 6), int(s.ships))
            for s in action_set
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        shortlist_uniq.append(action_set)
        if len(shortlist_uniq) >= top_w_tier2:
            break

    if not shortlist_uniq:
        return best_set

    tier2_scores_by_archetype = _tier2_score_all_archetypes(
        state, my_id, num_agents,
        shortlist_uniq, archetype_panel, H,
    )
    # Min over archetypes (maximin / min-regret over raw scores).
    aggregated = np.min(tier2_scores_by_archetype, axis=0)  # (W',)
    pick = int(np.argmax(aggregated))
    return shortlist_uniq[pick]


def _tier2_score_all_archetypes(
    state: GameState,
    my_id: int,
    num_agents: int,
    shortlist: list[list[ActionSpec]],
    archetype_panel: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    H: int,
) -> np.ndarray:
    """Run Tier-2 H-step rollout scorer for each archetype and stack.

    Returns shape `(n_archetypes, W')` float32.
    """
    W = len(shortlist)
    # Pad to TIER2_BATCH for JIT stability.
    while len(shortlist) < TIER2_BATCH:
        shortlist = shortlist + [[]]
    my_pids_ch, my_angles_ch, my_ships_ch = _action_specs_to_multi_turn_arrays(
        shortlist[:TIER2_BATCH], H,
    )
    arch_scores: list[np.ndarray] = []
    for _name, opp_pids_h, opp_angles_h, opp_ships_h in archetype_panel:
        scores = score_candidates_multi_turn_rollout_jit(
            state,
            jnp.asarray(my_pids_ch),
            jnp.asarray(my_angles_ch),
            jnp.asarray(my_ships_ch),
            jnp.asarray(opp_pids_h),
            jnp.asarray(opp_angles_h),
            jnp.asarray(opp_ships_h),
            H=H, my_id=my_id, num_agents=num_agents,
        )
        arch_scores.append(np.asarray(scores)[:W])
    return np.stack(arch_scores, axis=0)


def _action_specs_to_multi_turn_arrays(
    candidates: list[list[ActionSpec]],
    H: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack a list of multi-turn action sets into per-turn arrays.

    Each input set may contain ActionSpecs with `launch_turn in
    range(H)`. Output shape `(C, H, MAX_LAUNCH_PER_AGENT)` for each
    of pids/angles/ships.
    """
    C = len(candidates)
    pids = -np.ones((C, H, MAX_LAUNCH_PER_AGENT), dtype=np.int32)
    angles = np.zeros((C, H, MAX_LAUNCH_PER_AGENT), dtype=np.float32)
    ships = np.zeros((C, H, MAX_LAUNCH_PER_AGENT), dtype=np.int32)
    for c, specs in enumerate(candidates):
        # Auto-allocate slot per (turn) — preserve input order.
        next_slot: dict[int, int] = {}
        for spec in specs:
            t = int(spec.launch_turn)
            if t < 0 or t >= H:
                continue
            slot = next_slot.get(t, 0)
            if slot >= MAX_LAUNCH_PER_AGENT:
                continue
            pids[c, t, slot] = int(spec.from_planet_id)
            angles[c, t, slot] = float(spec.dir_angle)
            ships[c, t, slot] = int(spec.ships)
            next_slot[t] = slot + 1
    return pids, angles, ships


def _adaptive_width(width: int, remaining_ms: float) -> int:
    if remaining_ms < 200.0:
        return max(2, width // 4)
    if remaining_ms < 400.0:
        return max(3, width // 2)
    return width


def _filter_compatible_multi_turn(
    current_set: list[ActionSpec],
    atomic_launches: list[ActionSpec],
) -> list[ActionSpec]:
    """Phase C compatibility: one launch per `(source, launch_turn)`.

    Allows the same source to fire on different turns within a single
    multi-turn plan (the 2-wave-from-one-source case Phase A blocked).
    """
    used: set[tuple[int, int]] = {
        (spec.from_planet_id, spec.launch_turn) for spec in current_set
    }
    return [
        a for a in atomic_launches
        if (a.from_planet_id, a.launch_turn) not in used
    ]
