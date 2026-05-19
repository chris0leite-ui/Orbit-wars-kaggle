"""Iterative-deepening tree search over a cluster sub-game.

Solves "what's the best move for `my_id` in this isolated cluster?"
where the opponent is assumed to play `lite_greedy_policy`. The opp is
fixed (no minimax over opp choices) — we're solving a single-player
sequential decision problem in our half, with a known reactive opp.

This is intentionally not full game-theoretic minimax: we want the
audit to compare "what would happen if WE played optimally, against
the same opp model the heuristic projects against." Full
maximin-over-opp can come in a second iteration if needed.

Action generation per ply for our seat: IDLE plus the cheapest
single-source capture per (source, target) pair plus the cheapest
multi-source bundle per target. Candidates are produced by the
existing primitives in `agents.trajectory_roi.main` so the audit
compares the heuristic's chooser against the same enumerator.

Leaf evaluation: `_terminal_value(snap, my_id)` from trajectory_roi
— `delta_us_minus_them` plus the centrality bonus. Same value
function the heuristic uses, so audit discrepancies indicate
chooser/scorer bugs, not value-function disagreements.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from lib import fast_sim
from lib.opp_model import lite_greedy_policy

from agents.trajectory_roi.main import (
    _build_centrality_cache,
    _emit_for_candidate,
    _obs_from_snap,
    _solve_multi_source,
    _solve_single_source,
    _terminal_value,
    Candidate,
)
from lib.trajectory_layer import World


# ---- tunables -----------------------------------------------------------

DEFAULT_MAX_DEPTH = 6
DEFAULT_BUDGET_MS = 2000.0
TOP_K_CANDIDATES = 4    # per ply for our side (plus IDLE)


# ---- types --------------------------------------------------------------


@dataclass(frozen=True)
class SolveResult:
    best_action: list                 # emit list (possibly [])
    value: float
    depth_reached: int
    nodes_searched: int
    elapsed_ms: float


# ---- internal -----------------------------------------------------------


def _generate_my_candidates(world: World, my_id: int,
                            centrality_cache: dict[int, float]) -> list[Candidate]:
    """Top-K our-side action candidates: each candidate IS a launch (or
    list of launches for multi-source). Always plus IDLE outside this
    function."""
    out: list[Candidate] = []
    my_planets = [p for p in world.planets if p.owner == my_id]
    targets = [p for p in world.planets if p.owner != my_id]

    for src in my_planets:
        for tgt in targets:
            c = _solve_single_source(src, tgt, world, my_id,
                                     centrality_cache,
                                     target_is_ours=False)
            if c is not None:
                out.append(c)
        # multi-source over each target
    for tgt in targets:
        c = _solve_multi_source(tgt, world, my_id, centrality_cache,
                                 target_is_ours=False)
        if c is not None:
            out.append(c)
    # Defense: any of MY threatened planets — let _solve handle the
    # `target_is_ours=True` branch; it returns None unless the planet
    # is actually threatened (no incoming opp fleets → returns None).
    for tgt in my_planets:
        for src in my_planets:
            if src.id == tgt.id:
                continue
            c = _solve_single_source(src, tgt, world, my_id,
                                     centrality_cache,
                                     target_is_ours=True)
            if c is not None:
                out.append(c)

    # Top-K by ROI.
    out.sort(key=lambda c: -c.roi)
    seen: set[tuple] = set()
    uniq: list[Candidate] = []
    for c in out:
        sig = (c.target_id, tuple((a.src_id, a.ships) for a in c.allocations))
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(c)
        if len(uniq) >= TOP_K_CANDIDATES:
            break
    return uniq


def _max_search(snap: fast_sim.Snapshot, my_id: int, opp_id: int,
                 depth: int, deadline: float,
                 stats: dict[str, int]) -> tuple[list, float]:
    """Return (best_action, value) for the current snap.

    `best_action` is the emit list for our seat at this ply; subsequent
    plies aren't returned (we only need the immediate move at the root)."""
    stats["nodes"] += 1
    if depth <= 0 or snap.fake_env.done or time.perf_counter() >= deadline:
        return [], _terminal_value(snap, my_id)

    # Rebuild a World view on the current snap so candidate-generation
    # primitives have what they expect.
    obs = _obs_from_snap(snap, my_id)
    world = World.from_obs(obs)
    centrality_cache = _build_centrality_cache(world)

    candidates = _generate_my_candidates(world, my_id, centrality_cache)
    # Always include IDLE as a baseline.
    candidate_emits: list[list] = [[]]
    for c in candidates:
        candidate_emits.append(_emit_for_candidate(c))

    opp_emit = lite_greedy_policy(_obs_from_snap(snap, opp_id))

    best_value = -float("inf")
    best_action: list = []
    for my_emit in candidate_emits:
        if time.perf_counter() >= deadline:
            break
        actions: list[Any] = [None, None]
        actions[my_id] = my_emit
        actions[opp_id] = opp_emit
        child_snap = fast_sim.step(snap, actions)
        _, child_value = _max_search(child_snap, my_id, opp_id,
                                      depth - 1, deadline, stats)
        if child_value > best_value:
            best_value = child_value
            best_action = my_emit
    return best_action, best_value


# ---- public entry -------------------------------------------------------


def solve(isolated_obs: dict, my_id: int, opp_id: int,
          max_depth: int = DEFAULT_MAX_DEPTH,
          budget_ms: float = DEFAULT_BUDGET_MS) -> SolveResult:
    """Iterative-deepening search over the cluster sub-game.

    Calls `_max_search` at increasing depths until the budget is
    exhausted; returns the deepest-completed result.
    """
    t_start = time.perf_counter()
    deadline = t_start + budget_ms / 1000.0
    stats: dict[str, int] = {"nodes": 0}

    snap = fast_sim.from_obs(isolated_obs, configuration=None)
    best_action: list = []
    best_value = -float("inf")
    depth_reached = 0
    for d in range(1, max_depth + 1):
        if time.perf_counter() >= deadline:
            break
        # Recursive search; allow it to consume the remaining budget.
        action_at_d, value_at_d = _max_search(snap, my_id, opp_id,
                                                d, deadline, stats)
        # If we hit the deadline mid-recursion, the result for this depth
        # is partial; prefer keeping the deeper-completed result.
        if time.perf_counter() >= deadline:
            break
        best_action = action_at_d
        best_value = value_at_d
        depth_reached = d

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    return SolveResult(
        best_action=best_action,
        value=best_value if depth_reached > 0 else _terminal_value(snap, my_id),
        depth_reached=depth_reached,
        nodes_searched=stats["nodes"],
        elapsed_ms=elapsed_ms,
    )
