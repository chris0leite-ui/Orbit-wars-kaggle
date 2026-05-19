"""Backwards-from-goal acquisition sequencer (P3).

Given a portfolio of target planets to acquire (from P2), produce a list
of scheduled launches that acquires them in priority order, respecting
per-source ship budgets across targets.

Output is `ScheduledLaunch` records with `turn_offset >= 0`; the agent
(P0) only emits the turn_offset==0 ones each turn and re-plans next turn.

Reuses `_solve_single_source` and `_solve_multi_source` from
trajectory_roi for the sizing math (binary search on defenders), via a
per-source budget-override pattern (dataclasses.replace on the
PlanetView's `ships` field).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from agents.trajectory_roi.main import (
    Candidate, _build_centrality_cache, _solve_multi_source, _solve_single_source,
)
from lib.trajectory_layer import World


MAX_WAIT_TURNS = 30
MIN_LAUNCH_SHIPS = 5  # env minimum; matches trajectory_roi's constant


@dataclass(frozen=True)
class ScheduledLaunch:
    turn_offset: int       # 0 == this turn; >0 == future (re-planned each turn)
    src_id: int
    target_id: int
    ships: int
    aim_angle: float


def _src_with_ships(src, ships: int):
    """Return a PlanetView copy with `ships` overridden. Used to ask
    `_solve_single_source` "what would the launch look like if this
    source had N ships?" (Frozen dataclass → dataclasses.replace)."""
    return dataclasses.replace(src, ships=float(ships))


def _available_ships_at(src, src_id, wait: int,
                          reservations: dict[int, list[tuple[int, int]]]) -> int:
    """Ships available at source `src_id` at relative turn `wait`,
    accounting for prior reservations that consume ships at turns <= wait."""
    base = int(src.ships) + wait * int(src.production)
    consumed = sum(s for w, s in reservations.get(src_id, []) if w <= wait)
    return base - consumed


def _try_single_source_with_wait(src, target, world: World, my_id: int,
                                    centrality_cache, target_is_ours: bool,
                                    reservations: dict[int, list[tuple[int, int]]]
                                    ) -> tuple[int, Candidate] | None:
    """Search wait_offset in [0, MAX_WAIT_TURNS] for the cheapest launch
    from `src` to `target`, respecting prior reservations on `src`.

    Per-turn budget model: at wait W, source has
    `current_ships + W*production - sum(reservations at wait <= W)` ships
    available. Target defenders accrue at `tgt.ships + W*tgt.production`.

    Returns (wait_offset, Candidate) or None if no feasible W exists.
    ETA computed from current positions (orbital drift over W approximated
    as zero — matches trajectory_roi's primitive)."""
    src_id = src.id
    tgt_prod = int(target.production)
    best: tuple[int, Candidate] | None = None
    for wait in range(0, MAX_WAIT_TURNS + 1):
        avail = _available_ships_at(src, src_id, wait, reservations)
        if avail < MIN_LAUNCH_SHIPS:
            continue
        tgt_ships_w = int(target.ships) + wait * tgt_prod
        src_w = _src_with_ships(src, avail)
        tgt_w = dataclasses.replace(target, ships=float(tgt_ships_w))
        cand = _solve_single_source(src_w, tgt_w, world, my_id,
                                     centrality_cache, target_is_ours)
        if cand is None:
            continue
        # Lowest total_ships (= cheapest acquisition) wins; tie-break on
        # earlier wait so we don't delay unnecessarily.
        if best is None or (cand.total_ships, wait) < (best[1].total_ships, best[0]):
            best = (wait, cand)
    return best


def backwards_acquisition_plan(world: World, my_id: int,
                                 portfolio: list[int]) -> list[ScheduledLaunch]:
    """Schedule the launches needed to acquire `portfolio` planets in
    order. Returns one ScheduledLaunch per allocation (so multi-source
    bundles produce N records, all with the same target_id and
    turn_offset).

    Source budgets are reserved across targets — once a source is
    committed to target A, it can't double-spend on target B in the
    same plan. This is the closed-form analog of trajectory_roi's
    `joint_solve_forward` greedy assignment; the difference is the
    selection criterion (cheapest, given goal-directed portfolio order)
    vs. trajectory_roi's ROI-marginal scoring.
    """
    centrality_cache = _build_centrality_cache(world)
    my_planets = {p.id: p for p in world.planets if p.owner == my_id}
    # reservations[src_id] = list of (wait_offset, ships_committed).
    reservations: dict[int, list[tuple[int, int]]] = {pid: [] for pid in my_planets}

    scheduled: list[ScheduledLaunch] = []
    for target_id in portfolio:
        target = world._planet_by_id.get(target_id)
        if target is None or target.owner == my_id:
            continue
        target_is_ours = False  # portfolio entries are always not-mine

        # Try every source; pick the cheapest (lowest total_ships) feasible
        # (src, wait) pair respecting prior reservations.
        best_single: tuple[int, int, Candidate] | None = None
        for src_id, src in my_planets.items():
            if _available_ships_at(src, src_id, 0, reservations) < 0:
                continue  # already over-allocated for turn 0 — shouldn't happen
            found = _try_single_source_with_wait(
                src, target, world, my_id,
                centrality_cache, target_is_ours, reservations,
            )
            if found is None:
                continue
            wait, cand = found
            if best_single is None or (cand.total_ships, wait) < (best_single[2].total_ships, best_single[0]):
                best_single = (wait, src_id, cand)

        # Multi-source: clamp each src's ships to its turn-0 availability
        # (multi-source is wait_offset=0 by construction).
        budgeted_planets = tuple(
            _src_with_ships(p,
                            _available_ships_at(p, p.id, 0, reservations))
            if p.id in my_planets else p
            for p in world.planets
        )
        budgeted_world = dataclasses.replace(world, planets=budgeted_planets,
                                              _planet_by_id={p.id: p for p in budgeted_planets})
        multi = _solve_multi_source(target, budgeted_world, my_id,
                                     centrality_cache, target_is_ours)

        # Pick the cheaper of single (best wait) vs multi (turn 0).
        if best_single is None and multi is None:
            continue
        if best_single is None:
            chosen_wait, chosen_cand = 0, multi
        elif multi is None:
            chosen_wait, _, chosen_cand = best_single
        elif multi.total_ships < best_single[2].total_ships:
            chosen_wait, chosen_cand = 0, multi
        else:
            chosen_wait, _, chosen_cand = best_single

        # Reserve and record.
        for a in chosen_cand.allocations:
            reservations.setdefault(a.src_id, []).append((chosen_wait, a.ships))
            scheduled.append(ScheduledLaunch(
                turn_offset=chosen_wait,
                src_id=a.src_id,
                target_id=target_id,
                ships=a.ships,
                aim_angle=a.aim_angle,
            ))
    return scheduled
