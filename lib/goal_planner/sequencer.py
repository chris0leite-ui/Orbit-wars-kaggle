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
from lib.goal_planner.validate import launch_reaches_target
from lib.trajectory_layer import World


MAX_WAIT_TURNS = 30
MIN_LAUNCH_SHIPS = 5  # env minimum; matches trajectory_roi's constant
MAX_VALIDATE_PER_TARGET = 12  # cap fallback iterations to keep p95 bounded


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


def _collect_single_source_options(src, target, world: World, my_id: int,
                                      centrality_cache, target_is_ours: bool,
                                      reservations: dict[int, list[tuple[int, int]]]
                                      ) -> list[tuple[int, Candidate]]:
    """Return ALL feasible (wait_offset, Candidate) pairs for src→target,
    sorted cheapest-first. Used by `backwards_acquisition_plan` to fall
    through to alternates when the cheapest fails physics validation
    (e.g. trajectory crosses the sun).

    Per-turn budget model: at wait W, source has
    `current_ships + W*production - sum(reservations at wait <= W)` ships
    available. Target defenders accrue at `tgt.ships + W*tgt.production`
    (for neutrals, env rules say neutrals don't actually accrue — but
    `_solve_single_source`'s `_net_defenders` already handles that
    distinction)."""
    src_id = src.id
    tgt_prod = int(target.production)
    options: list[tuple[int, Candidate]] = []
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
        options.append((wait, cand))
    # Lowest total_ships first; tie-break on earlier wait.
    options.sort(key=lambda wc: (wc[1].total_ships, wc[0]))
    return options


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

        # Collect ALL feasible options (single-source per src + multi-source),
        # then iterate in cheapest-first order and accept the first that
        # passes the physics gate (`launch_reaches_target`). This is the
        # late-with-fallback pattern — primitives like _aim_and_eta don't
        # check sun safety, so we drop trajectory-invalid options here.
        options: list[tuple[int, Candidate]] = []
        for src_id, src in my_planets.items():
            if _available_ships_at(src, src_id, 0, reservations) < 0:
                continue
            options.extend(_collect_single_source_options(
                src, target, world, my_id,
                centrality_cache, target_is_ours, reservations,
            ))

        # Multi-source: clamp each src's ships to its turn-0 availability.
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
        if multi is not None:
            options.append((0, multi))
        if not options:
            continue
        options.sort(key=lambda wc: (wc[1].total_ships, wc[0]))
        # Cap validation to keep p95 bounded. predict_fleet_fate is
        # ~1-2ms per call; without the cap, a sun-blocked source can
        # blow through the entire wait sweep (30 options × 5 sources =
        # 150 calls per target). Top-12 cheapest is enough fallback to
        # find a valid alternative if one exists.
        options = options[:MAX_VALIDATE_PER_TARGET]

        chosen: tuple[int, Candidate] | None = None
        for wait, cand in options:
            # Validate every allocation's trajectory. Multi-source
            # bundles can have one bad leg; we require ALL legs valid.
            ok = True
            for a in cand.allocations:
                src_p = my_planets.get(a.src_id)
                if src_p is None:
                    ok = False
                    break
                if not launch_reaches_target(src_p, target, a.aim_angle,
                                              a.ships, world):
                    ok = False
                    break
            if ok:
                chosen = (wait, cand)
                break
        if chosen is None:
            continue
        chosen_wait, chosen_cand = chosen

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
