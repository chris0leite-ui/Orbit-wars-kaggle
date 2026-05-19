"""Smallest-sufficient-portfolio identifier (P2).

Given a world where `is_winning_state` is currently False, return the
ordered list of not-currently-mine planet ids that we must acquire to
flip the predicate to True. Ordered by acquisition cost; the smallest
set wins.

Algorithm: greedy by production-per-ship-cost. Iterate candidates in
descending order of (production / ships_to_capture); accumulate into
the portfolio until the hypothetical predicate flips. Returns `[]` if
no subset of available planets is sufficient (the agent's later phases
handle this with a best-effort fallback).
"""

from __future__ import annotations

from lib.goal_planner.predicate import (
    is_winning_state, is_winning_state_if_owned,
)
from lib.trajectory_layer import World


def _ships_to_capture(planet, my_id: int) -> int:
    """Closed-form lower bound on ships needed to capture this planet
    from anywhere (ignoring ETA and defender production).

    For neutrals: planet.ships + 1.
    For opp: planet.ships + 1 (we ignore production accrual during
    flight here; the sequencer (P3) will refine sizing per-source per-ETA.
    For the *portfolio identifier*, this is the marginal cost signal —
    not the actual launch sizing.

    Mine planets are not eligible (they're not part of the portfolio).
    """
    if planet.owner == my_id:
        return 0
    return int(planet.ships) + 1


def _candidate_score(planet, my_id: int) -> tuple[float, int]:
    """Sort key for greedy: prioritize HIGH production-per-ship-cost,
    break ties by LOWER ship cost (cheaper first).

    Return is a tuple to use in `sorted(... key=lambda)` where we negate
    the primary signal so descending becomes ascending.
    """
    cost = _ships_to_capture(planet, my_id)
    prod = int(planet.production)
    if cost == 0:
        return (-float("inf"), 0)
    ratio = prod / cost
    return (-ratio, cost)


def smallest_winning_portfolio(world: World, my_id: int,
                                 opp_id: int) -> list[int]:
    """Greedy build of the smallest portfolio that flips `is_winning_state`.

    Returns an ordered list of planet ids (highest priority first). Empty
    list if no subset of available not-mine planets can flip the
    predicate.

    Special case: if `is_winning_state(world, my_id, opp_id)` is already
    True, returns `[]` (no acquisition needed — caller can route to
    defense-only).
    """
    if is_winning_state(world, my_id, opp_id):
        return []

    candidates = [p for p in world.planets if p.owner != my_id]
    candidates.sort(key=lambda p: _candidate_score(p, my_id))

    portfolio: list[int] = []
    portfolio_set: set[int] = set()
    for p in candidates:
        portfolio.append(p.id)
        portfolio_set.add(p.id)
        if is_winning_state_if_owned(world, my_id, opp_id, portfolio_set):
            return portfolio

    return []
