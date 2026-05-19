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


CAPTURE_SAFETY_MARGIN = 2


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


def _is_feasibly_capturable(target, mine_planets: list) -> bool:
    """Closed-form check: can ANY mine source eventually capture this
    target?

    Two ways to capture:
      a) NOW: some source already has > target.ships + safety
      b) WAIT: some source's prod >= target.prod (so we accumulate
         faster than the defender — we'll eventually overcome). Edge:
         if target prod is 0, any positive src prod works.

    If neither holds across all sources, we can NEVER acquire this
    target — exclude from the portfolio. Targets with no source within
    MAX_ETA range are also infeasible but we don't check geometry here
    (the sequencer (P3) will reject those; portfolio's job is the
    ship-pool / prod-rate gate)."""
    t_ships = int(target.ships)
    t_prod = int(target.production)
    for src in mine_planets:
        s_ships = int(src.ships)
        s_prod = int(src.production)
        if s_ships > t_ships + CAPTURE_SAFETY_MARGIN:
            return True
        if s_prod >= t_prod and s_prod > 0:
            return True
    return False


def smallest_winning_portfolio(world: World, my_id: int,
                                 opp_id: int) -> list[int]:
    """Greedy build of the smallest portfolio that flips `is_winning_state`.

    Returns an ordered list of planet ids (highest priority first). Empty
    list if no subset of available not-mine planets can flip the
    predicate.

    Filtering: candidates we can never catch up to (target prod outruns
    ALL mine sources' prod AND no source can capture now) are excluded.
    Greedy ratio sort would otherwise prefer high-prod-high-cost targets
    that bait us into unwinnable wait-then-fire chases.

    Special case: if `is_winning_state(world, my_id, opp_id)` is already
    True, returns `[]` (no acquisition needed — caller can route to
    defense-only).
    """
    if is_winning_state(world, my_id, opp_id):
        return []

    mine_planets = [p for p in world.planets if p.owner == my_id]
    candidates = [p for p in world.planets
                  if p.owner != my_id and _is_feasibly_capturable(p, mine_planets)]
    candidates.sort(key=lambda p: _candidate_score(p, my_id))

    portfolio: list[int] = []
    portfolio_set: set[int] = set()
    for p in candidates:
        portfolio.append(p.id)
        portfolio_set.add(p.id)
        if is_winning_state_if_owned(world, my_id, opp_id, portfolio_set):
            return portfolio

    return []
