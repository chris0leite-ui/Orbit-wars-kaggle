"""Smallest-sufficient-portfolio identifier.

Ported from origin/claude/ml-competition-strategy-PFhzM:lib/goal_planner/portfolio.py.
Adapted to current World (planets_by_id dict; no .cfg).

Given a world where `is_winning_state` is False, returns the ordered list
of not-currently-mine planet ids whose acquisition would flip the
predicate to True. Greedy by production-per-ship-cost.
"""

from __future__ import annotations

from lib.intent import World
from lib.joint_solver.predicate import (
    EPISODE_STEPS,
    is_winning_state,
    is_winning_state_if_owned,
)


CAPTURE_SAFETY_MARGIN = 2


def _ships_to_capture(planet, my_id: int) -> int:
    """Closed-form lower bound on ships needed to capture this planet."""
    if planet.owner == my_id:
        return 0
    return int(planet.ships) + 1


def _candidate_score(planet, my_id: int) -> tuple[float, int]:
    """Greedy sort key: high production-per-ship-cost first, then cheap first."""
    cost = _ships_to_capture(planet, my_id)
    prod = int(planet.production)
    if cost == 0:
        return (-float("inf"), 0)
    ratio = prod / cost
    return (-ratio, cost)


def _is_feasibly_capturable(target, mine_planets: list) -> bool:
    """Can ANY mine source ever capture this target (closed form)?"""
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


def smallest_winning_portfolio(world: World, my_id: int, opp_id: int,
                               episode_steps: int = EPISODE_STEPS) -> list[int]:
    """Greedy smallest portfolio that flips `is_winning_state`.

    Returns ordered planet ids (priority first). Empty list if no subset
    of available not-mine planets is sufficient, OR if we're already winning.
    """
    if is_winning_state(world, my_id, opp_id, episode_steps):
        return []

    planets = list(world.planets_by_id.values())
    mine_planets = [p for p in planets if p.owner == my_id]
    candidates = [p for p in planets
                  if p.owner != my_id and _is_feasibly_capturable(p, mine_planets)]
    candidates.sort(key=lambda p: _candidate_score(p, my_id))

    portfolio: list[int] = []
    portfolio_set: set[int] = set()
    for p in candidates:
        portfolio.append(p.id)
        portfolio_set.add(p.id)
        if is_winning_state_if_owned(world, my_id, opp_id, portfolio_set,
                                     episode_steps):
            return portfolio
    return []
