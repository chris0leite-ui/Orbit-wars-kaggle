"""Inflection predicate (Step 2 — STUB).

When Step 2 lands this module will provide:

    evaluate_inflection(world, model, me, opp_id, *,
                        max_size=2, horizon=25, margin=1.10) -> Optional[StrikePlan]

Enumerate target subsets S of opp planets (|S| in {1, 2}). For each S,
search the earliest arrival step T such that for every P in S:

    sum_over_my_sources(find_shot_for_arrival(src, P, T).ship_count)
        > model.ships_at(P, T) * margin

AND `lib.joint_solver.predicate.is_winning_state_if_owned(world, me,
opp_id, S)` holds after the hypothetical captures. Return the earliest-T
`StrikePlan(target_ids, arrival_step, shots)` or None.

Step 1 ships this stub so the package imports cleanly and the dispatcher
can route safely to CONSOLIDATION whenever evaluate_inflection returns
None — which is always, in Step 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class StrikePlan:
    """Plan for a single coordinated multi-source wave attack.

    Populated in Step 3 once the precision intercept solver is wired in.
    """
    target_ids: frozenset[int]
    arrival_step: int
    shots: tuple[Any, ...]  # tuple[precision.intercept.Shot, ...]


def opp_id_2p(world, me: int) -> int:
    """Return the single opponent's id in 2P games; -1 in 4P (skip)."""
    seats = set()
    for p in world.planets_by_id.values():
        if p.owner >= 0:
            seats.add(int(p.owner))
    others = [s for s in seats if s != int(me)]
    if len(others) != 1:
        return -1
    return others[0]


def evaluate_inflection(world, model, me: int, opp_id: int, *,
                        max_size: int = 2, horizon: int = 25,
                        margin: float = 1.10) -> Optional[StrikePlan]:
    """Step 1 stub — always returns None. Step 2 will implement.

    Returning None means "stay in CONSOLIDATION" — the safe default.
    """
    return None
