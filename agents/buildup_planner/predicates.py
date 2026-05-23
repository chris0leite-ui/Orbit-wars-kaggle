"""Inflection predicate (Step 2 — observation-only).

Enumerates target subsets S of opp planets (|S| in {1, 2}). For each S,
searches the earliest arrival step T in `[step+ETA_MIN, step+horizon]`
such that for every P in S:

    sum_over_my_sources(find_shot_for_arrival(src, P, T).ship_count)
        > model.ships_at(P, T) * margin

AND `lib.joint_solver.predicate.is_winning_state_if_owned(world, me,
opp_id, S)` holds after the hypothetical captures. Returns the earliest-T
`StrikePlan(target_ids, arrival_step, shots)` or None.

Step 2 ships this WITHOUT enforcing per-source ship-budget conflict
(a source may be counted toward multiple targets in the same S — the
optimistic upper bound on what's feasible). The Step 3 striker MUST
re-validate per-shot with `lib.trajectory.predict_fleet_fate` and
atomic-drop the whole strike on ANY failure, including budget overflow.
For Step 2 (observation-only) the over-count inflates the elect-rate;
the audit log measures it so we can size the Step-3 tightening.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Optional

from lib.joint_solver.predicate import is_winning_state_if_owned

from agents.precision import intercept


# Smallest arrival ETA we'll even consider. k=1 = arrives next tick, which
# the precision solver supports; keeping ETA_MIN=2 just gives the inverse
# solver a tick of headroom for the (k-1, k, k+1) verification window.
ETA_MIN = 2


@dataclass(frozen=True)
class StrikePlan:
    """Plan for a single coordinated multi-source wave attack."""
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
    """Search for the earliest coordinated capture that flips the predicate.

    Returns the earliest-T StrikePlan found, or None.

    `world`  — lib.intent.World (used for is_winning_state_if_owned).
    `model`  — lib.world_model.WorldModel (model.ships_at gives the
               opp garrison at offset T from now).
    `me`     — my seat id.
    `opp_id` — single opponent id (-1 = 4P; caller MUST short-circuit).
    """
    if opp_id < 0:
        return None  # 4P — predicate is 2P-only.

    # Build the precision world view once per call. World.obs_raw holds
    # the same obs dict / struct that the dispatcher received from
    # Kaggle, so this is the canonical input to parse_world.
    try:
        world_d = intercept.parse_world(world.obs_raw)
    except Exception:
        return None

    cache = intercept.SweepCache(world_d["omega"], world_d["step"])

    my_sources = [pv for pv in world_d["planets"]
                  if pv.owner == me and pv.ships >= 1]
    opp_targets = [pv for pv in world_d["planets"] if pv.owner == opp_id]
    if not my_sources or not opp_targets:
        return None

    # Enumerate subsets size 1 → max_size. We return the earliest-T plan
    # across all subsets, so we have to look at every subset for every T
    # before declaring "earliest." But we can prune by T-first: for each
    # T, try every subset; first feasible candidate at that T wins (and
    # earlier-T always beats later-T).
    subsets: list[tuple] = []
    for size in range(1, max_size + 1):
        for S in combinations(opp_targets, size):
            subsets.append(S)

    cur_step = int(world.step)
    for T_offset in range(ETA_MIN, horizon + 1):
        T_abs = cur_step + T_offset
        for S in subsets:
            # Per-target feasibility + accumulate shots.
            shots_for_S: list = []
            feasible = True
            for tgt_pv in S:
                shots_for_P: list = []
                total_ships = 0
                for src_pv in my_sources:
                    if src_pv.id == tgt_pv.id:
                        continue
                    shot = intercept.find_shot_for_arrival(
                        src_pv, tgt_pv, T_abs, world_d, cache=cache
                    )
                    if shot is None:
                        continue
                    shots_for_P.append(shot)
                    total_ships += int(shot.ship_count)
                opp_garrison = model.ships_at(int(tgt_pv.id), T_offset)
                if opp_garrison is None:
                    feasible = False
                    break
                if total_ships <= float(opp_garrison) * float(margin):
                    feasible = False
                    break
                shots_for_S.extend(shots_for_P)
            if not feasible:
                continue
            # Closed-form gate: after these captures, does the
            # production lead overwhelm opp's recovery?
            target_ids = frozenset(int(p.id) for p in S)
            if not is_winning_state_if_owned(
                world, my_id=int(me), opp_id=int(opp_id),
                extra_planet_ids=set(target_ids)
            ):
                continue
            # First feasible (T, S) at the smallest T wins.
            return StrikePlan(
                target_ids=target_ids,
                arrival_step=int(T_abs),
                shots=tuple(shots_for_S),
            )

    return None
