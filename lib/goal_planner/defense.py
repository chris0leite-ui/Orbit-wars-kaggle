"""Defense actions for goal-directed planner (P4).

Identifies mine planets under threat from incoming opp fleets and
schedules reinforcement launches from non-committed sources.

Unlike trajectory_roi's `_solve_single_source(target_is_ours=True)`,
which evaluates net deficit at the *reinforcer's* arrival turn (and
returns None when the reinforcer arrives before opp fleets do — i.e.,
the common preemptive case), this module uses a defense-specific sizer:

  1. Find the latest opp fleet ETA T_max aimed at the target.
  2. Sum opp ships arriving by T_max.
  3. Compute my garrison at T_max without reinforce: ships + prod * T_max.
  4. Deficit = opp_total - my_garrison + safety.
  5. Sample sources by their ETA to target; require reinforce ETA <= T_max
     AND source has enough ships in its remaining budget.
  6. Pick the cheapest (smallest total_ships) feasible reinforcer.

Priority: threatened mine planets ranked by `production × remaining_turns`
(contribution to the winning-state predicate).

Out of scope for v1: defending portfolio members not yet captured
(interception). Deferred.
"""

from __future__ import annotations

import math

from agents.trajectory_roi.main import (
    Allocation, Candidate, _aim_and_eta, _fleet_eta_to_planet,
)
from lib.goal_planner.predicate import remaining_turns
from lib.goal_planner.sequencer import (
    MIN_LAUNCH_SHIPS, ScheduledLaunch, _available_ships_at, _src_with_ships,
)
from lib.goal_planner.validate import launch_reaches_target
from lib.trajectory_layer import World


SAFETY_MARGIN = 2


def _opp_threat_summary(world: World, my_id: int, target
                          ) -> tuple[int, int] | None:
    """Return (latest_opp_eta, total_opp_ships_arriving_by_then) or None
    if no opp fleet is aimed at this target."""
    threats: list[tuple[int, int]] = []
    for f in world.fleets:
        if f.owner == my_id:
            continue
        eta = _fleet_eta_to_planet(f, target)
        if eta is None:
            continue
        threats.append((eta, int(f.ships)))
    if not threats:
        return None
    T_max = max(eta for eta, _ in threats)
    total = sum(s for eta, s in threats if eta <= T_max)
    return T_max, total


def _solve_defense(src, target, world: World, my_id: int,
                    src_ships: int) -> Candidate | None:
    """Closed-form reinforce sizing: smallest K from src to target such
    that my garrison at the opp's latest arrival turn (post-reinforce)
    exceeds opp's total threat by SAFETY_MARGIN.

    `src_ships` is the source's available budget (caller passes the
    reservation-adjusted value).
    """
    threat = _opp_threat_summary(world, my_id, target)
    if threat is None:
        return None
    T_max, total_opp = threat
    my_garrison_at_T = int(target.ships) + int(target.production) * T_max
    deficit = total_opp - my_garrison_at_T + SAFETY_MARGIN
    if deficit <= 0:
        return None  # garrison already sufficient
    K = max(MIN_LAUNCH_SHIPS, int(deficit))
    if K > src_ships:
        return None
    ae = _aim_and_eta(src, target, K, world.omega)
    if ae is None:
        return None
    angle, eta = ae
    if eta > T_max:
        return None  # reinforcer too slow — wouldn't arrive in time
    return Candidate(
        flavor="defense",
        target_id=target.id,
        arrival_turn=eta,
        allocations=(Allocation(src.id, K, angle),),
        raw_value=float(deficit),
        total_ships=K,
    )


def defense_actions(world: World, my_id: int, opp_id: int,
                     reservations: dict[int, list[tuple[int, int]]] | None = None
                     ) -> list[ScheduledLaunch]:
    """For each mine planet under incoming opp threat, schedule a
    reinforcement launch from the cheapest non-committed source.

    Targets are processed in descending priority `production × remaining`.
    Source budgets honored via the per-source per-turn reservation pool
    shared with the sequencer (P3); pass `reservations` to chain.

    Returns ScheduledLaunch records with `turn_offset == 0` (defense is
    always immediate)."""
    if reservations is None:
        reservations = {}
    my_planets = {p.id: p for p in world.planets if p.owner == my_id}
    rem = remaining_turns(world)

    threatened: list[tuple[float, int]] = []
    for pid, p in my_planets.items():
        if _opp_threat_summary(world, my_id, p) is not None:
            threatened.append((float(p.production) * float(rem), pid))
    threatened.sort(reverse=True)

    scheduled: list[ScheduledLaunch] = []
    for _priority, tgt_id in threatened:
        target = my_planets[tgt_id]
        # Collect all feasible reinforcers (sorted cheapest first), then
        # pick the cheapest that passes physics validation. Same late-
        # with-fallback pattern as the sequencer (P3).
        options: list[tuple[int, Candidate]] = []
        for src_id, src in my_planets.items():
            if src_id == tgt_id:
                continue
            avail = _available_ships_at(src, src_id, 0, reservations)
            if avail < MIN_LAUNCH_SHIPS:
                continue
            cand = _solve_defense(src, target, world, my_id, src_ships=avail)
            if cand is None:
                continue
            options.append((src_id, cand))
        options.sort(key=lambda sc: sc[1].total_ships)

        chosen: Candidate | None = None
        for _src_id, cand in options:
            ok = True
            for a in cand.allocations:
                src_p = my_planets.get(a.src_id)
                if src_p is None or not launch_reaches_target(
                        src_p, target, a.aim_angle, a.ships, world):
                    ok = False
                    break
            if ok:
                chosen = cand
                break
        if chosen is None:
            continue
        for a in chosen.allocations:
            reservations.setdefault(a.src_id, []).append((0, a.ships))
            scheduled.append(ScheduledLaunch(
                turn_offset=0,
                src_id=a.src_id,
                target_id=tgt_id,
                ships=a.ships,
                aim_angle=a.aim_angle,
            ))
    return scheduled
