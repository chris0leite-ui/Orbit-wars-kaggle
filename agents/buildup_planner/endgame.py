"""ENDGAME / FINISHER phase — elimination-driver wave.

Captures EVERY remaining opp planet in a single coordinated wave when opp
is reduced to <= K_FINISH planets AND the closed-form
`is_winning_state_if_owned(world, me, opp, all-opp-planets)` gate still
holds. Drives strict ELIMINATION (opp ends with zero planets AND zero
in-flight fleets, terminating the game per `lib/game/interpreter.py:856-864`)
rather than score-leading at the turn cap.

Distinction from STRIKE (`predicates.evaluate_inflection`):
  - STRIKE searches target subsets that flip the production gate; it
    accepts partial capture as "winning eventually."
  - FINISHER fixes |S| = |all_opp_planets| — full territorial sweep.

Per-source budget is enforced UP FRONT during planning (greedy
single-source-per-target assignment with budget deduction). The STRIKE
predicate's optimistic over-count (documented Step-2 limitation) is NOT
mirrored here — the FINISHER plan is already feasible at search time.

Emission re-uses `strike.step`'s atomic-drop machinery: both plan shapes
carry the same `(src_id, tgt_id, eta, ship_count, angle)` shot contract,
so the per-source budget re-check + per-shot `predict_fleet_fate`
re-validation pass is identical.
"""
from __future__ import annotations

import os
from typing import Optional

from lib.joint_solver.predicate import is_winning_state_if_owned

from agents.precision.intercept import (
    SweepCache,
    find_shot_for_arrival,
    parse_world,
)
from agents.buildup_planner.predicates import ETA_MIN, StrikePlan


# K_FINISH bounds the trigger — keep small so the trigger gate is cheap
# and the search space is bounded. HORIZON matches `evaluate_inflection`.
K_FINISH = 3
HORIZON = 25
MARGIN = 1.10


def _enabled() -> bool:
    """Default ON. Env hook for ablation only."""
    return os.environ.get("BUILDUP_PLANNER_FINISHER_ENABLED", "1") == "1"


def quick_trigger(raw_planets, me: int) -> Optional[tuple[int, int]]:
    """Cheap O(|planets|) pre-check before paying the World/Model build.

    Returns `(opp_id, opp_planet_count)` when the game is 2P AND opp owns
    >=1 and <=K_FINISH planets; else None (skip FINISHER this turn).

    `raw_planets` is the obs.planets list of tuples; tuple index 1 is owner.
    4P games (multiple distinct non-me owners) short-circuit to None — the
    closed-form gate is 2P-only, matching `evaluate_inflection`'s policy.
    """
    if not _enabled():
        return None
    opp_id = -1
    opp_count = 0
    for p in raw_planets:
        owner = int(p[1])
        if owner < 0 or owner == int(me):
            continue
        if opp_id < 0:
            opp_id = owner
        elif owner != opp_id:
            # Multiple non-me owners → 4P; FINISHER is 2P-only.
            return None
        opp_count += 1
    if opp_id < 0 or opp_count == 0 or opp_count > K_FINISH:
        return None
    return (opp_id, opp_count)


def evaluate(world, model, me: int, opp_id: int, *,
             k_finish: int = K_FINISH,
             horizon: int = HORIZON,
             margin: float = MARGIN) -> Optional[StrikePlan]:
    """Return a wave that captures EVERY opp planet at one arrival step, or None.

    Triggers only when (a) opp owns >=1 and <=k_finish planets, AND
    (b) `is_winning_state_if_owned(world, me, opp, extra=all-opp-planets)`
    is True (the closed-form gate certifies the post-capture state is a
    win even allowing for opp recovery).

    Per-source ship budget is enforced via greedy assignment during the
    search (no optimistic over-count). Returns the earliest-T feasible
    plan, or None if no T in `[ETA_MIN, horizon]` admits a feasible
    full-board assignment.
    """
    if not _enabled() or opp_id < 0:
        return None

    opp_planet_ids = {
        int(p.id) for p in world.planets_by_id.values()
        if int(p.owner) == int(opp_id)
    }
    if not opp_planet_ids or len(opp_planet_ids) > k_finish:
        return None

    if not is_winning_state_if_owned(
        world, my_id=int(me), opp_id=int(opp_id),
        extra_planet_ids=set(opp_planet_ids),
    ):
        return None

    try:
        world_d = parse_world(world.obs_raw)
    except Exception:
        return None
    cache = SweepCache(world_d["omega"], world_d["step"])

    my_sources = [pv for pv in world_d["planets"]
                  if pv.owner == int(me) and pv.ships >= 1]
    opp_targets = [pv for pv in world_d["planets"]
                   if pv.owner == int(opp_id)]
    if not my_sources or not opp_targets:
        return None

    cur_step = int(world.step)
    for T_offset in range(ETA_MIN, horizon + 1):
        T_abs = cur_step + T_offset
        # Per-source remaining-ship budget for this candidate arrival T.
        budget = {int(s.id): int(s.ships) for s in my_sources}
        # Resolve opp garrisons at T; bail early if any is None (missing
        # from the WorldModel projection — typically near-future capture
        # already in flight by us, in which case the FINISHER assignment
        # can't reason about it cleanly).
        opp_at_T: list[tuple] = []
        skip_T = False
        for tgt_pv in opp_targets:
            g = model.ships_at(int(tgt_pv.id), T_offset)
            if g is None:
                skip_T = True
                break
            opp_at_T.append((tgt_pv, float(g)))
        if skip_T:
            continue
        # Hardest-first: largest projected garrison first, so the
        # easier targets get the leftover sources.
        opp_at_T.sort(key=lambda tg: -tg[1])

        shots_for_T: list = []
        feasible = True
        for tgt_pv, garrison in opp_at_T:
            best_shot = None
            best_src_id = None
            for src_pv in my_sources:
                if int(src_pv.id) == int(tgt_pv.id):
                    continue
                avail = budget.get(int(src_pv.id), 0)
                if avail < 1:
                    continue
                shot = find_shot_for_arrival(
                    src_pv, tgt_pv, T_abs, world_d, cache=cache
                )
                if shot is None:
                    continue
                needed = int(shot.ship_count)
                if needed > avail:
                    continue
                if float(needed) <= garrison * float(margin):
                    continue  # fails the take-the-planet inequality
                if best_shot is None or needed < int(best_shot.ship_count):
                    best_shot = shot
                    best_src_id = int(src_pv.id)
            if best_shot is None:
                feasible = False
                break
            shots_for_T.append(best_shot)
            budget[best_src_id] -= int(best_shot.ship_count)

        if feasible:
            return StrikePlan(
                target_ids=frozenset(int(p.id) for p in opp_targets),
                arrival_step=int(T_abs),
                shots=tuple(shots_for_T),
            )

    return None
