"""Mid-game multi-source dogpile — coordinated capture of top-K opp planets
by production.

Distinction from STRIKE / FINISHER:
  - STRIKE (`predicates.evaluate_inflection`) — searches target subsets that
    flip `is_winning_state_if_owned`. The gate requires the post-capture
    state to be a *guaranteed* win allowing for opp recovery; rarely fires
    mid-game when opp still has production to recover.
  - FINISHER (`endgame.evaluate`) — fixes the target subset to ALL opp
    planets and only triggers when |opp| <= K_FINISH. Endgame elimination.
  - DOGPILE (this module) — fires mid-game on the TOP-K opp planets by
    production, gated by a relaxed `our_production > opp_production * margin`
    inequality after the hypothetical captures. Goal: snip opp's production
    base to establish a long-term economic lead without waiting for the
    closed-form win-gate to certify.

Per-source ship budget is enforced UP FRONT during planning (greedy
single-source-per-target assignment with budget deduction). On any infeasible
T or assignment, returns None — caller falls through to baseline.

Emission re-uses `strike.step`'s atomic-drop machinery via main.py wiring.
"""
from __future__ import annotations

import os
from typing import Optional

from agents.precision.intercept import (
    SweepCache,
    find_shot_for_arrival,
    parse_world,
)
from agents.buildup_planner.predicates import ETA_MIN, StrikePlan


# Default tuning. K is small to bound search; HORIZON shorter than FINISHER's
# because mid-game opp can rebuild faster — we want NEAR-future captures.
K_DOGPILE = 3
HORIZON = 15
MARGIN = 1.20  # take-the-planet inequality (more conservative than FINISHER's 1.10)
PROD_ADVANTAGE_MARGIN = 1.10  # post-capture our_prod > opp_prod * THIS


def _enabled() -> bool:
    """Default OFF for Phase 1 land; flipped ON in Phase 1.5 after A/B."""
    return os.environ.get("BUILDUP_PLANNER_DOGPILE_ENABLED", "0") == "1"


def quick_trigger(raw_planets, me: int, k_finish: int = 3) -> Optional[tuple[int, int]]:
    """Cheap O(|planets|) pre-check before paying the World/Model build.

    Returns `(opp_id, opp_planet_count)` when the game is 2P AND opp owns
    > k_finish planets (i.e. NOT yet endgame — FINISHER handles that). 4P
    games short-circuit to None (multi-opp; this module is 2P-only).
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
            return None  # 4P
        opp_count += 1
    if opp_id < 0 or opp_count <= k_finish:
        # opp_count<=k_finish is FINISHER's territory — defer to it.
        return None
    return (opp_id, opp_count)


def _post_capture_prod_advantage(
    world, me: int, opp_id: int, captured_ids: set[int],
) -> tuple[float, float]:
    """Closed-form: total production we'd own after captures vs opp."""
    my_prod = 0.0
    opp_prod = 0.0
    for p in world.planets_by_id.values():
        owner = int(p.owner)
        pid = int(p.id)
        prod = float(getattr(p, "production", 0.0))
        if pid in captured_ids:
            my_prod += prod
        elif owner == int(me):
            my_prod += prod
        elif owner == int(opp_id):
            opp_prod += prod
    return my_prod, opp_prod


def evaluate(
    world, model, me: int, opp_id: int, *,
    k: int = K_DOGPILE,
    horizon: int = HORIZON,
    margin: float = MARGIN,
    prod_advantage_margin: float = PROD_ADVANTAGE_MARGIN,
) -> Optional[StrikePlan]:
    """Return a wave that captures the top-K opp planets at one arrival T, or None.

    Triggers regardless of |opp planets|; caller MUST gate via `quick_trigger`
    to avoid stealing FINISHER's territory.

    Search strategy:
      1. Pick top-K opp planets by production (descending).
      2. For T_offset in [ETA_MIN, horizon], try to greedy-assign sources
         to targets (hardest-garrison-first), honoring per-source ship budget.
      3. On the first feasible T, check the relaxed gate
         `our_prod_post > opp_prod_post * prod_advantage_margin`.
      4. Emit the plan; else continue to next T or return None.
    """
    if not _enabled() or opp_id < 0:
        return None

    opp_targets_all = [
        p for p in world.planets_by_id.values()
        if int(p.owner) == int(opp_id)
    ]
    if len(opp_targets_all) <= k:
        # Caller should have gated this on quick_trigger, but defensive:
        # if opp has <= k planets, this is FINISHER's job, not dogpile's.
        return None

    # Top-K by production, descending. Tie-break by id for determinism.
    opp_targets_all.sort(key=lambda p: (-float(getattr(p, "production", 0.0)), int(p.id)))
    top_k_targets = opp_targets_all[:k]
    top_k_ids = {int(p.id) for p in top_k_targets}

    # Closed-form gate FIRST (cheap, avoids the wallclock of the search if
    # the post-capture state isn't economically winning enough).
    my_prod_post, opp_prod_post = _post_capture_prod_advantage(
        world, me, opp_id, top_k_ids,
    )
    if my_prod_post <= opp_prod_post * float(prod_advantage_margin):
        return None

    try:
        world_d = parse_world(world.obs_raw)
    except Exception:
        return None
    cache = SweepCache(world_d["omega"], world_d["step"])

    my_sources = [pv for pv in world_d["planets"]
                  if pv.owner == int(me) and pv.ships >= 1]
    if not my_sources:
        return None

    # PlanetView versions of our top-K targets (same id, different object).
    top_k_pvs = [
        pv for pv in world_d["planets"] if int(pv.id) in top_k_ids
    ]
    if len(top_k_pvs) < k:
        # Some target id missing from parse_world; bail safely.
        return None

    cur_step = int(world.step)
    for T_offset in range(ETA_MIN, horizon + 1):
        T_abs = cur_step + T_offset
        budget = {int(s.id): int(s.ships) for s in my_sources}

        # Resolve opp garrisons at T; bail early if any None.
        opp_at_T: list[tuple] = []
        skip_T = False
        for tgt_pv in top_k_pvs:
            g = model.ships_at(int(tgt_pv.id), T_offset)
            if g is None:
                skip_T = True
                break
            opp_at_T.append((tgt_pv, float(g)))
        if skip_T:
            continue
        # Hardest-first: largest projected garrison first, so the easier
        # targets get the leftover sources.
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
                    src_pv, tgt_pv, T_abs, world_d, cache=cache,
                )
                if shot is None:
                    continue
                needed = int(shot.ship_count)
                if needed > avail:
                    continue
                if float(needed) <= garrison * float(margin):
                    continue  # fails take-the-planet inequality
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
                target_ids=frozenset(top_k_ids),
                arrival_step=int(T_abs),
                shots=tuple(shots_for_T),
            )

    return None
