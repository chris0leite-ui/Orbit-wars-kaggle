"""chooser_roi — ROI-prior + opp-modifier chooser.

Architectural pivot (2026-05-19, PI-directed): replace the trajectory rollout
foundation with a closed-form ROI prior + thin opp-vulnerability posterior.
Dispatched via BASELINE_CHOOSER=roi from agents/baseline/main.py.

Pipeline per turn:
  1. solo_roi(src, tgt, ships, eta, wait_N) computes a closed-form ROI per
     proposer candidate using lib/scoring + lib/world_model primitives.
  2. (Phase 3, pending) coalition_roi enumerates N-way joint launches per
     target via merged arrival-ledger walk; emit if joint > sum-of-solo.
  3. (Phase 4, pending) opp_modifier_check scans emit set for exposed
     sources and downsizes/drops candidates the opp would profitably
     counter-target.

Current implementation: Phase 2 (solo_roi + greedy emit). No rollout.

See /root/.claude/plans/okay-we-can-do-elegant-lampson.md.
"""

from __future__ import annotations

import math
import os

from lib.scoring import (
    T_TOTAL_DEFAULT,
    expected_hold,
    margin_multiplier,
    pv_horizon,
)


# Cost coefficients — to be calibrated via bench in Phase 6. Initial
# values are deliberately conservative so the ROI prior favours
# small-cost moves over speculative captures while we tune.
SHIP_COST_COEF: float = float(os.environ.get("ROI_SHIP_COST", "0.05"))
WAIT_COST_COEF: float = float(os.environ.get("ROI_WAIT_COST", "0.5"))


def solo_roi(
    src,
    tgt,
    ships: int,
    eta: int,
    wait_N: int,
    world,
    model,
    me: int,
    step: int,
    max_horizon: int,
    gamma: float = 0.99,
) -> float:
    """Closed-form ROI score for one (src, tgt, ships, eta, wait_N).

    Returns -inf for refused candidates (past horizon, bounced combat).
    Captures bounce with a small negative ship-cost penalty so the
    chooser's "do nothing" baseline (ROI=0) outranks them. Reinforce
    moves (target already ours) return 0 — no margin gained.
    """
    arrival = int(wait_N) + int(eta)
    if arrival > max_horizon:
        return float("-inf")

    arrivals = list(model.ledger.get(int(tgt.id), []))
    arrivals.append((arrival, int(me), int(ships)))

    from lib.world_model import predict_garrison_at  # local — keeps lib/ env-free
    owner_arr, _garrison_arr = predict_garrison_at(tgt, arrival, arrivals)

    if owner_arr != me:
        return -SHIP_COST_COEF * int(ships)

    if int(tgt.owner) == me:
        return 0.0  # reinforce — proposer-side threat reinforce handles defense

    hold = expected_hold(int(tgt.id), arrival, world, model, t_total=T_TOTAL_DEFAULT)
    if hold <= 0:
        return -SHIP_COST_COEF * int(ships)

    hold_end_step_offset = arrival + hold
    pv_held = pv_horizon(int(step), arrival, gamma=gamma,
                         t_total=int(step) + hold_end_step_offset)
    mult = margin_multiplier(tgt, me)

    gross = mult * float(tgt.production) * pv_held
    ship_cost = SHIP_COST_COEF * int(ships)
    wait_cost = WAIT_COST_COEF * int(wait_N) * float(src.production)

    return gross - ship_cost - wait_cost


def choose_roi(
    snap_base,
    prerank,
    me: int,
    num_seats: int,
    wallclock_ms: float,
    min_horizon: int,
    max_horizon: int,
    gamma: float,
    world,
    model,
    step: int,
) -> list:
    """ROI-prior greedy emit.

    For each proposer candidate, score via solo_roi. Sort descending.
    Greedy commit one launch per source per target, fire-now only.
    Reinforce candidates (target already ours) score 0 and never emit
    here — proposer-side threat reinforce handles defense.
    """
    if not prerank:
        return []

    scored: list = []
    for entry in prerank:
        _cheap, src, tgt, ships, angle, eta, _horizon, wait_N = entry
        score = solo_roi(
            src, tgt, int(ships), int(eta), int(wait_N),
            world, model, int(me), int(step), int(max_horizon),
            gamma=gamma,
        )
        if score == float("-inf"):
            continue
        if score <= 0.0:
            continue
        scored.append((score, src, tgt, int(ships), float(angle), int(wait_N)))

    if not scored:
        return []

    scored.sort(key=lambda e: -e[0])

    used_srcs: set[int] = set()
    used_tgts: set[int] = set()
    moves: list[list] = []
    for _score, src, tgt, ships, angle, wait_N in scored:
        sid, tid = int(src.id), int(tgt.id)
        if sid in used_srcs or tid in used_tgts:
            continue
        used_srcs.add(sid)
        used_tgts.add(tid)
        if wait_N == 0:
            moves.append([sid, angle, ships])
    return moves
