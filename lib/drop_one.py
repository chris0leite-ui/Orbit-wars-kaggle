"""Layer D — plan-level drop-one validator.

After the chooser/MILP picks a plan (set of launches), each leg has
been valued individually. In a submodular objective (which Φ is
approximately), legs that look positive individually can be dominated
once other legs are committed. Drop-one runs leave-one-out: for each
leg, recompute total plan value without that leg; if the marginal
contribution is below SAFETY_MARGIN, prune the leg.

This catches "speculative leg adds nominal but contributes zero
marginal because the larger leg already wins the same target."

Plan value approximation (closed-form, no rollout):
  sum over distinct captured targets of: production · remaining_turns

A leg that captures the same target as a faster leg is dominated
(its capture is redundant). A leg that bounces contributes 0.

O(N) plan-value evaluations per turn where N is plan size (≤ ~6 turns
typical), each one a few `predict_fleet_fate` calls. Cheap.
"""
from __future__ import annotations

import os
from typing import Any, Iterable

from lib.trajectory import predict_fleet_fate

# Default plan-value horizon — match PI's fast-elim target (Rule 48 + the
# 2026-05-24 truncated-A/B protocol). Tunable via env var.
_PLAN_VALUE_HORIZON = int(os.environ.get("DROP_ONE_HORIZON", "250"))

DROP_ONE_ENABLED = os.environ.get("BASELINE_DROP_ONE_VALIDATE", "0") == "1"
SAFETY_MARGIN = float(os.environ.get("BASELINE_DROP_ONE_SAFETY", "30.0"))


def _fleet_outcome(src, angle: float, ships: int, world):
    """Trace a fleet from src at `angle` with `ships` and return
    (hit_planet_id, eta_step) for the first planet collision, or
    (None, None) for sun/oob/timeout."""
    try:
        # Use src as the dummy target; we read hit_planet_id directly
        # (predict_fleet_fate sets it for any planet collision, regardless
        # of whether outcome is "target" or "planet").
        fate = predict_fleet_fate(src, src, angle, ships, world, wait_N=0)
    except Exception:
        return None, None
    if fate is None or fate.hit_planet_id is None:
        return None, None
    return int(fate.hit_planet_id), int(fate.step)


def plan_production_advantage(moves, world, model, my_id: int) -> float:
    """Closed-form estimate of total production swing this plan delivers
    over the next DROP_ONE_HORIZON ticks.

    For each unique target a launch successfully captures, adds
    `target.production · remaining_turns_after_arrival` to the total.

    Bounces, reinforces (target already ours), and duplicate-target legs
    (only the fastest counts as the unique capture) contribute zero.
    """
    captured = {}  # tgt_id -> (eta, ships_delivered)
    step = int(getattr(world, "step", 0))
    for m in moves:
        try:
            src_id = int(m[0])
            angle = float(m[1])
            ships = int(m[2])
        except (TypeError, ValueError, IndexError):
            continue
        src = world.planets_by_id.get(src_id)
        if src is None:
            continue
        hit_id, eta = _fleet_outcome(src, angle, ships, world)
        if hit_id is None or eta is None:
            continue
        # Reinforce check: target already ours at arrival → not a capture.
        try:
            pred_owner = model.owner_at(int(hit_id), int(eta))
        except Exception:
            pred_owner = None
        if pred_owner == int(my_id):
            continue
        # Bounce check: ships ≤ garrison-at-arrival → no capture.
        try:
            pred_ships = float(model.ships_at(int(hit_id), int(eta)) or 0.0)
        except Exception:
            pred_ships = 0.0
        if ships <= pred_ships:
            continue
        # Successful capture; keep only the fastest-arriving capture per target.
        prev = captured.get(hit_id)
        if prev is None or eta < prev[0]:
            captured[hit_id] = (eta, ships)

    total = 0.0
    for hit_id, (eta, _) in captured.items():
        tgt = world.planets_by_id.get(hit_id)
        if tgt is None:
            continue
        remaining = max(0, _PLAN_VALUE_HORIZON - step - eta)
        total += float(tgt.production) * float(remaining)
    return total


def drop_one_validate(
    moves, world, model, my_id: int, *,
    safety_margin: float | None = None,
) -> list:
    """Prune legs whose marginal contribution to plan value is below
    `safety_margin` (default: env var BASELINE_DROP_ONE_SAFETY = 30.0
    production-ticks, ~one prod=3 planet captured for 10 turns).

    O(N+1) plan-value evaluations where N = len(moves).
    Idempotent on no-op (≤1 moves, or all marginal).
    """
    if not DROP_ONE_ENABLED:
        return list(moves)
    if len(moves) <= 1:
        return list(moves)
    margin = SAFETY_MARGIN if safety_margin is None else float(safety_margin)

    full_value = plan_production_advantage(moves, world, model, my_id)
    keep = []
    for i, move in enumerate(moves):
        without_i = list(moves[:i]) + list(moves[i+1:])
        value_without = plan_production_advantage(
            without_i, world, model, my_id,
        )
        marginal = full_value - value_without
        if marginal >= margin:
            keep.append(move)
        # else: this leg contributes < safety_margin marginal -> DROP
    return keep
