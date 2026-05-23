"""Per-(src, tgt, launch_tick) candidate enumeration.

Uses the precision-physics substrate already on this branch:

  aim_and_eta(src, tgt, ships, omega, wait_N=)   from agents.baseline.proposer
  predict_fleet_fate(src, tgt, angle, ships, world, wait_N=)   from lib.trajectory
  predict_garrison_at(tgt, eta, arrivals)        from lib.world_model
  _target_holdable_after_capture(...)            from agents.baseline.proposer
                                                  (the B1 fix gated by
                                                  BASELINE_ORBITAL_SAFETY=1
                                                  → +63 μ on the ladder)

A candidate survives only if predict_fleet_fate confirms the fleet hits
the target (not sun, not OOB, not another planet, not timeout), the
delivered ships beat the predicted garrison at arrival, and the capture
is holdable against the nearest opp's counter-launch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from agents.baseline.proposer import (
    _target_holdable_after_capture,
    aim_and_eta,
)
from lib.trajectory import predict_fleet_fate
from lib.world_model import predict_garrison_at


EPISODE_STEPS = 500
MAX_LAUNCH_TICK = 15     # search wait_N ∈ {0..15} (wide enough to let
                         #   low-ship sources accrue production before firing)
SHIP_REFINE_ITERS = 2    # 2-pass ships ↔ eta fixed-point
MIN_FLEET = 1


@dataclass
class Candidate:
    src_id: int
    tgt_id: int
    launch_tick: int
    angle: float
    ships: int
    eta: int
    arrival_step: int
    value: float            # capture NPV (own production stream gained)


def _refine_ships(src, tgt, launch_tick, omega, base_arrivals):
    """Two-pass: estimate ships needed to capture at arrival tick.

    Returns (ships, angle, eta, arrival_step) or None on failure.
    """
    ships = max(MIN_FLEET, int(tgt.ships) + 1)
    angle = 0.0
    eta = 0
    arrival_step = 0
    for _ in range(SHIP_REFINE_ITERS):
        res = aim_and_eta(src, tgt, ships, omega, wait_N=launch_tick)
        if res is None:
            return None
        angle, eta = res
        arrival_step = int(launch_tick) + int(eta)
        _owner_at_arr, gar_at_arr = predict_garrison_at(
            tgt, arrival_step, base_arrivals,
        )
        needed = int(math.ceil(float(gar_at_arr))) + 1
        if needed == ships:
            break
        ships = max(MIN_FLEET, needed)
    return ships, float(angle), int(eta), int(arrival_step)


def _capture_value(tgt, arrival_step: int) -> float:
    """NPV of capturing `tgt` at `arrival_step`: own-side production stream
    from arrival to episode end."""
    remaining = max(0, EPISODE_STEPS - int(arrival_step))
    return float(tgt.production) * float(remaining)


def enumerate_candidates(world, model, my_id: int, omega: float,
                         comet_ids: set | None = None) -> list[Candidate]:
    """Enumerate every viable (src, tgt, launch_tick) capture candidate.

    Filters applied (in order):
      1. ships needed ≤ source's current garrison (time-indexed).
      2. predict_fleet_fate.outcome == "target" (not sun / OOB / wrong planet).
      3. delivered ships > predicted garrison at arrival (capture succeeds).
      4. target NOT already ours at arrival (skip pure reinforce in v1).
      5. _target_holdable_after_capture (B1 — opp can't recapture cheaply).
         RELAXED in dominant-endgame: when we own ≥3× as many planets as
         the opponent, the recapture-risk model over-rejects (opp's few
         remaining planets WILL counter, but our wide base lets us
         re-take cheaply). The PI elim-gate failure on seed 14514 was the
         hold filter blocking every approach to a 3-planet opp pocket.
      6. value > 0 (skip end-of-game captures with zero remaining production).
    """
    comet_ids = comet_ids or set()
    my_planets = [p for p in world.planets_by_id.values()
                  if int(p.owner) == int(my_id)]
    targets = [p for p in world.planets_by_id.values()
               if int(p.owner) != int(my_id)
               and int(p.id) not in comet_ids]
    if not my_planets or not targets:
        return []

    opp_planets = [p for p in world.planets_by_id.values()
                   if int(p.owner) != int(my_id) and int(p.owner) >= 0]
    dominant_endgame = (
        len(opp_planets) > 0 and len(my_planets) >= 3 * len(opp_planets)
    )

    candidates: list[Candidate] = []
    for src in my_planets:
        ships_now = int(src.ships)
        prod = int(src.production)
        if ships_now < MIN_FLEET and prod <= 0:
            continue
        for tgt in targets:
            if int(tgt.id) == int(src.id):
                continue
            base_arrivals = list(model.ledger.get(int(tgt.id), []))
            for launch_tick in range(MAX_LAUNCH_TICK + 1):
                # Time-indexed budget: by tick `launch_tick`, src will have
                # ships_now + prod*launch_tick (production accrues each turn).
                budget_at_launch = ships_now + prod * int(launch_tick)
                if budget_at_launch < MIN_FLEET:
                    continue
                refined = _refine_ships(
                    src, tgt, launch_tick, omega, base_arrivals,
                )
                if refined is None:
                    continue
                ships, angle, eta, arrival_step = refined
                if ships > budget_at_launch:
                    continue
                fate = predict_fleet_fate(
                    src, tgt, angle, ships, world, wait_N=launch_tick,
                )
                if fate.outcome != "target":
                    continue
                if fate.hit_planet_id is None or int(fate.hit_planet_id) != int(tgt.id):
                    continue
                owner_at_arr, gar_at_arr = predict_garrison_at(
                    tgt, arrival_step, base_arrivals,
                )
                if int(owner_at_arr) == int(my_id):
                    continue  # already ours by arrival — reinforce, skip v1
                if float(ships) <= float(gar_at_arr):
                    continue  # not enough to capture
                if not dominant_endgame and not _target_holdable_after_capture(
                    src, tgt, ships, launch_tick, eta, world, model, my_id,
                ):
                    continue
                value = _capture_value(tgt, arrival_step)
                if value <= 0.0:
                    continue
                candidates.append(Candidate(
                    src_id=int(src.id),
                    tgt_id=int(tgt.id),
                    launch_tick=int(launch_tick),
                    angle=float(angle),
                    ships=int(ships),
                    eta=int(eta),
                    arrival_step=int(arrival_step),
                    value=float(value),
                ))
    return candidates
