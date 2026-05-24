"""Layer R — epistemic risk pricing (reliability multiplier).

Every candidate has two distinct numbers we've been conflating:
  - nominal value  (if my plan plays out, this is worth X)
  - reliability    (probability my plan plays out as predicted)

This module computes reliability(action) in [0, 1] for any launch
candidate. Multiplying nominal_value × reliability is the risk-adjusted
value the MILP / proposer / chooser should actually rank by.

Failure modes captured (conditionally independent → multiplicative):
  - eta_reliability   — opp has `eta` ticks to disrupt while we fly.
  - wait_reliability  — opp has `wait_N` ticks to disrupt before fire.
  - landing_reliability — pessimistic effective-landing after opp counter.

Math notes:
  - exp(-tau/scale) is the max-entropy reliability function given only
    a timescale; doesn't assume a specific opp policy structure.
  - landing_reliability normalizes pessimistic-residual by nominal n,
    so a small fleet with negative pessimistic residual gets ~0, a
    fleet with positive surplus gets ~1.
"""
from __future__ import annotations

import math
import os
from typing import Optional

from lib.trajectory import fleet_speed

# Env vars (all defaults are no-op when reliability is gated off).
_TAU_ETA = float(os.environ.get("RELIABILITY_ETA_SCALE", "20"))
_TAU_WAIT = float(os.environ.get("RELIABILITY_WAIT_SCALE", "10"))
_OPP_REACT = float(os.environ.get("RELIABILITY_OPP_REACT", "0.4"))
_OPP_REACH_LAG = int(os.environ.get("RELIABILITY_OPP_REACH_LAG", "4"))
_MIN_OPP_SOURCE = int(os.environ.get("RELIABILITY_MIN_OPP_SOURCE", "3"))


def _opp_reachable_ships(tgt, arrival_step: int, world, my_id: int) -> int:
    """Strongest single enemy planet that can reach `tgt` within
    `arrival_step + OPP_REACH_LAG`. Returns 0 if none."""
    best = 0
    for p in world.planets_by_id.values():
        owner = int(p.owner)
        if owner == int(my_id) or owner < 0:
            continue
        ships_avail = int(p.ships)
        if ships_avail < _MIN_OPP_SOURCE:
            continue
        dx = float(p.x) - float(tgt.x)
        dy = float(p.y) - float(tgt.y)
        d = math.hypot(dx, dy)
        v = fleet_speed(ships_avail)
        if v <= 0:
            continue
        eta = int(math.ceil(d / v))
        if eta <= int(arrival_step) + _OPP_REACH_LAG:
            if ships_avail > best:
                best = ships_avail
    return best


def reliability(
    tgt,
    ships: int,
    eta: int,
    wait_N: int,
    world,
    my_id: int,
    *,
    tau_eta: Optional[float] = None,
    tau_wait: Optional[float] = None,
    opp_react: Optional[float] = None,
) -> float:
    """Probability the predicted-future state at our arrival matches
    reality enough that nominal value is realizable. Returns [0, 1].

    Parameters:
      tgt       — target planet (any object with .x, .y, .production).
      ships     — launch size.
      eta       — flight time (post-wait), in ticks.
      wait_N    — wait ticks before fire.
      world     — World; needs `planets_by_id` for opp reachability scan.
      my_id     — player id (so we know which planets are 'opp').
      tau_eta, tau_wait, opp_react — optional overrides of env-var defaults.
    """
    te = float(_TAU_ETA if tau_eta is None else tau_eta)
    tw = float(_TAU_WAIT if tau_wait is None else tau_wait)
    rxn = float(_OPP_REACT if opp_react is None else opp_react)

    eta_rel = math.exp(-float(eta) / max(1.0, te))
    wait_rel = math.exp(-float(wait_N) / max(1.0, tw))

    opp_reach = _opp_reachable_ships(tgt, int(eta) + int(wait_N), world, my_id)
    pessimistic = (
        float(ships)
        - float(tgt.production) * float(eta)
        - rxn * float(opp_reach)
    )
    landing_rel = max(0.0, pessimistic) / max(1.0, float(ships))

    return eta_rel * wait_rel * landing_rel


# Module-level toggle for the wiring sites. Same env var pattern as Layer Z.
RELIABILITY_PRICING_ENABLED = (
    os.environ.get("BASELINE_RELIABILITY_PRICING", "0") == "1"
)
