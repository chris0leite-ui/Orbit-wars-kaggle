"""Fire-now capture-size candidate enumeration.

For each (owned source, enemy target) pair, emit up to TWO candidates:
  eta       = lib.scoring.eta_proxy(src, tgt)             (conservative)
  capture   = max(MIN_FLEET, lib.scoring.s_needed(tgt, eta))
                                                          (prod-accrual aware)
  overcommit= min(avail, 2 * capture)
  angle     = atan2(tgt.y - src.y, tgt.x - src.x)         (straight-line)

s_needed = tgt.ships + tgt.production * eta + 1 — the same formula the
mechanism layer uses to size fleets, so we don't under-send and bounce
on a garrison that grew during transit.

Two ship-count options let the chooser pick between a minimum-cost
snipe and an overcommit that survives the opp's follow-up reinforcement
(critical against high-spam opponents like `nearest` and `roi`). The
chooser already does forward-search valuation, so we don't need to
prune here — we just expose the option.

Skipped if the source can't afford even the capture size.
"""

from __future__ import annotations

import math

from lib.scoring import eta_proxy, s_needed

MIN_FLEET = 2


def propose(my_planets, enemy_planets) -> list[tuple]:
    cands: list[tuple] = []
    for src in my_planets:
        avail = int(src.ships)
        if avail < MIN_FLEET:
            continue
        for tgt in enemy_planets:
            eta = eta_proxy(src, tgt)
            capture = max(MIN_FLEET, int(s_needed(tgt, eta)))
            if avail < capture:
                continue
            angle = math.atan2(float(tgt.y) - float(src.y),
                               float(tgt.x) - float(src.x))
            cands.append((src, tgt, capture, angle))
            overcommit = min(avail, 2 * capture)
            if overcommit > capture:
                cands.append((src, tgt, overcommit, angle))
    return cands
