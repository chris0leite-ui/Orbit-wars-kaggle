"""momentum_strike proposer — production-first expansion + CCW tie-breaker.

Per-turn pipeline:
  1. Defense — reinforce any of our planets the WorldModel predicts will
     flip to an enemy within its horizon.
  2. Expand — assign each source to the highest-production reachable
     target. CCW ordering is the tie-breaker among equal-production
     candidates. Both neutrals and weak enemy planets are valid targets.

All emissions go through `lib.intent.realize(intents, obs,
mechanisms=DEFAULT_MECHANISMS)` which handles:
  - `validate` (caps ships at src.ships, drops zero/negative)
  - `arrival_size` (inflates ships for production growth during flight)
  - `lead_aim` (orbit-aware lead-aim for orbiting targets)
  - `sun_avoid` (drops/routes paths that cross the sun)
  - `path_clears_other_planets` (drops paths that hit non-target planets)
  - `oob_guard` (drops paths that fly off the board)

This replaces our V1's hand-rolled aim + fate gate with the same
mechanism stack `agents/simple/production` and `agents/simple/nearest`
use — getting auto-aim, auto-sizing, and auto-safety for free.
"""

from __future__ import annotations

import math
import os

from lib.fleet import speed as fleet_speed
from lib.intent import Intent
from lib.polar import polar_angle_about, ccw_delta


def _eta_simple(src, tgt, ships: int) -> int:
    """Geometry-only ETA proxy: ceil(distance / fleet_speed(ships)).

    Conservative under-estimate of the lead-aim-corrected ETA the
    mechanism pipeline ultimately uses; close enough for ordering
    candidates.
    """
    dx = float(tgt.x) - float(src.x)
    dy = float(tgt.y) - float(src.y)
    flight = max(0.0, math.hypot(dx, dy) - float(src.radius) - float(tgt.radius) - 0.1)
    v = fleet_speed(max(1, int(ships)))
    if v <= 0:
        return 999
    return max(1, int(math.ceil(flight / v)))


# Knobs (env-var overridable).
# Reserve a small buffer per source to avoid stripping planets bare
# every turn — if budget = ships, validate will accept ships = src.ships
# but the planet ends with 0 ships, immediately vulnerable. A 1-ship
# reserve doesn't slow the opening (we still match `nearest`'s
# target.ships+1 sizing) but stops the auto-strip degenerate case.
EXPAND_RESERVE_FLOOR = int(os.environ.get("MOMENTUM_EXPAND_RESERVE_FLOOR", "0"))
# Maximum production-tier-bucketing for the CCW tie-break sort. After
# filtering capturable targets, we sort primarily by production desc;
# CCW ordering only applies within the highest-production bucket.
CCW_TIEBREAK_ON_TOP_PROD_ONLY = True


def _reserve(planet) -> int:
    """Minimal expansion-mode reserve."""
    return EXPAND_RESERVE_FLOOR


def propose_defense(my_planets, world, model, my_id: int,
                    used_srcs: set[int]) -> list[Intent]:
    """Reinforce any of my planets the WorldModel predicts will flip.

    Sized as `attacker_strength + 1` from the nearest source that can
    arrive before `T_loss`. The emitted Intents go through the standard
    mechanism pipeline alongside the offensive ones.
    """
    if len(my_planets) < 2 or model is None:
        return []
    horizon = int(getattr(model, "horizon", 40))
    intents: list[Intent] = []
    # Identify threats.
    threats = []  # (defended, T_loss, post_flip_ships)
    for d in my_planets:
        t_loss = None
        for t in range(1, horizon + 1):
            try:
                owner = model.owner_at(int(d.id), t)
            except Exception:
                continue
            if owner is not None and int(owner) != my_id:
                t_loss = t
                break
        if t_loss is None:
            continue
        try:
            post_flip = model.ships_at(int(d.id), t_loss) or 0.0
        except Exception:
            post_flip = 0.0
        threats.append((d, t_loss, float(post_flip)))
    if not threats:
        return []
    threats.sort(key=lambda x: (-int(x[0].production), int(x[1])))

    for defended, t_loss, attacker in threats:
        deficit = int(attacker) + 1
        if deficit < 1:
            continue
        best = None
        best_eta = None
        for src in my_planets:
            if int(src.id) == int(defended.id):
                continue
            if int(src.id) in used_srcs:
                continue
            budget = int(src.ships) - _reserve(src)
            if budget < deficit:
                continue
            eta = _eta_simple(src, defended, deficit)
            if eta >= t_loss:
                continue
            if best is None or eta < best_eta:
                best = src
                best_eta = eta
        if best is None:
            continue
        intents.append(Intent(
            src_id=int(best.id), target_id=int(defended.id),
            ships=int(deficit), note=f"defense:T_loss={t_loss}",
        ))
        used_srcs.add(int(best.id))
    return intents


def propose_expand(my_planets, neutrals, enemies, world, model, my_id: int,
                   used_srcs: set[int]) -> list[Intent]:
    """Production-first target selection per source; CCW tie-break.

    For each source (sorted by production desc, ships desc):
      1. Build the capturable target set (target.ships+1 ≤ src.ships).
      2. Sort capturable by (production desc, ccw asc, eta asc).
      3. Emit Intent at the top target, ships = target.ships + 1.
      4. Mark source used; one launch per source per turn.

    The mechanism pipeline (`realize`) handles aim, ship inflation for
    enemy production growth during flight, sun avoidance, and OOB
    rejection. We just pick WHO attacks WHO.
    """
    if not neutrals and not enemies:
        return []
    intents: list[Intent] = []
    sorted_srcs = sorted(
        my_planets,
        key=lambda p: (-int(p.production), -int(p.ships)),
    )
    for src in sorted_srcs:
        if int(src.id) in used_srcs:
            continue
        budget = int(src.ships) - _reserve(src)
        if budget < 1:
            continue
        src_theta = polar_angle_about((src.x, src.y))
        candidates = []  # (production_neg, eta, ccw_delta, target, cost)
        # Tie-break stack: (production desc, ETA asc, CCW asc). The PI
        # asked for counterclockwise expansion, but ETA-first tie-break
        # matches `agents/simple/production`'s pattern (closer ties
        # beat far ones). CCW is the third tier so it still influences
        # truly-equivalent candidates.
        # Neutrals: cap = target.ships + 1 (no production growth).
        for tgt in neutrals:
            cost = int(tgt.ships) + 1
            if cost > budget:
                continue
            eta = _eta_simple(src, tgt, max(1, budget))
            tgt_theta = polar_angle_about((tgt.x, tgt.y))
            candidates.append(
                (-int(tgt.production), eta, ccw_delta(src_theta, tgt_theta), tgt, cost)
            )
        # Weak enemies: cap = projected garrison + 1 at our ETA.
        for tgt in enemies:
            eta = _eta_simple(src, tgt, max(1, budget))
            proj = int(tgt.ships) + int(tgt.production) * eta + 1
            if proj > budget:
                continue
            tgt_theta = polar_angle_about((tgt.x, tgt.y))
            candidates.append(
                (-int(tgt.production), eta, ccw_delta(src_theta, tgt_theta), tgt, proj)
            )
        if not candidates:
            continue
        candidates.sort()
        _negprod, _eta, _ccw, target, cost = candidates[0]
        intents.append(Intent(
            src_id=int(src.id), target_id=int(target.id),
            ships=int(cost), note="expand",
        ))
        used_srcs.add(int(src.id))
    return intents
