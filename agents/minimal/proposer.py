"""Fire-now candidate enumeration: attack + reinforce.

Attack (per (owned src, enemy tgt) pair):
  eta    = lib.scoring.eta_proxy(src, tgt)             (conservative)
  ships  = max(MIN_FLEET, lib.scoring.s_needed(tgt, eta))
                                                       (prod-accrual aware)
  angle  = atan2(tgt.y - src.y, tgt.x - src.x)         (straight-line)

Reinforce (per (owned src, owned tgt) where tgt is under enemy threat):
  enemy_eta      = model.time_to_enemy_threat(tgt.id, me, world)
  enemy_inflight = Σ enemy fleet ships arriving by enemy_eta+1
  enemy_potential= max enemy planet ships (if no in-flight)
                   — covers the case where an enemy planet COULD launch
                   at us before we can react; missed-this-once root cause
                   for hold-then-fall losses (audit: 2026-05-23 trace).
  enemy_strength = max(enemy_inflight, enemy_potential)
  my_garrison    = tgt.ships + tgt.production * enemy_eta
  shortfall      = enemy_strength - my_garrison + 1
  ships          = max(MIN_FLEET, shortfall)              (skip if shortfall <= 0)

  Plus the ETA-arrival-in-time check: skip reinforces that arrive
  after the threat already takes the planet (my_eta >= enemy_eta).

Skipped if the source can't afford the required ship count.
"""

from __future__ import annotations

import math

from lib.scoring import eta_proxy, s_needed

MIN_FLEET = 2


def _attack_candidates(my_planets, enemy_planets) -> list[tuple]:
    cands: list[tuple] = []
    for src in my_planets:
        avail = int(src.ships)
        if avail < MIN_FLEET:
            continue
        for tgt in enemy_planets:
            eta = eta_proxy(src, tgt)
            ships = max(MIN_FLEET, int(s_needed(tgt, eta)))
            if avail < ships:
                continue
            angle = math.atan2(float(tgt.y) - float(src.y),
                               float(tgt.x) - float(src.x))
            cands.append((src, tgt, ships, angle))
    return cands


def _reinforce_size(tgt, model, me: int, world) -> tuple[int, int]:
    """Returns (ships_needed, enemy_eta). ships_needed=0 means skip."""
    enemy_eta = model.time_to_enemy_threat(int(tgt.id), me, world)
    if enemy_eta is None:
        return 0, 0
    enemy_inflight = sum(
        ships
        for (eta_arr, owner, ships) in model.ledger.get(int(tgt.id), [])
        if owner != me and eta_arr <= enemy_eta + 1
    )
    enemy_potential = 0.0
    if enemy_inflight <= 0:
        best = 0.0
        for p in world.planets_by_id.values():
            if int(p.owner) < 0 or int(p.owner) == me:
                continue
            if float(p.ships) > best:
                best = float(p.ships)
        enemy_potential = best
    enemy_strength = max(enemy_inflight, enemy_potential)
    my_garrison = float(tgt.ships) + float(tgt.production) * enemy_eta
    shortfall = enemy_strength - my_garrison + 1
    return max(0, int(math.ceil(shortfall))), enemy_eta


def _reinforce_candidates(my_planets, threatened_mine,
                          model, me: int, world) -> list[tuple]:
    cands: list[tuple] = []
    for tgt in threatened_mine:
        ships_needed, enemy_eta = _reinforce_size(tgt, model, me, world)
        if ships_needed < MIN_FLEET:
            continue
        for src in my_planets:
            if int(src.id) == int(tgt.id):
                continue
            if int(src.ships) < ships_needed:
                continue
            if eta_proxy(src, tgt) >= enemy_eta:
                continue  # reinforce arrives too late, planet falls first
            angle = math.atan2(float(tgt.y) - float(src.y),
                               float(tgt.x) - float(src.x))
            cands.append((src, tgt, ships_needed, angle))
    return cands


def propose(my_planets, enemy_planets, threatened_mine=None,
            model=None, me: int = 0, world=None) -> list[tuple]:
    cands = _attack_candidates(my_planets, enemy_planets)
    if threatened_mine and model is not None and world is not None:
        cands.extend(
            _reinforce_candidates(my_planets, threatened_mine, model, me, world)
        )
    return cands
