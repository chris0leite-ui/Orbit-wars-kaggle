"""v2 — roi + arrival-ledger awareness.

Strategy upgrade over v1.2/roi:
- Per-source pick the highest-ROI target *that won't already be ours
  at arrival* (via `WorldModel.owner_at` lookup).
- Bump intent.ships to cover the predicted garrison + reinforcements
  at arrival (not just current ships + 1).
- If the best target is dropped (would be redundant), re-pick the next
  best — sources don't end the turn idle.

Why this beats stand-alone `arrival_ledger` mechanism:
- The mechanism layer can't re-pick after a drop (it's a pure
  list-filter). Strategy-level integration with WorldModel lets each
  source actually USE its turn.
- Targets that will be ours at arrival no longer attract redundant
  fleets — those ships are deployed elsewhere.
- Enemy-defended targets get correctly-sized fleets up front, so the
  capture doesn't fail at the last 5 ships.

Pipeline: still emits Intent(src, target, ships) for the mechanism layer
(validate / arrival_size / lead_aim_v2 / sun_avoid / path_clears_other_planets
/ oob_guard) to finalise. Mechanism stack unchanged.
"""

from __future__ import annotations

import math
import random

from lib.fleet import speed as fleet_speed
from lib.intent import Intent, World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.world_model import WorldModel, comet_remaining_lifetime


def propose_intents(obs) -> list[Intent]:
    """Per-source: rank targets by ROI; pick best that isn't redundant.

    Uses `WorldModel.owner_at` / `ships_at` to predict the (owner, ships)
    of each candidate target at our fleet's arrival step. Re-picks if
    the best target is "already ours with surplus."
    """
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []

    my_planets = [p for p in world.planets_by_id.values() if p.owner == world.my_id]
    if not my_planets:
        return []
    targets = [p for p in world.planets_by_id.values() if p.id not in {q.id for q in my_planets}]
    # Note: filtering by `id not in {my_ids}` keeps friendly planets out;
    # reinforce missions are a v3 concern.
    targets = [p for p in world.planets_by_id.values() if p.owner != world.my_id]
    if not targets:
        return []

    wm = WorldModel.from_world(world)
    rng = random.Random(world.step ^ (world.my_id + 1) * 1009)
    intents: list[Intent] = []

    for src in my_planets:
        scored: list[tuple] = []
        for t in targets:
            d = math.hypot(t.x - src.x, t.y - src.y)
            base_ships = max(1, t.ships + 1)
            v = fleet_speed(base_ships)
            eta = int(math.ceil(d / max(v, 1e-6)))
            pred_owner = wm.owner_at(t.id, eta)
            pred_ships = wm.ships_at(t.id, eta) or 0.0
            # Drop targets that will already be ours with surplus garrison
            # (don't-double-commit). Don't apply any affordability filter
            # or ship-bumping here — the mechanism layer handles that.
            # An earlier v2 attempt bumped ships per WorldModel-predicted
            # defense and filtered unaffordable bumped-targets; that made
            # the agent prefer LOW-ROI affordable targets over HIGH-ROI
            # ones it could've afforded with arrival_size's monotonic bump.
            # 0/64 WR (audit/tournaments/20260510T215806Z.json). Roll back.
            if pred_owner == world.my_id and pred_ships >= base_ships:
                continue
            # Cost-aware ROI (2026-05-11 fix per docs/strategies/simple-roi.md
            # "Where ROI can lose"): value = production × time-we-hold-it,
            # cost = ships-we-send + distance (additive, not pure value/cost
            # which over-corrects toward 1-ship targets).
            # Comet lifetime correction: a comet leaving the board in 5
            # turns scores 0 if our eta > 5 (don't chase departing comets).
            if t.id in world.comet_ids:
                rem = comet_remaining_lifetime(t.id, world)
                time_to_hold = max(0, (rem or 0) - eta)
            else:
                time_to_hold = max(1, 500 - world.step - eta)
            value = t.production * time_to_hold
            roi = value / (base_ships + d + 1.0)
            scored.append(((-roi, d), rng.random(), t, base_ships))

        if not scored:
            continue
        scored.sort(key=lambda e: (e[0], e[1]))
        _, _, target, ships = scored[0]
        intents.append(Intent(src_id=src.id, target_id=target.id, ships=ships))
    return intents


def agent(obs):
    return realize(propose_intents(obs), obs, mechanisms=DEFAULT_MECHANISMS)
