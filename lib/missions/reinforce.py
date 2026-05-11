"""Reinforce mission builder — defend our planets predicted to fall.

For each of OUR planets `D`, scan the WorldModel timeline for the first
step `T_loss` at which ownership flips to an enemy (i.e. an incoming
enemy fleet captures it). If `T_loss` is within the planner's horizon,
build one Mission per (source `S`, threatened `D`) pair where `S` is
another planet we own and a fleet from `S` can arrive BEFORE `T_loss`.

This is the first non-offensive mission class — the framework's payoff
for the Block E scaffolding. v3_snipe used to ignore inbound enemy
fleets entirely; per `docs/strategies/simple-roi.md` line 130 ("No
defence"), garrisons stayed put as enemies captured our planets.

Ship sizing: we send `pred_enemy_arriving + 1` ships so the combat
resolver leaves us with the surplus (≥1 ship) and ownership.

Score: same cost-aware ROI shape as snipe, applied to the DEFENDED
planet's production over its remaining game lifetime:

    value = D.production × max(1, 500 - step - eta)
    score = value / (ships_sent + distance + 1)

Note we use D's full lifetime, not just `T_loss - now` — defending
keeps the planet's production stream alive for the rest of the game.
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.geometry import sym_hypot
from lib.intent import World
from lib.mission import Mission
from lib.world_model import WorldModel

EPISODE_STEPS = 500


def propose_reinforce_missions(
    world: World, model: WorldModel,
) -> list[Mission]:
    """Build reinforce candidates for every (our source, our threatened
    planet) pair where we can arrive before the predicted loss step."""
    if not world.planets_by_id:
        return []
    my_planets = [
        p for p in world.planets_by_id.values() if p.owner == world.my_id
    ]
    if len(my_planets) < 2:
        # Need at least one source AND one target — same planet can't
        # reinforce itself (it'd be sending ships to itself, no effect).
        return []

    horizon = model.horizon
    step_now = int(world.step)

    # Identify under-threat planets and their predicted loss step.
    threatened: list[tuple] = []  # (planet, T_loss, enemy_ships_arriving)
    for d in my_planets:
        # Scan timeline for first step where ownership flips off us.
        t_loss: int | None = None
        for t in range(1, horizon + 1):
            owner = model.owner_at(d.id, t)
            if owner is not None and owner != world.my_id:
                t_loss = t
                break
        if t_loss is None:
            continue
        # Approx defenders needed: predicted ships of the enemy AT T_loss
        # (the post-flip garrison reflects the surviving attacker count).
        post_flip_ships = model.ships_at(d.id, t_loss) or 0.0
        threatened.append((d, t_loss, post_flip_ships))

    if not threatened:
        return []

    missions: list[Mission] = []
    for d, t_loss, attacker_strength in threatened:
        for s in my_planets:
            if s.id == d.id:
                continue
            # Fleet size = enough to repel the attacker plus a 1-ship buffer.
            # +1 is the same convention snipe uses for capture overhead.
            cost = max(1, int(attacker_strength) + 1)
            v = fleet_speed(cost)
            d_dist = sym_hypot(d.x - s.x, d.y - s.y)
            eta = int(math.ceil(d_dist / max(v, 1e-6))) if v > 0 else horizon + 1
            if eta >= t_loss:
                # We can't get there in time — the planet falls before
                # we arrive. Skip; a recapture mission (v3.2) would pick
                # this up instead.
                continue
            time_to_hold = max(1, EPISODE_STEPS - step_now - eta)
            value = d.production * time_to_hold
            score = value / (cost + d_dist + 1.0)
            missions.append(Mission(
                mission_class="reinforce",
                src_id=s.id,
                target_id=d.id,
                ships=cost,
                score=score,
                eta=eta,
            ))
    return missions
