"""v3.5.1 ablation: aggressive snipe ship sizing.

Single surgical change vs v3_snipe: snipe sends a larger fraction of
source garrison per launch (matches top-10 mean fleet 38 vs midpack 29
from `knowledge-base/concepts/top-performer-strategies.md`).

Ships sent for a snipe Mission:
- Old: `max(1, t.ships + 1)`              (minimum viable; arrival_size bumps if needed)
- New: `max(target_min, min(src.ships * 0.7, src.ships - 5))` IFF src.ships > 12
       else fall back to old formula (don't strand small sources)

The +1-cushion to leave 5 ships at source preserves a minimal defender
(below this, validate.mechanism may drop the intent if local garrison
isn't enough to cover the bumped size).
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.intent import World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.mission import Mission
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import (
    COMET_BONUS,
    EPISODE_STEPS,
    LEADER_MULTIPLIER,
    NEUTRAL_BONUS,
    _leader_pid,
)
from lib.planner import settle_plan
from lib.world_model import WorldModel, comet_remaining_lifetime

SHIP_FRACTION = 0.7    # fraction of src.ships sent
MIN_RESERVE = 5        # always leave this many at source (if affordable)
MIN_GARRISON_FOR_AGGRESSIVE = 12  # below this, use old minimum-viable formula


def propose_snipe_aggressive(world: World, model: WorldModel) -> list[Mission]:
    if not world.planets_by_id:
        return []
    my_planets = [p for p in world.planets_by_id.values() if p.owner == world.my_id]
    if not my_planets:
        return []
    targets = [p for p in world.planets_by_id.values() if p.owner != world.my_id]
    if not targets:
        return []
    step_now = int(world.step)
    leader_pid, our_rank = _leader_pid(world)
    spoiler_on = leader_pid is not None and our_rank is not None and our_rank >= 2

    missions: list[Mission] = []
    for src in my_planets:
        for t in targets:
            d = math.hypot(t.x - src.x, t.y - src.y)
            # Target minimum: the same defense-aware estimate the static
            # arrival_size uses. snipe.py itself uses just t.ships + 1
            # and lets arrival_size bump later, but for sizing decisions
            # we want a single consistent number.
            target_min = max(1, int(t.ships) + 1)
            if src.ships > MIN_GARRISON_FOR_AGGRESSIVE:
                fraction_size = max(1, int(src.ships * SHIP_FRACTION))
                cap = max(1, int(src.ships) - MIN_RESERVE)
                base_ships = max(target_min, min(fraction_size, cap))
            else:
                base_ships = target_min
            v = fleet_speed(base_ships)
            eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
            pred_owner = model.owner_at(t.id, eta)
            pred_ships = model.ships_at(t.id, eta) or 0.0
            if pred_owner == world.my_id and pred_ships >= base_ships:
                continue
            is_comet = t.id in world.comet_ids
            if is_comet:
                rem = comet_remaining_lifetime(t.id, world)
                time_to_hold = max(0, (rem or 0) - eta)
            else:
                time_to_hold = max(1, EPISODE_STEPS - step_now - eta)
            value = t.production * time_to_hold
            priority = 1.0
            if t.owner == -1:
                priority *= COMET_BONUS if is_comet else NEUTRAL_BONUS
            if spoiler_on and t.owner == leader_pid:
                priority *= LEADER_MULTIPLIER
            score = priority * value / (0.5 * base_ships + d + 1.0)
            missions.append(Mission(
                mission_class="snipe",
                src_id=src.id,
                target_id=t.id,
                ships=base_ships,
                score=score,
                eta=eta,
            ))
    return missions


def agent(obs):
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    model = WorldModel.from_world(world)
    missions = (
        propose_snipe_aggressive(world, model)
        + propose_reinforce_missions(world, model)
    )
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)
