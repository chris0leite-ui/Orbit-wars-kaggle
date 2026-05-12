"""v3.5.1 ablation: recapture with tighter parameters.

The v3.5 recapture Mission was the closest single ablation to neutral
(53.1%, Wilson lo 36.4%). This variant tightens parameters per the
audit/2026-05-12-v3.5-stack-results.md debug hypothesis:

- RECAPTURE_WINDOW: 50 → 15 turns (only act on FRESH losses)
- Distance filter: require any owned source within d <= 25
  (don't ferry from across the board)
- RECAPTURE_BONUS_PEAK: 1.5 → 1.8 (more decisive when conditions met)
- RECENTLY_LOST_GARRISON_MAX: 50 → 25 (skip if enemy already fortified)
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.intent import World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.mission import Mission
from lib.missions.recapture import _RecaptureState
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.world_model import WorldModel

EPISODE_STEPS = 500
TIGHT_WINDOW = 15
RECAPTURE_BONUS_PEAK = 1.8
FORTIFIED_MAX = 25
MAX_SOURCE_DISTANCE = 25.0


# Local state instance — independent from lib.missions.recapture._STATE so
# we don't share cross-test state across module reloads.
_STATE = _RecaptureState()


def propose_recapture_tight(world: World, model: WorldModel) -> list[Mission]:
    """Tightened recapture: short window, distance-bounded, decisive bonus."""
    _STATE.update(world)
    if not _STATE.lost_at:
        return []
    step_now = int(world.step)
    my_planets = [
        p for p in world.planets_by_id.values() if p.owner == world.my_id
    ]
    if not my_planets:
        return []

    cutoff = step_now - TIGHT_WINDOW
    fresh_losses = {
        pid: step_lost for pid, step_lost in _STATE.lost_at.items()
        if step_lost >= cutoff
    }
    if not fresh_losses:
        return []

    missions: list[Mission] = []
    for lost_pid, step_lost in fresh_losses.items():
        t = world.planets_by_id.get(lost_pid)
        if t is None or t.owner == world.my_id:
            continue
        if t.ships > FORTIFIED_MAX:
            continue
        # Find ANY owned source within MAX_SOURCE_DISTANCE.
        in_range = []
        for src in my_planets:
            d = math.hypot(t.x - src.x, t.y - src.y)
            if d <= MAX_SOURCE_DISTANCE:
                in_range.append((src, d))
        if not in_range:
            continue
        elapsed = step_now - step_lost
        urgency = max(0.0, 1.0 - elapsed / TIGHT_WINDOW)
        bonus = 1.0 + (RECAPTURE_BONUS_PEAK - 1.0) * urgency
        for src, d in in_range:
            base_ships = max(1, int(t.ships) + 1)
            if base_ships >= src.ships:
                continue
            v = fleet_speed(base_ships)
            eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
            pred_owner = model.owner_at(t.id, eta)
            if pred_owner == world.my_id:
                continue
            time_to_hold = max(1, EPISODE_STEPS - step_now - eta)
            value = t.production * time_to_hold
            score = bonus * value / (0.5 * base_ships + d + 1.0)
            missions.append(Mission(
                mission_class="recapture",
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
        propose_snipe_missions(world, model)
        + propose_reinforce_missions(world, model)
        + propose_recapture_tight(world, model)
    )
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)
