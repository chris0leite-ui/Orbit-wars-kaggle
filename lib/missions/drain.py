"""Source-drain Mission class — flush surplus garrisons from safe planets.

Top-10 fingerprint analysis finds ladder leaders' mean garrison-at-launch
is 11 ships; midpack sits at 22. The gap is "what stays on owned
planets vs what's in motion." TrueSkill rewards win/loss; ships sitting
at home only produce passive value, while ships in flight capture
territory.

This Mission class identifies planets where:
  - `src.ships > MIN_DRAIN_SHIPS` (we genuinely have surplus)
  - `WorldModel.incoming_enemy_eta(src.id) is None` (no inbound enemy)
    OR the enemy ETA is comfortably greater than the drain ETA
  - There's a non-owned target reachable that won't strand us

It proposes a Mission that sends ~`src.ships - RESERVE_KEEP` ships
(leaving a small defensive garrison ≈ 8) toward the best non-owned
neutral or enemy planet.

Critically: drain does NOT use a flat multiplier — its score uses the
same cost-aware denominator as snipe but with a moderate `DRAIN_BONUS`
that fires ONLY when the planet has been verified safe. This is a
conditional, not a flat bias.
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.intent import World
from lib.mission import Mission
from lib.world_model import WorldModel

EPISODE_STEPS = 500
MIN_DRAIN_SHIPS = 30           # only drain when there's genuine surplus
RESERVE_KEEP = 8               # always leave a defender behind
SAFE_ETA_BUFFER = 5            # require enemy ETA > our ETA + this
DRAIN_BONUS = 1.10             # mild bonus for using SAFE surplus

# Mission Renaissance gate. Default 0 = disabled. A/B candidate: 1.
USE_DRAIN_MISSION = 0

# Mirror snipe.py: when True, drain's affordability check uses
# pred_ships at our ETA, not current garrison. Avoids skipping targets
# that an inbound enemy will reduce below current ships by our arrival.
USE_PRED_SHIPS_FOR_SIZING = True


def propose_drain_missions(world: World, model: WorldModel) -> list[Mission]:
    """One drain Mission per (safe high-garrison source, best target) pair.

    Skips sources that have any inbound enemy within a short window;
    skips targets that the source can't afford after RESERVE_KEEP.
    """
    if not USE_DRAIN_MISSION:
        return []
    my_planets = [
        p for p in world.planets_by_id.values()
        if p.owner == world.my_id and p.ships > MIN_DRAIN_SHIPS
    ]
    if not my_planets:
        return []
    targets = [
        p for p in world.planets_by_id.values()
        if p.owner != world.my_id
    ]
    if not targets:
        return []

    step_now = int(world.step)
    missions: list[Mission] = []
    for src in my_planets:
        # Drain ships = surplus over the reserve.
        drain_ships = int(src.ships) - RESERVE_KEEP
        if drain_ships <= 0:
            continue
        # Safety gate: refuse to drain if a non-trivial enemy fleet is
        # inbound to this source within (our typical attack ETA + buffer).
        enemy_eta = model.incoming_enemy_eta(src.id, world.my_id)
        for t in targets:
            d = math.hypot(t.x - src.x, t.y - src.y)
            # We want to send `drain_ships` — fleet speed depends on that.
            v = fleet_speed(drain_ships)
            our_eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
            if enemy_eta is not None and enemy_eta <= our_eta + SAFE_ETA_BUFFER:
                # Source has imminent enemy arrival — don't strand it.
                continue
            # Predicted ownership/garrison at arrival — skip if already
            # ours; otherwise use pred_ships (not current `t.ships`) for
            # the affordability gate so an inbound enemy fleet shrinking
            # the target before our arrival doesn't bar capturing it.
            pred_owner = model.owner_at(t.id, our_eta)
            if pred_owner == world.my_id:
                continue
            base_capture = max(1, int(t.ships) + 1)
            if USE_PRED_SHIPS_FOR_SIZING:
                # DOWNSIZE-only: mirror snipe.py. Use pred_ships only when
                # it's LOWER than current (inbound enemy fleet reducing the
                # target's garrison). Production growth on owned targets is
                # arrival_size's job; double-counting bloats fleet sizes.
                pred_ships = model.ships_at(t.id, our_eta)
                if pred_ships is not None and pred_ships < float(t.ships):
                    base_capture = max(1, int(math.ceil(float(pred_ships))) + 1)
            if drain_ships < base_capture:
                continue
            # Score uses the standard cost-aware ROI shape (rebalanced
            # denominator from wave 1b), bumped by DRAIN_BONUS because
            # this is verified-safe surplus.
            is_comet = t.id in world.comet_ids
            if is_comet:
                from lib.world_model import comet_remaining_lifetime
                rem = comet_remaining_lifetime(t.id, world)
                time_to_hold = max(0, (rem or 0) - our_eta)
            else:
                time_to_hold = max(1, EPISODE_STEPS - step_now - our_eta)
            value = float(t.production) * time_to_hold
            score = DRAIN_BONUS * value / (0.5 * drain_ships + d + 1.0)
            missions.append(Mission(
                mission_class="drain",
                src_id=src.id,
                target_id=t.id,
                ships=drain_ships,
                score=score,
                eta=our_eta,
            ))
    return missions
