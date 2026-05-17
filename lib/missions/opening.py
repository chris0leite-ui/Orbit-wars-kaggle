"""Opening-landgrab Mission class — fires only at steps 0-5.

Top-10 fingerprint analysis (`knowledge-base/concepts/top-performer-strategies.md`)
finds that ladder leaders' median first-launch step = 4.1 while midpack
sits at 10.5. Six wasted opening turns forfeit ~6× per-planet
production ≈ 30 ships of board control. The Mission's job is to
guarantee that EVERY owned planet with ships > 8 launches in the
opening window, even if no other Mission class would.

Scoring uses **H7 front-loaded value**: `(remaining_steps)^1.5` instead
of linear, weighted toward nearby high-production neutrals. Distance
in the denominator only — no ship-cost weight, because we deliberately
WANT to drain home garrisons in the opening (top-10 mean
garrison-at-launch = 11, midpack 22; opening is the cheapest place
to align that gap).

Settle_plan arbitrates if the same source has both an opening and a
snipe candidate — opening's higher score will typically win in the
opening window because the front-loaded value blows past snipe's
linear discount.

Conditional, not flat-multiplier:
- Only fires when `world.step <= OPENING_WINDOW`.
- Skips sources with `ships <= 8` (don't strand a defender on a
  high-prod planet just to grab another).
- Skips targets in the comet set (comets at step 50, 150, ... — no
  comets exist during the opening window).
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.intent import World
from lib.mission import Mission
from lib.world_model import WorldModel

EPISODE_STEPS = 500
OPENING_WINDOW = 5            # inclusive; fires for steps 0..5
MIN_LAUNCH_GARRISON = 8       # don't strand a defender below this
FRONT_LOAD_EXPONENT = 1.5     # H7 from main's hypothesis board

# Mission Renaissance gate. Default 1 — Opening proposer is enabled in
# v7's pipeline per main's cb02fd9 (H11 wired into _build_incumbent_intents
# and shipped as v7_1). Set to 0 explicitly to ablate Opening for an
# A/B (e.g. the Mission Renaissance per-mission run found Opening
# borderline at 62.5% Wilson [48.4, 74.8] on top of PV, but main's
# v7_1 ships with it on, so the default mirrors that.)
USE_OPENING_MISSION = 1


def propose_opening_missions(world: World, model: WorldModel,
                             window: int = OPENING_WINDOW) -> list[Mission]:
    """One Mission per (our source with ships>8, neutral target) pair,
    fired only during the opening window. Score = production ×
    (remaining_steps)^1.5 / (distance + 1).

    `window` (default = `OPENING_WINDOW` = 5) lets callers extend the
    opening-fire window without monkey-patching the module constant.
    v23 uses window=15 to cover the empirical opening gap (v15's first
    15 turns under-launch by ~72% vs top-10; audit/2026-05-14-opening-
    atlas.json). Default unchanged so v7_1/geo/geo_recap callers see
    identical behaviour.
    """
    if not USE_OPENING_MISSION:
        return []
    if int(world.step) > window:
        return []
    my_planets = [
        p for p in world.planets_by_id.values()
        if p.owner == world.my_id and p.ships > MIN_LAUNCH_GARRISON
    ]
    if not my_planets:
        return []
    # Opening is neutral-only — enemies in the opening window are far
    # behind their own home cluster, distance dominates ROI, and
    # contested captures are rare. The snipe mission class still
    # proposes enemy targets if any are viable.
    neutrals = [
        p for p in world.planets_by_id.values()
        if p.owner == -1 and p.id not in world.comet_ids
    ]
    if not neutrals:
        return []

    step_now = int(world.step)
    missions: list[Mission] = []
    for src in my_planets:
        # Each source picks its single best opening shot; settle_plan
        # arbitrates further when multiple Missions converge on one
        # target.
        for t in neutrals:
            d = math.hypot(t.x - src.x, t.y - src.y)
            # Ships = target garrison + 1 (no production growth: neutrals
            # don't produce during flight).
            base_ships = max(1, int(t.ships) + 1)
            if base_ships >= src.ships:
                # Source can't cover this target without stranding itself.
                continue
            v = fleet_speed(base_ships)
            eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
            remaining = max(1, EPISODE_STEPS - step_now - eta)
            # Front-loaded value: opening captures earn 500-turn
            # production at full weight, not a linear decay.
            value = float(t.production) * (remaining ** FRONT_LOAD_EXPONENT)
            # Pure distance discount — no ship-cost in the denominator.
            # We WANT to send larger fleets earlier; that's the whole
            # point of the opening mission class.
            score = value / (d + 1.0)
            missions.append(Mission(
                mission_class="opening",
                src_id=src.id,
                target_id=t.id,
                ships=base_ships,
                score=score,
                eta=eta,
            ))
    return missions
