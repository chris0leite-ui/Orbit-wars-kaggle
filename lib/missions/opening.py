"""Opening-landgrab Mission class — fires only at steps 0-5.

Top-10 fingerprint analysis (`knowledge-base/concepts/top-performer-strategies.md`)
finds that ladder leaders' median first-launch step = 4.1 while midpack
sits at 10.5. Six wasted opening turns forfeit ~6× per-planet
production ≈ 30 ships of board control.

**2026-05-12 sizing fix (bowwowforeach archetype).** The prior
implementation sent `max(1, t.ships + 1)` ships — for the typical
step-0 neutral with 0 ships, that meant a **1-ship launch**, identical
to v3_snipe. Ablation showed `opening_only` regressed (40.6% Wilson
lo, audit/2026-05-12-v3.5-stack-results.md); the firing trigger
worked but the launch itself did nothing differentiating.

Top-10 fingerprint: garrison-at-launch 7.7 (bowwowforeach #1) vs
midpack 22. Translating: send ~all-but-2 of a 10-ship home source.
The new sizing is `max(target_min, src.ships - OPENING_RESERVE)` —
fleet large enough to capture AND large enough to drain the source.

Scoring uses **H7 front-loaded value**: `(remaining_steps)^1.5` instead
of linear, weighted toward nearby high-production neutrals. Distance
in the denominator only — no ship-cost weight, because we deliberately
WANT to drain home garrisons in the opening.

Settle_plan arbitrates if the same source has both an opening and a
snipe candidate — opening's higher score will typically win in the
opening window because the front-loaded value blows past snipe's
linear discount.

Conditional, not flat-multiplier:
- Only fires when `world.step <= OPENING_WINDOW`.
- Skips sources with `ships <= MIN_LAUNCH_GARRISON` (8 is the home-
  planet starting count minus 2-ship reserve).
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
MIN_LAUNCH_GARRISON = 8       # source must hold > MIN to launch
OPENING_RESERVE = 2           # ships left at source post-launch
FRONT_LOAD_EXPONENT = 1.5     # H7 from main's hypothesis board


def propose_opening_missions(
    world: World,
    model: WorldModel,
    *,
    window: int = OPENING_WINDOW,
    min_garrison: int = MIN_LAUNCH_GARRISON,
    reserve: int = OPENING_RESERVE,
    allow_enemies: bool = False,
) -> list[Mission]:
    """One Mission per (our source with ships>min_garrison, eligible
    target) pair, fired only during steps 0..window. Sized to drain
    the source down to `reserve` ships (bowwowforeach archetype).
    Score = production × (remaining_steps)^1.5 / (distance + 1).

    Variants:
    - **A (default, FAILED 2026-05-12):** window=5, min_garrison=8,
      reserve=2, allow_enemies=False. Drains step-0 home from 10→2.
    - **B (timing-matched bowwow):** window=5, min_garrison=14,
      reserve=7, allow_enemies=False. Waits 2-3 steps for production,
      then sends a built-up fleet, leaving ~bowwow's measured 7.7-ship
      garrison-at-launch.
    - **C (enemy-target allowed):** window=5, min_garrison=8,
      reserve=2, allow_enemies=True. Bowwow picks 42% enemy targets;
      this variant lets the opener cover enemy-home raids.
    """
    if int(world.step) > window:
        return []
    my_planets = [
        p for p in world.planets_by_id.values()
        if p.owner == world.my_id and p.ships > min_garrison
    ]
    if not my_planets:
        return []
    # Eligible targets: non-comet planets that aren't ours. Neutral-only
    # by default; allow_enemies=True opens up enemy-home raids.
    targets = [
        p for p in world.planets_by_id.values()
        if p.id not in world.comet_ids
        and p.owner != world.my_id
        and (allow_enemies or p.owner == -1)
    ]
    if not targets:
        return []

    step_now = int(world.step)
    missions: list[Mission] = []
    for src in my_planets:
        # Each source picks its single best opening shot; settle_plan
        # arbitrates further when multiple Missions converge on one
        # target.
        for t in targets:
            d = math.hypot(t.x - src.x, t.y - src.y)
            target_min = max(1, int(t.ships) + 1)
            if target_min > src.ships:
                # Source can't capture this target at all — skip.
                continue
            # Bowwowforeach-style empty-the-source sizing: send the
            # MAX of (capture-cost, src - reserve). For a 10-ship home
            # targeting a 0-ship neutral with reserve=2 that's max(1, 8) = 8.
            base_ships = min(src.ships, max(target_min, src.ships - reserve))
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
