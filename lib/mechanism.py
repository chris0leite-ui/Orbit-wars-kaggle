"""The "obvious rules" mechanism layer.

Every mechanism is `Callable[[list[Intent], World], list[Intent]]`. The
canonical pipeline order is below; `DEFAULT_MECHANISMS` matches it.

Pipeline order (rationale comments next to `DEFAULT_MECHANISMS`):

    validate   — drop unsafe intents up front so nothing downstream
                 computes against bad data.
    arrival_size  (3.5.B) — recompute `ships` for enemy targets accounting
                 for production growth during fleet flight. MUST run before
                 `lead_aim` because lead time depends on fleet size, which
                 depends on ship count.
    lead_aim   — populate `aim_angle` with an orbit-aware lead for orbiting
                 non-comet targets; comets and statics fall through to
                 current-position atan2.
    comet_aim  (3.5.C) — populate `aim_angle` for comet targets via
                 path-indexed prediction. Runs AFTER `lead_aim` so lead_aim
                 can no-op on comet targets and let comet_aim own them.
    sun_avoid  (3.5.D) — if direct path crosses the sun, route via waypoint
                 or drop the intent. Runs LAST because it needs the angle
                 set by lead_aim/comet_aim.

Step 3.5.A only includes `validate` and `lead_aim` — the parity-preserving
subset. `arrival_size`, `comet_aim`, `sun_avoid` are added by 3.5.B/C/D.
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.intent import Intent, World
from lib.orbit import is_orbiting, predict_relative


# ---------------------------------------------------------------------------
# validate — drop intents that violate ownership / garrison constraints
# ---------------------------------------------------------------------------


def validate(intents: list[Intent], world: World) -> list[Intent]:
    """Pass through intents whose src is owned and garrison covers ships.

    Drops intents where:
    - src planet is unknown (env hasn't surfaced it),
    - src is not owned by us (strategy bug, defensive),
    - target is the source itself (self-target),
    - ships <= 0 or ships > current src.ships.

    Note: this enforces the *current* garrison sufficiency. If `arrival_size`
    later bumps ships above the garrison, that intent gets dropped at the
    final emission step in `realize()` (ships <= 0 check is the safety net).
    To reject early-bumped-too-large intents BEFORE lead_aim wastes work,
    rerun validate as the final stage (added when arrival_size lands).
    """
    out: list[Intent] = []
    for intent in intents:
        src = world.planets_by_id.get(intent.src_id)
        if src is None:
            continue
        if src.owner != world.my_id:
            continue
        if intent.target_id == intent.src_id:
            continue
        if intent.ships <= 0 or intent.ships > src.ships:
            continue
        out.append(intent)
    return out


# ---------------------------------------------------------------------------
# arrival_size — production-aware fleet sizing for enemy targets
# ---------------------------------------------------------------------------


def arrival_size(intents: list[Intent], world: World) -> list[Intent]:
    """Bump `ships` so an enemy-owned target's expected garrison at arrival is
    captured (covered by `target.ships + production * eta + 1`).

    Neutral targets (`owner == -1`) and friendly targets (`owner == world.my_id`)
    are pass-through — neutrals don't produce; friendlies are reinforce
    intents and don't need over-sizing here.

    The bump is monotonic (`max(intent.ships, needed)`), so a strategy that
    asked for an over-spec'd swarm doesn't get cut down. If even our full
    garrison can't cover the production-grown target, drop the intent —
    sending an under-sized fleet would be pure waste.

    ETA is computed from the **current** intent.ships (i.e. the strategy's
    pre-bump estimate). One pass; the larger fleet would arrive faster
    and need slightly less of a bump, but the over-budget is safe.
    """
    out: list[Intent] = []
    for intent in intents:
        src = world.planets_by_id.get(intent.src_id)
        target = world.planets_by_id.get(intent.target_id)
        if src is None or target is None:
            out.append(intent)
            continue
        if target.owner == -1 or target.owner == world.my_id:
            out.append(intent)
            continue
        d = math.hypot(target.x - src.x, target.y - src.y)
        v = fleet_speed(intent.ships)
        eta = math.ceil(d / v) if v > 0 else 0
        needed = target.ships + target.production * eta + 1
        intent.ships = max(intent.ships, needed)
        if intent.ships > src.ships:
            continue
        out.append(intent)
    return out


# ---------------------------------------------------------------------------
# lead_aim — orbit-aware lead, ports v1's _aim_angle exactly
# ---------------------------------------------------------------------------


def lead_aim(intents: list[Intent], world: World) -> list[Intent]:
    """Populate `aim_angle` for each intent.

    For orbiting non-comet targets, performs one fixed-point iteration over
    `(arrival_time, predicted_position)` — the same algorithm v1 used in
    its embedded `_aim_angle`. For static planets and comets, falls through
    to atan2 of the current target position. Comet path-indexed leading is
    `comet_aim`'s job (3.5.C); this mechanism intentionally aims comets at
    current position so `comet_aim` can override.

    Intents that already have `aim_angle` set (e.g. by an earlier
    mechanism) are left untouched.
    """
    for intent in intents:
        if intent.aim_angle is not None:
            continue
        src = world.planets_by_id.get(intent.src_id)
        target = world.planets_by_id.get(intent.target_id)
        if src is None or target is None:
            continue

        target_xy = (target.x, target.y)
        target_tuple = [
            target.id, target.owner, target.x, target.y,
            target.radius, target.ships, target.production,
        ]
        is_orbit = (
            is_orbiting(target_tuple)
            and target.id not in world.comet_ids
        )
        if is_orbit and world.omega != 0.0:
            v = fleet_speed(intent.ships)
            tx, ty = target.x, target.y
            for _ in range(2):
                d = math.hypot(tx - src.x, ty - src.y)
                t = d / v
                tx, ty = predict_relative(target_tuple, world.omega, t)
            target_xy = (tx, ty)
        intent.aim_angle = math.atan2(target_xy[1] - src.y, target_xy[0] - src.x)
    return intents


# ---------------------------------------------------------------------------
# Canonical pipeline
# ---------------------------------------------------------------------------

# `arrival_size` runs BEFORE `lead_aim` because lead time depends on fleet
# size, which arrival_size revises. comet_aim/sun_avoid land in 3.5.C/D.
DEFAULT_MECHANISMS = [validate, arrival_size, lead_aim]

# Pinned subset for the v1 parity gate — must match pre-refactor v1
# behaviour exactly. Don't add new mechanisms here without bumping the
# pre-refactor snapshot.
PARITY_MECHANISMS = [validate, lead_aim]

__all__ = ["DEFAULT_MECHANISMS", "PARITY_MECHANISMS", "validate", "arrival_size", "lead_aim"]
