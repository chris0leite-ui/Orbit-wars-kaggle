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
from lib.geometry import path_clears_sun
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
# comet_aim — path-indexed lead for comet targets
# ---------------------------------------------------------------------------


def _comet_path_lookup(world: World) -> dict[int, tuple[list, int]]:
    """Build {planet_id: (path, path_index)} for every comet in the obs.

    `obs["comets"]` is a list of groups, each `{planet_ids, paths, path_index}`.
    `paths[i]` is the trajectory of `planet_ids[i]` — a list of `[x, y]`
    pairs. `path_index` is shared across the group; advances 1 per turn.
    """
    raw = world.obs_raw
    comets = (
        raw.get("comets", []) if isinstance(raw, dict) else getattr(raw, "comets", [])
    )
    out: dict[int, tuple[list, int]] = {}
    for group in comets:
        if hasattr(group, "keys"):
            planet_ids = list(group["planet_ids"])
            paths = list(group["paths"])
            path_index = int(group["path_index"])
        else:
            planet_ids = list(group.planet_ids)
            paths = list(group.paths)
            path_index = int(group.path_index)
        for idx, pid in enumerate(planet_ids):
            out[int(pid)] = (paths[idx], path_index)
    return out


def comet_aim(intents: list[Intent], world: World) -> list[Intent]:
    """Populate `aim_angle` for comet targets via the path-indexed lead.

    Comets follow pre-computed elliptical paths, NOT the rotation formula —
    so `lead_aim`'s orbit prediction would mis-aim them. This mechanism
    fires on targets in `world.comet_ids`, looks up the comet's path,
    projects to `path_index + eta_turns`, and aims at the projected point.

    If `path_index + eta_turns` exceeds the path length the comet exits
    the board before our fleet arrives — drop the intent (sending an
    on-the-way fleet at an exit-bound comet would be wasted).

    **Status: experimental, NOT in DEFAULT_MECHANISMS.** The 3.5.C ablation
    tournament showed this single-pass version loses 9/40 = 22.5% vs the
    parity baseline. See the rationale comment near `DEFAULT_MECHANISMS`
    for the diagnosis. Kept as a registered mechanism so tournament panels
    can opt it in for future experiments (e.g. paired with a
    `search_safe_intercept` fallback at v3).
    """
    if not world.comet_ids:
        return intents
    paths_by_id = _comet_path_lookup(world)

    out: list[Intent] = []
    for intent in intents:
        if intent.target_id not in world.comet_ids:
            out.append(intent)
            continue
        if intent.aim_angle is not None:
            out.append(intent)
            continue
        src = world.planets_by_id.get(intent.src_id)
        target = world.planets_by_id.get(intent.target_id)
        path_info = paths_by_id.get(intent.target_id)
        if src is None or target is None or path_info is None:
            out.append(intent)
            continue
        path, path_index = path_info
        v = fleet_speed(intent.ships)
        d = math.hypot(target.x - src.x, target.y - src.y)
        eta = math.ceil(d / v) if v > 0 else 0
        future_index = path_index + eta
        if future_index >= len(path):
            # Comet exits before the fleet arrives — drop rather than waste ships.
            continue
        fx, fy = path[future_index]
        intent.aim_angle = math.atan2(fy - src.y, fx - src.x)
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

# Pipeline order rationale:
#   validate      — drop unsafe intents up-front so nothing downstream
#                   computes against bad data.
#   arrival_size  — bump fleet size for enemy targets BEFORE lead_aim/comet_aim
#                   because lead time (and thus the projected position) depends
#                   on fleet size via fleet_speed.
#   lead_aim      — populates aim_angle for everything else (orbiting non-comets
#                   get the orbit-fixed-point lead; statics get plain atan2;
#                   comets get current-position atan2 — see note below).
#   sun_avoid (3.5.D) — last; needs the angle set by lead_aim/comet_aim.
#
# ---------------------------------------------------------------------------
# sun_avoid — drop intents whose straight-line path crosses the sun
# ---------------------------------------------------------------------------


def sun_avoid(intents: list[Intent], world: World) -> list[Intent]:
    """Drop intents whose direct fleet path would intersect the sun.

    The env destroys any fleet whose continuous segment crosses the sun
    (data/README.md::Fleet Movement). If we can predict that loss before
    launching, we keep the ships in garrison instead — production catches
    up over the next few turns and the next launch can re-evaluate.

    This drop-only version is deliberately conservative. A future variant
    could re-aim at a friendly waypoint planet whose two-leg path clears
    the sun, but that requires multi-turn planning (the env can't actually
    bend fleet trajectories mid-flight; we'd just be sending ships TO the
    waypoint and re-launching from there next turn).

    Path check uses planet centers with a 1-unit safety margin — the env
    spawns the fleet just outside the source radius, but the safety
    margin absorbs that.
    """
    out: list[Intent] = []
    for intent in intents:
        if intent.aim_angle is None:
            # Not yet aimed (e.g. comet_aim dropped, or some prior mechanism
            # hasn't run). Pass through and let realize()'s emission filter
            # drop unaimed intents.
            out.append(intent)
            continue
        src = world.planets_by_id.get(intent.src_id)
        target = world.planets_by_id.get(intent.target_id)
        if src is None or target is None:
            out.append(intent)
            continue
        if path_clears_sun((src.x, src.y), (target.x, target.y), safety=1.0):
            out.append(intent)
            continue
        # Sun-blocked → drop. The fleet would die in flight; better to keep
        # ships in the source garrison.
    return out


# `comet_aim` is implemented + unit-tested but EXCLUDED from DEFAULT_MECHANISMS
# because the ablation tournament (audit/tournaments/20260510T090723Z.json)
# showed it loses 9/40 = 22.5% vs the parity baseline. Plausible cause: with a
# one-shot ETA estimate, the forward projection can overshoot — the env's
# continuous collision check actually rewards `lead_aim`'s current-position
# aim more often than `comet_aim`'s far-projection on small fleets at
# log-curve speeds. The 3 public top notebooks (Roman 1224 et al) pair their
# version with `search_safe_intercept` fallback (try multiple arrival times)
# which we don't yet implement. Revisit when v3's world-model lands.
# `sun_avoid` is implemented + unit-tested but EXCLUDED from DEFAULT_MECHANISMS
# because the 3.5.D ablation tournament (audit/tournaments/...) showed it
# loses 13/40 = 32.5% vs the parity baseline. Diagnosis: drop-only is correct
# at the mechanism level (don't lose the fleet to the sun), but v1's
# nearest-target strategy gets *stuck* — the sun-blocked target stays
# nearest each turn, sun_avoid keeps dropping, ships pile up in garrison
# doing nothing. v2's arrival-ledger strategy can pivot to a different
# target when the nearest is blocked, so sun_avoid will become positive
# there. Tracked for v2 in DEFAULT_MECHANISMS.
DEFAULT_MECHANISMS = [validate, arrival_size, lead_aim]

# Pinned subset for the v1 parity gate — must match pre-refactor v1
# behaviour exactly. Don't add new mechanisms here without bumping the
# pre-refactor snapshot.
PARITY_MECHANISMS = [validate, lead_aim]

__all__ = [
    "DEFAULT_MECHANISMS",
    "PARITY_MECHANISMS",
    "validate",
    "arrival_size",
    "comet_aim",
    "lead_aim",
    "sun_avoid",
]
