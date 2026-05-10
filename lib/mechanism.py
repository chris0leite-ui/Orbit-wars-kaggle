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

from lib.aim import aim_orbiting, swept_pair_hit
from lib.fleet import speed as fleet_speed
from lib.geometry import BOARD_SIZE, path_clears_sun
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
            # Fleet spawns just outside source (src.radius + 0.1) and
            # captures when it crosses into target.radius. Subtract both
            # from center-to-center distance to get actual flight distance.
            # Without this, ETA overestimates and lead is too far ahead —
            # systematic miss in the orbit-forward direction.
            r_offset = src.radius + target.radius + 0.1
            tx, ty = target.x, target.y
            for _ in range(2):
                d = math.hypot(tx - src.x, ty - src.y)
                flight_d = max(0.0, d - r_offset)
                t = flight_d / v
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
    """Drop intents whose actual fleet path would intersect the sun.

    Punch-#7 upgrade (2026-05-10 PM): use the lead-predicted arrival point
    if `intent.arrival_xy` is set (populated by `lead_aim_v2`); otherwise
    fall back to the target's current xy (matches the pre-upgrade behaviour
    and keeps the old test suite green).

    The check is against the fleet's straight-line segment from the actual
    spawn point (just outside src.radius along aim_angle) to the arrival
    endpoint. 1-unit safety margin absorbs float drift on tangent cases.

    Drop-only — re-routing via a waypoint planet is a v3 mission concern
    (`stage_then_strike`), not a mechanism. Sun-blocked intents keep their
    ships in garrison; the next launch re-evaluates.
    """
    out: list[Intent] = []
    for intent in intents:
        if intent.aim_angle is None:
            out.append(intent)
            continue
        src = world.planets_by_id.get(intent.src_id)
        target = world.planets_by_id.get(intent.target_id)
        if src is None or target is None:
            out.append(intent)
            continue
        # Endpoint: lead-predicted arrival if available, else target.xy.
        end_xy = intent.arrival_xy if intent.arrival_xy is not None else (target.x, target.y)
        if path_clears_sun((src.x, src.y), end_xy, safety=1.0):
            out.append(intent)
            continue
        # Sun-blocked → drop.
    return out


# ---------------------------------------------------------------------------
# lead_aim_v2 — 5-iter fixed-point + search_safe_intercept fallback
# ---------------------------------------------------------------------------


def lead_aim_v2(intents: list[Intent], world: World) -> list[Intent]:
    """Populate `aim_angle` AND `arrival_xy` for each intent via the
    public-kernel pattern: 5-iter fixed-point + safe-intercept fallback.

    Differences from the legacy `lead_aim`:
    - 5 iterations (was 2) with explicit XY convergence check.
    - `search_safe_intercept` fallback when the fixed-point doesn't
      converge (orbital targets at long range, where eta oscillates).
    - Populates `intent.arrival_xy` so `sun_avoid`, `path_clears_other_planets`,
      and `oob_guard` downstream can reason about the actual fleet endpoint.
    - For static targets and comets, falls through to atan2 of current
      target position (same as legacy lead_aim; `comet_aim` overrides
      comets when enabled).

    Intents that already have `aim_angle` set are left untouched
    (mechanism ordering: a future planner-set aim shouldn't be clobbered).
    """
    for intent in intents:
        if intent.aim_angle is not None:
            continue
        src = world.planets_by_id.get(intent.src_id)
        target = world.planets_by_id.get(intent.target_id)
        if src is None or target is None:
            continue

        target_tuple = [
            target.id, target.owner, target.x, target.y,
            target.radius, target.ships, target.production,
        ]
        is_orbit = (
            is_orbiting(target_tuple)
            and target.id not in world.comet_ids
        )

        if is_orbit and world.omega != 0.0:
            result = aim_orbiting(
                (src.x, src.y), src.radius,
                target_tuple, target.radius,
                intent.ships, world.omega,
            )
            if result is None:
                # No valid intercept — let realize() drop the intent
                # via the aim_angle=None gate.
                continue
            intent.aim_angle, intent.arrival_xy, _eta = result
        else:
            # Static or comet → aim at current; record arrival_xy for
            # downstream sun/OOB/path checks even though there's no lead.
            intent.aim_angle = math.atan2(target.y - src.y, target.x - src.x)
            intent.arrival_xy = (target.x, target.y)
    return intents


# ---------------------------------------------------------------------------
# path_clears_other_planets — drop intents swept by a non-target planet
# ---------------------------------------------------------------------------


def path_clears_other_planets(intents: list[Intent], world: World) -> list[Intent]:
    """Drop intents whose flight path collides with a non-target planet.

    Capture-probe (audit/2026-05-10-capture-success-probe.md) showed
    10.7% of roi's fleets hit a non-target planet mid-flight — the
    biggest physics-loss bucket. This mechanism replays the env's
    `swept_pair_hit(fleet_seg, planet_seg, planet.radius)` check at each
    step of the fleet's projected flight, against every planet's
    projected per-step segment (orbital chord).

    Precomputes per-planet trajectories over a SEARCH_HORIZON of 60
    steps once per turn to amortise the cos/sin cost.
    """
    if not intents:
        return intents

    SEARCH_HORIZON = 60
    omega = world.omega

    # Precompute per-planet (positions[0..H], radius) once per turn.
    planet_traj: dict[int, tuple[list, float]] = {}
    for pid, p in world.planets_by_id.items():
        p_tuple = [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
        if is_orbiting(p_tuple) and omega != 0.0:
            positions = [
                predict_relative(p_tuple, omega, t) for t in range(SEARCH_HORIZON + 1)
            ]
        else:
            positions = [(p.x, p.y)] * (SEARCH_HORIZON + 1)
        planet_traj[pid] = (positions, p.radius)

    out: list[Intent] = []
    for intent in intents:
        if intent.aim_angle is None:
            out.append(intent)
            continue
        src = world.planets_by_id.get(intent.src_id)
        target = world.planets_by_id.get(intent.target_id)
        if src is None or target is None:
            out.append(intent)
            continue

        speed_val = fleet_speed(intent.ships)
        cos_a = math.cos(intent.aim_angle)
        sin_a = math.sin(intent.aim_angle)
        # Fleet spawn point (env: source + (r_src + 0.1) * direction).
        spawn_x = src.x + cos_a * (src.radius + 0.1)
        spawn_y = src.y + sin_a * (src.radius + 0.1)

        # Endpoint for distance estimate — predicted arrival if available.
        end_xy = intent.arrival_xy if intent.arrival_xy is not None else (target.x, target.y)
        total_dist = math.hypot(end_xy[0] - spawn_x, end_xy[1] - spawn_y)
        max_steps = min(
            SEARCH_HORIZON, int(math.ceil(total_dist / max(speed_val, 1e-6))) + 1,
        )

        collided = False
        for step in range(max_steps):
            fleet_old = (
                spawn_x + cos_a * speed_val * step,
                spawn_y + sin_a * speed_val * step,
            )
            fleet_new = (
                spawn_x + cos_a * speed_val * (step + 1),
                spawn_y + sin_a * speed_val * (step + 1),
            )
            for pid, (positions, prad) in planet_traj.items():
                if pid == target.id:
                    continue
                if pid == src.id and step == 0:
                    # First-step spawn is exactly outside src.radius — env
                    # explicitly does not let the fleet collide with its
                    # own source on the spawn step. Skip.
                    continue
                p_old = positions[step]
                p_new = positions[step + 1]
                if swept_pair_hit(fleet_old, fleet_new, p_old, p_new, prad):
                    collided = True
                    break
            if collided:
                break
        if not collided:
            out.append(intent)
        # Collided → drop. Ships stay in garrison for next-turn re-evaluation.
    return out


# ---------------------------------------------------------------------------
# oob_guard — drop intents whose projected endpoint exits the board
# ---------------------------------------------------------------------------


def oob_guard(intents: list[Intent], world: World) -> list[Intent]:
    """Drop intents whose projected endpoint goes off-board.

    Capture-probe showed 7.6% OOB. The cause is usually that lead_aim
    overshoots: the fleet flies past the target in a straight line and
    exits the [0, BOARD_SIZE] box. We drop those intents — the ships
    are wasted otherwise.

    Cheap endpoint test. Since fleets travel in straight lines and
    targets sit inside [0, BOARD_SIZE], if BOTH spawn and endpoint are
    inside the box, the path is inside too.
    """
    out: list[Intent] = []
    for intent in intents:
        if intent.aim_angle is None:
            out.append(intent)
            continue
        src = world.planets_by_id.get(intent.src_id)
        target = world.planets_by_id.get(intent.target_id)
        if src is None or target is None:
            out.append(intent)
            continue
        cos_a = math.cos(intent.aim_angle)
        sin_a = math.sin(intent.aim_angle)
        spawn_x = src.x + cos_a * (src.radius + 0.1)
        spawn_y = src.y + sin_a * (src.radius + 0.1)
        end_xy = intent.arrival_xy if intent.arrival_xy is not None else (target.x, target.y)
        total_dist = math.hypot(end_xy[0] - spawn_x, end_xy[1] - spawn_y)
        end_x = spawn_x + cos_a * total_dist
        end_y = spawn_y + sin_a * total_dist
        if (
            end_x < 0.0
            or end_x > BOARD_SIZE
            or end_y < 0.0
            or end_y > BOARD_SIZE
        ):
            continue
        out.append(intent)
    return out


# 2026-05-10 PM physics upgrade (capture-probe + Roman teardown):
# - `lead_aim_v2` replaces `lead_aim` in DEFAULT_MECHANISMS. 5-iter
#   fixed-point + `search_safe_intercept` fallback (lib/aim.py). Populates
#   `intent.arrival_xy` so downstream checks reason about the actual
#   flight endpoint.
# - `sun_avoid` re-enabled with the punch-#7 fix: uses `intent.arrival_xy`
#   if set (lead-predicted arrival) instead of `target.xy`. Previous
#   regressions are addressed because the check now matches the actual
#   fleet trajectory.
# - `path_clears_other_planets` added: addresses the 10.7% collided_other
#   bucket from the capture probe. Replays the env's swept-pair check
#   against every non-target planet's projected orbital chord.
# - `oob_guard` added: addresses the 7.6% OOB bucket. Drops intents whose
#   projected endpoint exits the board.
# - `comet_aim` remains EXCLUDED pending a comet-gated re-enable
#   (research-note §G.14: gate on `production * expected_lifetime > ships`).
DEFAULT_MECHANISMS = [
    validate,
    arrival_size,
    lead_aim_v2,
    sun_avoid,
    path_clears_other_planets,
    oob_guard,
]

# Frozen pre-upgrade stack (validate + arrival_size + 2-iter lead_aim only).
# Used by `agents/simple/roi_baseline.py` for A/B against the upgraded
# DEFAULT_MECHANISMS without round-tripping through a bundled submission.
DEFAULT_MECHANISMS_PRE_PHYSICS = [validate, arrival_size, lead_aim]

# Pinned subset for the v1 parity gate — must match pre-refactor v1
# behaviour exactly. Don't add new mechanisms here without bumping the
# pre-refactor snapshot.
PARITY_MECHANISMS = [validate, lead_aim]

__all__ = [
    "DEFAULT_MECHANISMS",
    "DEFAULT_MECHANISMS_PRE_PHYSICS",
    "PARITY_MECHANISMS",
    "validate",
    "arrival_size",
    "comet_aim",
    "lead_aim",
    "lead_aim_v2",
    "sun_avoid",
    "path_clears_other_planets",
    "oob_guard",
]
