# Bundled by scripts/bundle_agent.py from agents/simple/roi.py + lib/{geometry,fleet,orbit,intent,mechanism}.
# Single-file Kaggle submission for Orbit Wars.

from __future__ import annotations

# === inlined: lib/geometry.py ===


import math

# Board / sun geometry — match Configuration table in data/README.md.
BOARD_SIZE: float = 100.0
CENTER: float = 50.0           # both x and y; sun is at (CENTER, CENTER)
SUN_RADIUS: float = 10.0
ROTATION_RADIUS_LIMIT: float = 50.0  # planet rotates iff orbital_radius + planet_radius < this


Point = tuple[float, float]


def dist(a: Point, b: Point) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_to_segment_distance(p: Point, a: Point, b: Point) -> float:
    """Shortest distance from point `p` to segment a->b.

    Used to determine whether a fleet's straight-line path clips the sun
    (continuous collision check, per data/README.md::Fleet Movement).
    """
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def path_clears_sun(src: Point, dst: Point, safety: float = 0.0) -> bool:
    """True iff the segment src->dst stays at distance > SUN_RADIUS + safety
    from the sun. `safety` is a margin in board units (default 0 = exact rule).
    """
    return point_to_segment_distance((CENTER, CENTER), src, dst) > SUN_RADIUS + safety

# === inlined: lib/fleet.py ===


import math


DEFAULT_MAX_SPEED: float = 6.0
LOG_1000: float = math.log(1000.0)


def speed(ships: int | float, max_speed: float = DEFAULT_MAX_SPEED) -> float:
    """Speed (board units per turn) for a fleet of size `ships`.

    Spec corner cases:
    - 1 ship  → exactly 1.0 (log(1)=0).
    - 1000 ships → exactly `max_speed` (log(1000)/log(1000)=1).
    - ships <= 0  → 1.0 (avoid log of non-positive; treat as floor speed).
    - ships > 1000 → clamped at `max_speed` (the formula would over-shoot
      otherwise; the env caps fleet speed at maxSpeed).
    """
    if ships <= 1:
        return 1.0
    if ships >= 1000:
        return float(max_speed)
    ratio = math.log(ships) / LOG_1000
    return 1.0 + (max_speed - 1.0) * (ratio ** 1.5)


def travel_time(src: Point, dst: Point, ships: int | float,
                max_speed: float = DEFAULT_MAX_SPEED) -> float:
    """Float turns for a `ships`-ship fleet to traverse the straight-line path.

    Does NOT account for sun collisions or board boundaries; callers should
    pre-filter via `geometry.path_clears_sun`. Returns inf for zero-distance
    plus zero ships (degenerate launch).
    """
    d = dist(src, dst)
    if d == 0.0:
        return 0.0
    return d / speed(ships, max_speed)


def eta_turns(src: Point, dst: Point, ships: int | float,
              max_speed: float = DEFAULT_MAX_SPEED) -> int:
    """Integer-turn ETA (ceil of travel_time). The arrival turn relative to
    the obs from which this is called.
    """
    t = travel_time(src, dst, ships, max_speed)
    return int(math.ceil(t)) if t > 0 else 0

# === inlined: lib/orbit.py ===


import math



def is_orbiting(planet) -> bool:
    """planet = [id, owner, x, y, radius, ships, production]."""
    px, py, pr = planet[2], planet[3], planet[4]
    orb_r = math.hypot(px - CENTER, py - CENTER)
    return (orb_r + pr) < ROTATION_RADIUS_LIMIT


def predict_relative(current_planet, angular_velocity: float, lead_turns: float) -> Point:
    """Predict (x, y) `lead_turns` after the obs that yielded `current_planet`.

    Safe for an agent that doesn't track absolute step count: read polar
    angle of the current planet position and rotate forward by
    `omega * lead_turns`. Returns the current (x, y) for static planets too,
    since rotating a position outside the rotation limit is a noop physically
    but the formula still works mathematically — caller should pre-filter
    via `is_orbiting` if performance matters.
    """
    px, py = current_planet[2], current_planet[3]
    dx, dy = px - CENTER, py - CENTER
    orb_r = math.hypot(dx, dy)
    cur_angle = math.atan2(dy, dx)
    new_angle = cur_angle + angular_velocity * lead_turns
    return (
        CENTER + orb_r * math.cos(new_angle),
        CENTER + orb_r * math.sin(new_angle),
    )


def predict_absolute(initial_planet, angular_velocity: float, env_step_n: int) -> Point:
    """Predict (x, y) at `env.steps[env_step_n]` from `initial_planets`.

    Uses the empirically-correct N-1 rotation count for N>=1 (see
    `audit/2026-05-10-day-1-data-inventory.md::A.1`). The naive `omega*N`
    form is off by exactly one step's rotation (~1.27 board units on
    inner planets at the default angular velocity).
    """
    px, py = initial_planet[2], initial_planet[3]
    dx, dy = px - CENTER, py - CENTER
    orb_r = math.hypot(dx, dy)
    init_angle = math.atan2(dy, dx)
    n_rot = max(env_step_n - 1, 0)
    cur_angle = init_angle + angular_velocity * n_rot
    return (
        CENTER + orb_r * math.cos(cur_angle),
        CENTER + orb_r * math.sin(cur_angle),
    )

# === inlined: lib/intent.py ===


from dataclasses import dataclass

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


@dataclass
class Intent:
    """A strategy's request for a single fleet launch.

    `ships` is the strategy's *desired* size; `arrival_size` (the mechanism)
    may revise it upward to account for production growth during flight.
    `aim_angle` starts None and is populated by `lead_aim` / `comet_aim`.
    Mechanisms may also drop intents (e.g. `validate`, `sun_avoid` when no
    detour exists).
    """
    src_id: int
    target_id: int
    ships: int
    aim_angle: float | None = None
    note: str = ""


@dataclass
class World:
    """Frozen-once-per-turn view over an obs.

    Built once at the top of `realize()` and passed to every mechanism so
    each one is a pure function of `(intents, world)` — easy to test, easy
    to reorder. `obs_raw` is kept for mechanisms that need fields not yet
    materialised here (e.g. comet paths in `comet_aim`).
    """
    my_id: int
    planets_by_id: dict[int, "Planet"]
    omega: float
    comet_ids: frozenset[int]
    step: int
    obs_raw: object

    @classmethod
    def from_obs(cls, obs) -> "World":
        my_id = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        omega = (
            float(obs.get("angular_velocity", 0.0))
            if isinstance(obs, dict)
            else float(getattr(obs, "angular_velocity", 0.0))
        )
        raw_comet_ids = (
            obs.get("comet_planet_ids", [])
            if isinstance(obs, dict)
            else getattr(obs, "comet_planet_ids", [])
        )
        step = (
            int(obs.get("step", 0))
            if isinstance(obs, dict)
            else int(getattr(obs, "step", 0))
        )
        planets_by_id = {p[0]: Planet(*p) for p in raw_planets}
        comet_ids = frozenset(int(c) for c in raw_comet_ids) if raw_comet_ids else frozenset()
        return cls(
            my_id=my_id,
            planets_by_id=planets_by_id,
            omega=omega,
            comet_ids=comet_ids,
            step=step,
            obs_raw=obs,
        )


def realize(intents, obs, *, mechanisms) -> list[list]:
    """Apply the mechanism pipeline and emit env-format actions.

    Final emission to `[src_id, aim_angle, ships]` lists is hard-coded —
    NOT a user-pluggable mechanism. Intents missing `aim_angle` or with
    `ships <= 0` after the pipeline are silently dropped (treated as
    validation failures).
    """
    world = World.from_obs(obs)
    for m in mechanisms:
        intents = m(intents, world)
    return [
        [i.src_id, i.aim_angle, i.ships]
        for i in intents
        if i.ships > 0 and i.aim_angle is not None
    ]

# === inlined: lib/mechanism.py ===


import math

fleet_speed = speed


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

# === agent ===
"""roi — target the highest production-per-distance planet.

HYPOTHESIS: production-per-distance is the right travel-adjusted ROI
signal; pure-production (production.py) over-commits to far targets and
pure-nearest under-commits to high-yield ones. ROI should sit in the
middle and beat both.

Score: production / (distance + 1.0). The +1 prevents division-by-zero
when a target sits at the source's coordinates (degenerate, but defensive).
Tiebreaker: distance ascending.
"""


import random

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

# from lib.geometry import dist  # inlined by bundle_agent.py
# from lib.intent import Intent, realize  # inlined by bundle_agent.py
# from lib.mechanism import DEFAULT_MECHANISMS  # inlined by bundle_agent.py


def _score(mine: Planet, target: Planet) -> tuple:
    d = dist((mine.x, mine.y), (target.x, target.y))
    roi = target.production / (d + 1.0)
    return (-roi, d)   # argmax roi, tiebreak: nearest


def propose_intents(obs) -> list[Intent]:
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    step = (
        int(obs.get("step", 0))
        if isinstance(obs, dict)
        else int(getattr(obs, "step", 0))
    )

    planets = [Planet(*p) for p in raw_planets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    if not my_planets or not targets:
        return []

    rng = random.Random(step ^ (player + 1) * 1009)
    intents: list[Intent] = []
    for mine in my_planets:
        scored = [(_score(mine, t), rng.random(), t) for t in targets]
        scored.sort(key=lambda e: (e[0], e[1]))
        target = scored[0][2]
        intents.append(
            Intent(src_id=mine.id, target_id=target.id, ships=target.ships + 1)
        )
    return intents


def agent(obs):
    return realize(propose_intents(obs), obs, mechanisms=DEFAULT_MECHANISMS)
