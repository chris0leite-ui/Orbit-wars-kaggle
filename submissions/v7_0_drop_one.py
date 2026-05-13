# Bundled by scripts/bundle_agent.py from agents/v7_ablations/v7_0_drop_one + lib/{geometry,fleet,orbit,aim,combat,world_model,intent,trajectory,mechanism,mission,missions/snipe,missions/reinforce,missions/recapture,planner,lookahead_planner,game/interpreter,fast_sim,opp_model,v7_search,candidate_portfolios,value_heads}.
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


def sym_hypot(dx: float, dy: float) -> float:
    """Order-independent hypot — same bits for (dx, dy) and (dy, dx).

    Standard `math.hypot(a, b) = sqrt(a² + b²)` is mathematically
    symmetric in its arguments but NOT bit-exact under FP rounding:
    `a² + b²` and `b² + a²` can differ by 1 ULP because the addition
    is non-associative. Over thousands of mission-score comparisons,
    this 1-ULP noise turns near-ties into strict orderings, defeating
    σ-equivariant tie-breaks. `sym_hypot` canonicalises arguments to
    `hypot(min(|dx|,|dy|), max(|dx|,|dy|))` so σ-paired (src, target)
    pairs produce bit-equal distances.

    Ported from `origin/claude/game-theory-strategy-analysis-0oH4N`
    where the σ-equiv layer (this + planner _tb + score rounding) was
    the load-bearing change behind σ-equiv-v1 (μ=1041.4) and
    v7_minimax (μ=1063).
    """
    ax = abs(dx)
    ay = abs(dy)
    if ax > ay:
        ax, ay = ay, ax
    return math.hypot(ax, ay)


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

# === inlined: lib/aim.py ===


import math

fleet_speed = speed

# Tolerance bands tuned from public kernel patterns (Roman §K).
INTERCEPT_TOLERANCE = 1        # +/- step delta between predicted and candidate
SEARCH_HORIZON = 60            # max future steps to scan in safe-intercept
CONVERGENCE_XY_TOL = 0.3       # |dx|, |dy| convergence threshold in fixed-point
MAX_ITERATIONS = 5             # fixed-point iter cap (was 1 in v1, 2 in lead_aim)


def flight_distance(src_xy, src_radius, target_xy, target_radius):
    """Center-to-center distance minus launch offset minus capture radius.

    The env spawns the fleet just outside the source at `r_src + 0.1`, and
    captures when the fleet enters `target.radius`. So the actual *flight*
    distance is `dist(src, target) - r_src - r_target - 0.1`, clamped at 0
    to avoid negative ETA for degenerate launches at the source.
    """
    d = math.hypot(target_xy[0] - src_xy[0], target_xy[1] - src_xy[1])
    return max(0.0, d - src_radius - target_radius - 0.1)


def estimate_eta(src_xy, src_radius, target_xy, target_radius, ships):
    """Floating-point ETA in steps. None if speed is degenerate.

    Returns the FLOAT step count to traverse the flight distance at the
    fleet's speed (a function of ships via the log-curve). Callers ceil
    or floor as their step semantics require.
    """
    flight = flight_distance(src_xy, src_radius, target_xy, target_radius)
    v = fleet_speed(ships)
    if v <= 0:
        return None
    return flight / v


def search_safe_intercept(
    src_xy,
    src_radius,
    target_tuple,
    target_radius,
    ships,
    omega,
    horizon=SEARCH_HORIZON,
):
    """Self-consistent intercept search over candidate arrival turns.

    For each `candidate_turns` in `[1, horizon]`:
    1. Predict target position at `candidate_turns` (orbital projection).
    2. Estimate ETA from current source to that predicted position.
    3. If |ETA - candidate_turns| <= INTERCEPT_TOLERANCE, the candidate is
       self-consistent — i.e. firing at it now would actually intersect
       the target at the predicted step.

    Returns (aim_angle, arrival_xy, eta) for the best (smallest delta,
    then smallest turn count) candidate, or None if no candidate is
    self-consistent.

    This is the fallback for cases where 5-iter fixed-point doesn't
    converge — typically orbital targets at very long distances where
    `eta` oscillates between two values.
    """
    best = None
    best_score = None
    for cand_t in range(1, horizon + 1):
        pred_xy = predict_relative(target_tuple, omega, cand_t)
        eta = estimate_eta(src_xy, src_radius, pred_xy, target_radius, ships)
        if eta is None:
            continue
        delta = abs(eta - cand_t)
        if delta > INTERCEPT_TOLERANCE:
            continue
        score = (delta, cand_t)
        if best is None or score < best_score:
            best_score = score
            angle = math.atan2(pred_xy[1] - src_xy[1], pred_xy[0] - src_xy[0])
            best = (angle, pred_xy, eta)
    return best


def aim_orbiting(src_xy, src_radius, target_tuple, target_radius, ships, omega):
    """5-iter fixed-point lead for orbiting non-comet targets, with
    safe-intercept fallback when iteration doesn't converge.

    Returns (aim_angle, arrival_xy, eta) or None if no valid intercept.

    Algorithm:
    1. Start with target's current position.
    2. Estimate ETA to current target position.
    3. Predict target position at that ETA via orbital projection.
    4. Re-estimate ETA to the predicted position.
    5. Repeat up to MAX_ITERATIONS. Convergence = |dx|, |dy| < TOL.
    6. If converged, return; else fall back to search_safe_intercept.

    The fallback exists because at long ranges / fast orbital motion,
    the fixed-point can oscillate between two estimates rather than
    converge. Roman 1224 uses 5 iter + this fallback; we follow the
    pattern.
    """
    tx, ty = target_tuple[2], target_tuple[3]
    last_eta = None
    for _ in range(MAX_ITERATIONS):
        eta = estimate_eta(src_xy, src_radius, (tx, ty), target_radius, ships)
        if eta is None:
            return search_safe_intercept(
                src_xy, src_radius, target_tuple, target_radius, ships, omega,
            )
        ntx, nty = predict_relative(target_tuple, omega, eta)
        if (
            last_eta is not None
            and abs(ntx - tx) < CONVERGENCE_XY_TOL
            and abs(nty - ty) < CONVERGENCE_XY_TOL
        ):
            angle = math.atan2(nty - src_xy[1], ntx - src_xy[0])
            return angle, (ntx, nty), eta
        tx, ty = ntx, nty
        last_eta = eta

    # Non-convergence → safe-intercept fallback.
    fb = search_safe_intercept(
        src_xy, src_radius, target_tuple, target_radius, ships, omega,
    )
    if fb is not None:
        return fb

    # Last resort: return final iteration's guess (better than nothing —
    # the fleet still launches; physics decides the outcome).
    angle = math.atan2(ty - src_xy[1], tx - src_xy[0])
    return angle, (tx, ty), last_eta or 0.0


def swept_pair_hit(A, B, P0, P1, r):
    """Mirror of the env's swept-pair collision check (orbit_wars.py:46-67).

    True iff a fleet moving A->B and a planet moving P0->P1 come within
    `r` of each other for some t in [0, 1]. Treats both motions as
    linear over the tick (planet rotation is linearised to its chord).

    Used by `path_clears_other_planets` to detect mid-flight collisions
    with non-target planets — the largest physics-loss bucket from the
    capture probe (10.7%).
    """
    d0x, d0y = A[0] - P0[0], A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0

# === inlined: lib/combat.py ===



def resolve_arrivals(
    garrison_owner: int,
    garrison_ships: float,
    arrivals: list[tuple[int, int]],
) -> tuple[int, float]:
    """Resolve combat at a planet given current garrison + same-step arrivals.

    `arrivals` = list of `(owner, ships)` pairs.
    Returns `(new_owner, new_ships)`.

    Neutral planets (`owner == -1`) hold the garrison until combat resolves.
    Owner of -1 means neutral; non-negative ints are player IDs.
    """
    # Group arrivals by owner; sum ship counts.
    by_owner: dict[int, int] = {}
    for owner, ships in arrivals:
        if ships <= 0:
            continue
        by_owner[owner] = by_owner.get(owner, 0) + int(ships)

    if not by_owner:
        return garrison_owner, max(0.0, garrison_ships)

    # Rank attackers by ship count descending.
    ranked = sorted(by_owner.items(), key=lambda kv: kv[1], reverse=True)
    top_owner, top_ships = ranked[0]

    if len(ranked) > 1:
        second_ships = ranked[1][1]
        if top_ships == second_ships:
            # Two-way tie among attackers — all destroyed (rule 4).
            survivor_owner = -1
            survivor_ships = 0
        else:
            # Largest minus second-largest survives (rule 2).
            survivor_owner = top_owner
            survivor_ships = top_ships - second_ships
    else:
        survivor_owner = top_owner
        survivor_ships = top_ships

    if survivor_ships <= 0:
        return garrison_owner, max(0.0, garrison_ships)

    # Survivor vs garrison.
    if garrison_owner == survivor_owner:
        # Same owner — reinforce (rule 3a).
        return garrison_owner, garrison_ships + survivor_ships

    # Different owner — survivor attacks garrison (rule 3b).
    garrison_ships -= survivor_ships
    if garrison_ships < 0:
        # Survivor wins; remaining = -garrison_ships.
        return survivor_owner, -garrison_ships
    return garrison_owner, garrison_ships

# === inlined: lib/world_model.py ===


import math
from collections import defaultdict
from dataclasses import dataclass

fleet_speed = speed

# Raised 110 → 250 (2026-05-11): reinforce class was firing 0.2
# candidates/turn because long-runway threats were invisible past
# step 110 + eta. Matches the EPISODE_STEPS/2 framing in the score
# formula (`time_to_hold = 500 - step - eta`); timeline-build cost
# scales linearly so per-turn p95 should remain well under the 1s
# actTimeout. See audit/2026-05-11-v3-snipe-critical-review.md §P2.
DEFAULT_HORIZON = 250


def fleet_target_planet(fleet, planets, max_horizon: int = DEFAULT_HORIZON):
    """Trace `fleet` along its angle, find first planet it'd hit.

    Returns `(target_planet, eta_turns)` or `(None, None)` if no planet
    intersects the fleet's trajectory within `max_horizon` steps.

    Used to build the arrival ledger from in-flight fleets — the env
    doesn't expose a fleet's intended target, only its angle.

    Note: this is a *non-orbiting* ray-cast — it doesn't account for
    target planets moving while the fleet is in flight. For inner
    orbiting planets the attribution can be off by a step or two. The
    arrival ledger uses these eta estimates to *roughly* predict
    ownership; sub-step precision matters less than not double-committing.
    """
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    spd = fleet_speed(fleet.ships)
    if spd <= 0:
        return None, None

    best_planet = None
    best_turns = None
    for p in planets:
        dx = p.x - fleet.x
        dy = p.y - fleet.y
        proj = dx * dir_x + dy * dir_y
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        r_sq = p.radius * p.radius
        if perp_sq >= r_sq:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, r_sq - perp_sq)))
        turns = hit_d / spd
        if turns <= max_horizon and (best_turns is None or turns < best_turns):
            best_turns = turns
            best_planet = p
    if best_planet is None:
        return None, None
    return best_planet, int(math.ceil(best_turns))


def build_arrival_ledger(fleets, planets, horizon: int = DEFAULT_HORIZON):
    """{planet_id: [(eta, owner, ships), ...]} for in-flight fleets.

    Fleets that won't hit any planet within `horizon` are dropped (they
    will exit the board or die in sun/non-target collision — out of
    scope for the timeline).
    """
    ledger: dict[int, list[tuple[int, int, int]]] = {p.id: [] for p in planets}
    for fleet in fleets:
        target, eta = fleet_target_planet(fleet, planets, horizon)
        if target is None:
            continue
        ledger[target.id].append((eta, int(fleet.owner), int(fleet.ships)))
    return ledger


def simulate_planet_timeline(planet, arrivals, horizon: int = DEFAULT_HORIZON):
    """Per-planet step-by-step ownership/garrison simulation.

    `arrivals` is a list of `(eta, owner, ships)`. For each step `t` in
    `[1, horizon]`:
    1. If currently owned (not neutral), produce `production` ships.
    2. Resolve same-step arrivals via `resolve_arrivals`.
    3. Record `owner_at[t]`, `ships_at[t]`.

    Returns a dict with `owner_at` (dict[int, int]), `ships_at`
    (dict[int, float]), and `horizon` (int).
    """
    horizon = max(0, int(math.ceil(horizon)))
    by_turn: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for eta, owner, ships in arrivals:
        if ships <= 0:
            continue
        bucket = max(1, int(math.ceil(eta)))
        if bucket > horizon:
            continue
        by_turn[bucket].append((owner, int(ships)))

    owner = planet.owner
    garrison = float(planet.ships)
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}

    for t in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production
        group = by_turn.get(t, [])
        if group:
            owner, garrison = resolve_arrivals(owner, garrison, group)
        owner_at[t] = owner
        ships_at[t] = max(0.0, garrison)

    return {"owner_at": owner_at, "ships_at": ships_at, "horizon": horizon}


def state_at_timeline(timeline, arrival_turn):
    """Read `(owner, ships)` from a timeline at a given turn.

    Clamps `arrival_turn` to `[0, timeline['horizon']]`. Reads from the
    `owner_at` / `ships_at` dicts.
    """
    t = min(max(0, int(math.ceil(arrival_turn))), timeline["horizon"])
    return timeline["owner_at"][t], timeline["ships_at"][t]


@dataclass
class WorldModel:
    """Per-turn arrival-ledger snapshot. Built once at the top of an
    agent's turn; consumed by the `arrival_ledger` mechanism today and
    by mission scoring tomorrow."""

    ledger: dict
    timelines: dict
    horizon: int = DEFAULT_HORIZON

    @classmethod
    def from_world(cls, world, horizon: int = DEFAULT_HORIZON):
        """Build from `lib.intent.World`'s obs_raw. Reads in-flight fleets
        directly from the raw obs because `World` doesn't materialise them."""
        raw = world.obs_raw
        if isinstance(raw, dict):
            fleets_raw = raw.get("fleets", [])
        else:
            fleets_raw = getattr(raw, "fleets", [])

        from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet  # local import — keeps lib/ env-free
        fleets = [Fleet(*f) for f in fleets_raw]
        planets = list(world.planets_by_id.values())
        ledger = build_arrival_ledger(fleets, planets, horizon)
        timelines = {
            p.id: simulate_planet_timeline(p, ledger[p.id], horizon) for p in planets
        }
        return cls(ledger=ledger, timelines=timelines, horizon=horizon)

    def owner_at(self, planet_id: int, step) -> int | None:
        """Predicted owner of `planet_id` at `step` from now (None if unknown)."""
        tl = self.timelines.get(planet_id)
        if tl is None:
            return None
        return state_at_timeline(tl, step)[0]

    def ships_at(self, planet_id: int, step) -> float | None:
        """Predicted garrison of `planet_id` at `step` from now (None if unknown)."""
        tl = self.timelines.get(planet_id)
        if tl is None:
            return None
        return state_at_timeline(tl, step)[1]

    def incoming_enemy_eta(self, planet_id: int, my_id: int) -> int | None:
        """Min ETA among in-flight fleets owned by a non-`my_id` player
        currently targeting `planet_id`. None if no enemy fleet is
        inbound within the horizon.

        Used by the source-drain mission to gate "is this planet safe
        to empty"; safe iff `incoming_enemy_eta is None or eta >
        our_attack_eta + buffer`."""
        arrivals = self.ledger.get(planet_id)
        if not arrivals:
            return None
        enemy_etas = [eta for (eta, owner, ships) in arrivals if owner != my_id and ships > 0]
        if not enemy_etas:
            return None
        return min(enemy_etas)


# ---------------------------------------------------------------------------
# Comet lifetime — public helper used by ROI scoring sites
# ---------------------------------------------------------------------------


def _comet_paths_by_id(world) -> dict[int, tuple[list, int]]:
    """{planet_id: (path, path_index)} for every comet in `world.obs_raw`.

    `obs["comets"]` is a list of groups, each `{planet_ids, paths,
    path_index}`. `paths[i]` is the trajectory of `planet_ids[i]` — a
    list of `[x, y]` pairs. `path_index` is shared across the group.

    Mirrors `lib/mechanism._comet_path_lookup` but promoted to a public
    helper because ROI scoring now needs it as well.
    """
    raw = world.obs_raw
    if raw is None:
        return {}
    if isinstance(raw, dict):
        comets = raw.get("comets", [])
    else:
        comets = getattr(raw, "comets", [])
    out: dict[int, tuple[list, int]] = {}
    for group in comets or []:
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


def comet_remaining_lifetime(planet_id: int, world) -> int | None:
    """Steps until `planet_id` leaves the board.

    Returns `len(path) - path_index` for comets, or `None` for non-comet
    planets (which have no finite lifetime in this sense — the static /
    orbiting planets stay until end-of-game).

    Used by ROI scoring sites (`lib/missions/snipe.py`,
    `agents/simple/roi.py`, `agents/v2/main.py`) to cap `time_to_hold`
    on comet targets: sending a fleet to a comet that leaves before we
    arrive is wasted ships.
    """
    paths_by_id = _comet_paths_by_id(world)
    entry = paths_by_id.get(int(planet_id))
    if entry is None:
        return None
    path, path_index = entry
    return max(0, len(path) - path_index)

# === inlined: lib/intent.py ===


from dataclasses import dataclass

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


@dataclass
class Intent:
    """A strategy's request for a single fleet launch.

    `ships` is the strategy's *desired* size; `arrival_size` (the mechanism)
    may revise it upward to account for production growth during flight.
    `aim_angle` starts None and is populated by `lead_aim` / `comet_aim` /
    `lead_aim_v2`. `arrival_xy` is populated by `lead_aim_v2` (and by
    `comet_aim` when re-enabled) so downstream mechanisms (`sun_avoid`,
    `path_clears_other_planets`, `oob_guard`) can check the actual fleet
    path endpoint rather than the target's current position. Mechanisms
    may also drop intents (e.g. `validate`, `sun_avoid` when no detour
    exists).
    """
    src_id: int
    target_id: int
    ships: int
    aim_angle: float | None = None
    arrival_xy: tuple[float, float] | None = None
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


def realize(intents, obs, *, mechanisms, model=None, reasons=None) -> list[list]:
    """Apply the mechanism pipeline and emit env-format actions.

    Final emission to `[src_id, aim_angle, ships]` lists is hard-coded —
    NOT a user-pluggable mechanism. Intents missing `aim_angle` or with
    `ships <= 0` after the pipeline are silently dropped.

    `model` is the per-turn WorldModel snapshot. Mechanisms that need it
    (e.g. `arrival_size` to size against adversary stacking) accept a
    3-arg signature; mechanisms that don't are still called as
    `(intents, world)` for backwards-compatibility.

    Idle-source tracing (opt-in): pass a `reasons` dict to receive
    per-source `MECHANISM_DROP:<mech_name>` attributions for intents
    dropped by any mechanism or by the final emit filter. Pairs with
    `lib.planner.settle_plan`'s reasons capture; downstream tooling
    can merge both dicts to bucket every idle source.
    """
    world = World.from_obs(obs)
    for m in mechanisms:
        code = getattr(m, "__code__", None)
        if reasons is not None:
            srcs_before = {i.src_id for i in intents}
        if code is not None and code.co_argcount >= 3:
            intents = m(intents, world, model)
        else:
            intents = m(intents, world)
        if reasons is not None:
            srcs_after = {i.src_id for i in intents}
            mech_name = getattr(m, "__name__", "unknown")
            for dropped in srcs_before - srcs_after:
                reasons[dropped] = f"MECHANISM_DROP:{mech_name}"
    out = []
    for i in intents:
        if i.ships > 0 and i.aim_angle is not None:
            out.append([i.src_id, i.aim_angle, i.ships])
        elif reasons is not None:
            if i.aim_angle is None:
                reasons[i.src_id] = "MECHANISM_DROP:final_emit_no_aim"
            else:
                reasons[i.src_id] = "MECHANISM_DROP:final_emit_zero_ships"
    return out

# === inlined: lib/trajectory.py ===


import math
from dataclasses import dataclass

fleet_speed = speed

# Max steps we simulate before giving up. A 1-ship fleet at speed 1.0
# can cross the 141.4-unit board diagonal in 142 steps; 200 covers
# every realistic case with comfortable margin.
DEFAULT_MAX_STEPS = 200

# Safety margin around the sun (units). The env's sun-check uses
# point-to-segment distance; we add a 0.5-unit cushion so float drift
# on tangent paths doesn't flip the verdict between sim and reality.
SUN_SAFETY = 0.5


@dataclass(frozen=True)
class FleetFate:
    outcome: str               # "target" | "planet" | "sun" | "oob" | "timeout"
    hit_planet_id: int | None  # set when outcome in {"target", "planet"}
    step: int                  # 1-based step at which the event occurred


def predict_fleet_fate(
    src, target, aim_angle: float, ships: int,
    world, max_steps: int = DEFAULT_MAX_STEPS,
) -> FleetFate:
    """Ray-cast a fleet's full trajectory until the first collision.

    Walks the fleet forward at `fleet_speed(ships)` per step, checking
    EACH per-step segment against:

    1. The sun (continuous point-to-segment distance to (CENTER, CENTER),
       with `SUN_SAFETY` cushion).
    2. The OOB box edges (segment endpoint outside [0, BOARD_SIZE]).
    3. Every planet's per-step swept segment (orbital chord for orbiting
       planets, constant position for static). Uses the env-mirroring
       `swept_pair_hit` so the prediction matches the env's collision
       resolution.

    Returns the FIRST hit. If we reach `max_steps` without collision the
    fate is `"timeout"` — should be rare on a 100x100 board.

    O(max_steps * planets) per call. On a 24-planet mid-game board with
    max_steps=200 that's ~4800 swept_pair_hit calls = ~1-2 ms.
    """
    omega = world.omega

    # Spawn position (env: src.center + (radius + 0.1) * direction).
    cos_a = math.cos(aim_angle)
    sin_a = math.sin(aim_angle)
    spawn_x = src.x + cos_a * (src.radius + 0.1)
    spawn_y = src.y + sin_a * (src.radius + 0.1)
    speed_val = fleet_speed(ships)
    if speed_val <= 0:
        # Shouldn't happen (fleet_speed is monotonically >= 1.0 for ships >= 1).
        return FleetFate("oob", None, 0)

    # Pre-compute per-planet positions at every step (orbital chord).
    planet_positions: dict[int, list[tuple[float, float]]] = {}
    for pid, p in world.planets_by_id.items():
        p_tuple = [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
        if is_orbiting(p_tuple) and omega != 0.0:
            planet_positions[pid] = [
                predict_relative(p_tuple, omega, t)
                for t in range(max_steps + 1)
            ]
        else:
            planet_positions[pid] = [(p.x, p.y)] * (max_steps + 1)

    target_id = target.id
    src_id = src.id
    for step in range(max_steps):
        fleet_old = (
            spawn_x + cos_a * speed_val * step,
            spawn_y + sin_a * speed_val * step,
        )
        fleet_new = (
            spawn_x + cos_a * speed_val * (step + 1),
            spawn_y + sin_a * speed_val * (step + 1),
        )

        # 1. Sun check — point-to-segment distance.
        sun_d = _segment_to_point_distance(fleet_old, fleet_new, (CENTER, CENTER))
        if sun_d < SUN_RADIUS + SUN_SAFETY:
            return FleetFate("sun", None, step + 1)

        # 2. OOB check — segment endpoint outside the box.
        if (
            fleet_new[0] < 0.0 or fleet_new[0] > BOARD_SIZE
            or fleet_new[1] < 0.0 or fleet_new[1] > BOARD_SIZE
        ):
            return FleetFate("oob", None, step + 1)

        # 3. Planet collision — swept_pair_hit against every planet.
        for pid, positions in planet_positions.items():
            # Spawn-step skip: env explicitly does not collide a fresh
            # fleet with its source planet on its first move.
            if pid == src_id and step == 0:
                continue
            p_old = positions[step]
            p_new = positions[step + 1]
            prad = world.planets_by_id[pid].radius
            if swept_pair_hit(fleet_old, fleet_new, p_old, p_new, prad):
                outcome = "target" if pid == target_id else "planet"
                return FleetFate(outcome, pid, step + 1)

    return FleetFate("timeout", None, max_steps)


def _segment_to_point_distance(a, b, p) -> float:
    """Shortest distance from segment a->b to point p."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)

# === inlined: lib/mechanism.py ===


import math

fleet_speed = speed


# ---------------------------------------------------------------------------
# gang_up_size constants — v3.6 multi-source coordination (Plan: 7-step
# problem-solving iteration). Off by default; opt-in for A/B.
# ---------------------------------------------------------------------------
# Phase-0 idle-source decomposition (audit/2026-05-11-idle-breakdown-v3-snipe-
# phase0.md) showed ~96% of all idle classifications come from `intent.ships
# > src.ships` in validate + arrival_size — a single source can't fund the
# capture alone. The combat rule (lib/combat.py::resolve_arrivals) confirms
# same-owner same-step arrivals sum ships before combat resolution, so two
# small sources arriving simultaneously CAN cover a target neither alone
# could. `gang_up_size` is a new mechanism that runs BEFORE `validate` so
# unaffordable single-source intents survive long enough to be paired.
GANG_UP_ENABLED = 0              # default OFF; opt-in for A/B
GANG_UP_ETA_TOLERANCE = 0        # ±turns allowed in shared-eta match
GANG_UP_MIN_SHARE_THRESHOLD = 2  # min sources to form a gang
GANG_UP_RESERVE = 0              # garrison kept home per source (defense)
GANG_UP_MAX_PASSES = 3           # convergence safety belt


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


def arrival_size(intents: list[Intent], world: World, model=None) -> list[Intent]:
    """Bump `ships` so an enemy-owned target's expected garrison at arrival
    is covered, accounting for in-flight adversary fleet stacking.

    Two sources for the "expected garrison at arrival":
    1. **Static estimate** (always available): `target.ships +
       target.production * eta + 1`. Assumes no enemy fleets reach the
       target before we do.
    2. **WorldModel estimate** (when `model` is provided): the simulator
       in `lib/world_model.py` already integrates in-flight adversary
       fleets and same-step combat into `ships_at(target_id, eta)`. This
       is the fix for the v3_snipe bounce-rate doubling
       (audit/2026-05-11-v3-snipe-critical-review.md §4.1): without the
       model, a two-attacker stack walking into our target leaves our
       arriving fleet under-sized by exactly the second attacker's count.

    We take `max(static, model)` so we never go below the static estimate
    (defensive against WorldModel mis-predictions for orbiting planets,
    `lib/world_model.py:46-51`). If `owner_at(target, eta) == world.my_id`
    the planet flips to us en route — drop the intent.

    Neutral targets and our own planets are pass-through.

    The bump is monotonic. If even our full garrison can't cover the
    needed size, drop — sending an under-sized fleet is pure waste.
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
        # Targeted off-by-one for dynamic targets (orbiting + comets):
        # the swept-pair collision check resolves combat at the entry-turn
        # position, which is one production tick AFTER the production tick
        # computed at eta. For static planets, eta is over-estimated by
        # (r_src + r_target)/v (fleet captures on radius-entry), so adding
        # the extra tick over-sizes. Audit:
        # `audit/2026-05-11-v3-snipe-games-analysis.md` items A + games §5.
        target_tuple = [
            target.id, target.owner, target.x, target.y,
            target.radius, target.ships, target.production,
        ]
        # is_orbiting() is geometric (target sits inside the rotation
        # radius); a target actually MOVES only when world.omega != 0.
        # Match lead_aim's gate (mechanism.py:254) so static-from-zero-omega
        # tests stay unchanged.
        is_dynamic = (
            target.id in world.comet_ids
            or (is_orbiting(target_tuple) and world.omega != 0.0)
        )
        prod_ticks = eta + (1 if is_dynamic else 0)
        static_needed = target.ships + target.production * prod_ticks + 1
        needed = static_needed
        if model is not None:
            pred_owner = model.owner_at(target.id, eta)
            if pred_owner == world.my_id:
                # Already ours by then — let the planner skip.
                continue
            pred_ships = model.ships_at(target.id, eta)
            if pred_ships is not None:
                needed = max(static_needed, int(math.ceil(pred_ships)) + 1)
        intent.ships = max(intent.ships, needed)
        if intent.ships > src.ships:
            continue
        out.append(intent)
    return out


# ---------------------------------------------------------------------------
# gang_up_size — multi-source coordination (v3.6)
# ---------------------------------------------------------------------------


def _max_ships_for_eta(distance: float, target_eta: int) -> int:
    """Return the largest ship count whose fleet_speed yields eta == target_eta.

    `fleet_speed(s)` is monotone non-decreasing in s and bounded above
    (max_speed). To get a specific eta we want the LARGEST s such that
    `ceil(distance / fleet_speed(s)) <= target_eta`. Larger ships → faster →
    smaller eta, so we binary-search the upper bound.

    Returns 1 if even 1 ship would still beat target_eta (target_eta is too
    generous; caller should accept the lone ship which arrives earlier).
    Returns 1000 (the saturation point of fleet_speed) if even max_speed
    can't reach target_eta — caller will need to widen tolerance.
    """
    if target_eta <= 0:
        return 1
    lo, hi = 1, 1000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        v = fleet_speed(mid)
        eta = math.ceil(distance / v) if v > 0 else target_eta + 1
        if eta <= target_eta:
            lo = mid
        else:
            hi = mid - 1
    return lo


def gang_up_size(
    intents: list[Intent], world: World, model: WorldModel | None = None,
) -> list[Intent]:
    """Coordinate multiple this-turn intents at the same target so their
    combined ships cover the predicted garrison.

    Default-off (`GANG_UP_ENABLED = 0`): pure pass-through. When enabled,
    runs BEFORE `validate` in `DEFAULT_MECHANISMS` so unaffordable single-
    source intents (which `validate` would otherwise drop on `intent.ships
    > src.ships`) survive long enough to be paired with siblings.

    Algorithm per target group of size >= GANG_UP_MIN_SHARE_THRESHOLD:
    1. Anchor eta = max(eta_solo) across the group. Slower sources can't
       speed up by sending more ships (fleet_speed is bounded), but
       faster sources CAN slow down by sending fewer ships.
    2. needed_total = max(static_at_anchor + 1, model.ships_at(target,
       anchor) + 1). Static = target.ships + production*anchor + 1.
    3. share_i proportional to src_i.ships, capped at src_i.ships -
       GANG_UP_RESERVE. Throttled DOWN so the source arrives at anchor
       (via _max_ships_for_eta).
    4. Up to GANG_UP_MAX_PASSES iterations: any source whose share_i
       implies a higher eta than the anchor → re-anchor up. Cap at 3;
       on failure to converge, drop the gang group (per-intent
       arrival_size handles them individually as today).
    5. Sources with share_i < 1 after capping are dropped from the gang
       and redistributed. If survivors == 1, the lone intent exits
       gang-up unmodified (preserves sole-source bit-identity).

    Single-intent targets pass through unmodified — sole-source path is
    a no-op (assertion enforced by test_gangup_sole_source_noop).
    """
    if not GANG_UP_ENABLED or not intents:
        return intents

    # Bucket intents by target_id.
    by_target: dict[int, list[Intent]] = {}
    for intent in intents:
        by_target.setdefault(intent.target_id, []).append(intent)

    # Process each multi-source group; collect modified intents.
    modified_intent_ids: set[int] = set()
    for target_id, group in by_target.items():
        if len(group) < GANG_UP_MIN_SHARE_THRESHOLD:
            continue
        target = world.planets_by_id.get(target_id)
        if target is None:
            continue
        # Neutrals don't grow during flight; gang-up is mostly relevant
        # for enemy targets, but we still allow it for neutrals when a
        # single source can't afford the static cost (very rare but
        # possible for early-game large neutrals).
        # Compute per-source eta_solo using their CURRENT intent.ships.
        infos = []
        for intent in group:
            src = world.planets_by_id.get(intent.src_id)
            if src is None:
                continue
            d = math.hypot(target.x - src.x, target.y - src.y)
            v = fleet_speed(intent.ships)
            eta_solo = math.ceil(d / v) if v > 0 else 0
            infos.append({
                "intent": intent, "src": src, "distance": d, "eta": eta_solo,
            })
        if len(infos) < GANG_UP_MIN_SHARE_THRESHOLD:
            continue

        # Convergence loop: anchor on slowest source; shrink faster
        # siblings to match; re-check until stable (or give up).
        converged = False
        for _ in range(GANG_UP_MAX_PASSES):
            anchor_eta = max(info["eta"] for info in infos)
            # needed_total at anchor_eta.
            if target.owner == -1 or target.owner == world.my_id:
                # Neutrals & own planets: no production growth during
                # flight; needed is just target.ships + 1.
                needed_total = max(1, int(target.ships) + 1)
            else:
                static_needed = (
                    int(target.ships)
                    + int(target.production) * anchor_eta
                    + 1
                )
                needed_total = static_needed
                if model is not None:
                    pred_ships = model.ships_at(target.id, anchor_eta)
                    if pred_ships is not None:
                        needed_total = max(
                            static_needed, int(math.ceil(pred_ships)) + 1,
                        )

            # Allocate shares proportional to src.ships, capped by
            # src.ships - reserve.
            total_src_ships = sum(info["src"].ships for info in infos)
            if total_src_ships <= 0:
                converged = True
                break

            new_etas = []
            for info in infos:
                raw_share = math.ceil(
                    needed_total * info["src"].ships / total_src_ships
                )
                cap = max(0, info["src"].ships - GANG_UP_RESERVE)
                # Also cap by _max_ships_for_eta so this source actually
                # arrives at anchor_eta (slower fleets need fewer ships).
                throttle = _max_ships_for_eta(info["distance"], anchor_eta)
                share = min(raw_share, cap, throttle)
                info["share"] = max(0, share)
                # Recompute eta with new share.
                v = fleet_speed(max(1, info["share"]))
                new_eta = math.ceil(info["distance"] / v) if v > 0 else 0
                new_etas.append(new_eta)

            new_anchor = max(new_etas)
            if new_anchor <= anchor_eta + GANG_UP_ETA_TOLERANCE:
                # Update anchor and apply throttles definitively.
                converged = True
                # Apply shares to intents; drop any source with share==0
                # from the gang (re-routed to per-intent arrival_size).
                gang_share_total = 0
                survivors = []
                for i, info in enumerate(infos):
                    info["eta"] = new_etas[i]
                    if info["share"] <= 0:
                        continue
                    survivors.append(info)
                    gang_share_total += info["share"]
                # If combined survivors still cover needed AND we have
                # ≥ min sources, write the throttled ships back to each
                # intent. Otherwise drop the gang (fall back to per-
                # intent arrival_size handling).
                if (
                    len(survivors) >= GANG_UP_MIN_SHARE_THRESHOLD
                    and gang_share_total >= needed_total
                ):
                    for info in survivors:
                        info["intent"].ships = int(info["share"])
                        modified_intent_ids.add(id(info["intent"]))
                break
            # Re-anchor on the new max eta and retry.
            for i, info in enumerate(infos):
                info["eta"] = new_etas[i]

        # If we ran out of passes without converging, leave intents
        # unmodified — per-intent arrival_size will drop them as today.
        _ = converged

    return intents


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

    2026-05-11 rewrite: now uses `lib.trajectory.predict_fleet_fate` to
    ray-cast the FULL fleet trajectory (not just the segment up to the
    predicted target arrival point). Previous endpoint-only check missed
    sun collisions in the trajectory's overshoot tail when the lead
    prediction misses (orbital drift, tangent shot). Live-replay
    evidence: 3.2% of our fleets died in the sun under the old guard.

    Drop-only — re-routing via a waypoint planet is a v3 mission concern.
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
        fate = predict_fleet_fate(src, target, intent.aim_angle, intent.ships, world)
        if fate.outcome == "sun":
            continue
        out.append(intent)
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

    2026-05-11 rewrite: delegates to `lib.trajectory.predict_fleet_fate`.
    Previous impl simulated only up to the predicted target arrival
    step (~total_dist / speed); the overshoot tail wasn't checked. Now
    we walk the full trajectory.

    Capture-probe (2026-05-10) showed 10.7% non-target-planet collisions
    as the biggest physics-loss bucket. Live-replay (2026-05-11) showed
    the residual was still significant because of the truncated-horizon
    bug. Full-trajectory ray-cast closes that gap.
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
        fate = predict_fleet_fate(src, target, intent.aim_angle, intent.ships, world)
        if fate.outcome == "planet":
            continue
        out.append(intent)
    return out


# ---------------------------------------------------------------------------
# oob_guard — drop intents whose projected endpoint exits the board
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# arrival_ledger — skip intents whose target will already be ours at arrival
# ---------------------------------------------------------------------------


def arrival_ledger(intents: list[Intent], world: World) -> list[Intent]:
    """Drop intents we don't need: target will be ours with enough ships
    at our arrival step.

    Builds a `WorldModel` snapshot (in-flight fleet arrival ledger +
    per-planet timeline) for this turn. For each intent:
    - Estimate arrival step via straight-line dist / fleet_speed.
    - Look up `(predicted_owner, predicted_ships)` at that step.
    - If predicted_owner == us AND predicted_ships >= intent.ships,
      drop the intent — adding another fleet would double-commit.

    Stronger variants (intercept enemy arrivals, gang-up timing) live
    in v3 mission classes; this is the minimum-viable v2 use case.

    Cost: O(planets * horizon + fleets * planets) per turn for the
    WorldModel build (~5 ms on a 40-planet board). Cached for the
    duration of one mechanism call.
    """
    if not intents:
        return intents
    wm = WorldModel.from_world(world)
    out: list[Intent] = []
    for intent in intents:
        src = world.planets_by_id.get(intent.src_id)
        target = world.planets_by_id.get(intent.target_id)
        if src is None or target is None:
            out.append(intent)
            continue
        # ETA: straight-line center-to-center / fleet_speed(intent.ships).
        # Rough — doesn't account for orbital motion of target. Adequate
        # for the "don't double-commit" use case.
        d = math.hypot(target.x - src.x, target.y - src.y)
        v = fleet_speed(intent.ships)
        eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
        pred_owner = wm.owner_at(target.id, eta)
        pred_ships = wm.ships_at(target.id, eta)
        if (
            pred_owner == world.my_id
            and pred_ships is not None
            and pred_ships >= intent.ships
        ):
            # Already going to be ours with surplus garrison; ship would
            # be wasted on a target we're about to own anyway.
            continue
        out.append(intent)
    return out


def oob_guard(intents: list[Intent], world: World) -> list[Intent]:
    """Drop intents whose fleet path would exit the [0, BOARD_SIZE] box
    before colliding with anything.

    2026-05-11 rewrite: delegates to `lib.trajectory.predict_fleet_fate`.
    Previous impl checked only the predicted endpoint; if the lead-
    prediction missed the target the fleet kept flying past it through
    empty space until OOB, but the guard didn't see that overshoot tail.

    Live-replay evidence (audit/live-episodes/52532938/): 7.5% of our
    fleets flew OOB under the old guard, including one 7-ship fleet
    that travelled 79 units through empty space before exiting the
    board (the predicted target had moved by the time we arrived).
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
        fate = predict_fleet_fate(src, target, intent.aim_angle, intent.ships, world)
        # Drop both `oob` and `timeout` outcomes: a 200-step ray-cast
        # without collision means the fleet doesn't reach anything
        # useful — same effective waste as flying OOB.
        if fate.outcome in ("oob", "timeout"):
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
    gang_up_size,                 # v3.6: no-op when GANG_UP_ENABLED=0
    validate,
    arrival_size,
    lead_aim_v2,
    sun_avoid,
    path_clears_other_planets,
    oob_guard,
]
# `arrival_ledger` is implemented but EXCLUDED from DEFAULT_MECHANISMS.
# Local A/B showed it regressed WR from 56% to 50% (Block C audit) because
# per-source greedy strategies don't re-pick after the mechanism drops an
# intent: the source planet ends the turn with no action. The mechanism's
# real value materialises when paired with the v3 planner (Block D), which
# can re-allocate the freed ships to a different target/mission. Keep here
# for direct use from the planner; do NOT add to DEFAULT until then.

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
    "gang_up_size",
    "comet_aim",
    "lead_aim",
    "lead_aim_v2",
    "sun_avoid",
    "path_clears_other_planets",
    "oob_guard",
    "arrival_ledger",
]

# === inlined: lib/mission.py ===


from dataclasses import dataclass



@dataclass
class Mission:
    """A typed fleet-launch candidate."""

    mission_class: str
    src_id: int
    target_id: int
    ships: int
    score: float
    eta: int               # estimated turns to arrival; for the planner's
                           # this-turn arrival-ledger tracking
    note: str = ""

    def to_intent(self) -> Intent:
        """Drop to the strategy/mechanism contract."""
        note = (
            f"{self.mission_class}:{self.note}" if self.note
            else self.mission_class
        )
        return Intent(
            src_id=self.src_id,
            target_id=self.target_id,
            ships=self.ships,
            note=note,
        )

# === inlined: lib/missions/snipe.py ===


import math

fleet_speed = speed

# sym_hypot was imported here for the σ-equiv layer (cherry-picked
# from origin/claude/game-theory-strategy-analysis-0oH4N). REVERTED for
# v9 (2026-05-12) — v7.6 bisect found σ-equiv regresses v7_0 by ~54pp.
# Restoring math.hypot for src↔target distance below.

# Total game length in steps (Configuration table, data/README.md).
EPISODE_STEPS = 500

# Priority multipliers (calibrated from games analysis).
# NEUTRAL_BONUS and COMET_BONUS were attempted at 1.5 / 1.3 but regressed in
# 32-seed 2P A/B (28.1% Wilson [18.6%, 40.1%]); they tipped the scorer toward
# easy neutrals when contested enemy planets were the binding constraint.
# Disabled (= 1.0) pending a more selective heuristic (opening-only, or
# distance-conditioned). See audit/2026-05-11-v3-snipe-games-analysis.md.
NEUTRAL_BONUS = 1.0
COMET_BONUS = 1.0
# LEADER_MULTIPLIER only fires when our_rank >= 2 (4P/larger games where we
# are below 2nd place). 2P games are unaffected. Pending 4P FFA validation.
LEADER_MULTIPLIER = 1.5

# Airtime penalty (v3.5, 2026-05-11): ships in flight are committed-cost.
# A fleet en route can't defend its home planet, can't be redirected, and
# may bounce if the world-state has shifted. Phase-0 idle-source decomposition
# (audit/2026-05-11-idle-breakdown-v3-snipe-phase0.md) showed ~96% of all
# idle classifications come from `intent.ships > src.ships` in validate +
# arrival_size, and the worst offenders are LONG-eta targets where
# arrival_size's `target.ships + production * eta + 1` over-estimates the
# source garrison. Penalising airtime in the score formula shifts target
# selection toward closer (lower-eta) captures, reducing both opportunity
# cost AND the dominant mechanism-drop bucket.
#
# Coefficient interpretation: adds `AIRTIME_PENALTY_WEIGHT * eta` to the
# denominator. eta is bounded in [1, ~30] for the 100x100 board, so at
# weight=1.0 the penalty caps at ~30 vs typical denominators of 50-150 — a
# moderate soft penalty.
#
# **v3.5 A/B verdict (audit/2026-05-11-v3.5-airtime-and-endgame-burn.md):**
# - AIRTIME=1.0 regresses heavily vs v3.4 baseline (43.8% Wilson at 32-seed).
# - AIRTIME=0.5 looked like +4.7pp lift at 32-seed but converged to 52.3%
#   Wilson=[43.7, 60.8] at 64-seed — statistically indistinguishable from
#   baseline.
# - Default reverted to 0.0 (identity). Constant kept for future research
#   (e.g., phase-decay variant, src-conditional variant, multiplicative form).
AIRTIME_PENALTY_WEIGHT = 0.0

# Endgame burn (v3.5, Exp 1): in the final ~30 turns of a game, neutrals
# matter more than enemy captures because (a) neutrals don't grow ships
# (no arrival_size bump → reliably launchable), (b) we have little time
# left to extract production value from contested captures. Boost neutral
# target priority by ENDGAME_NEUTRAL_BONUS once step >= ENDGAME_STEP.
#
# **v3.5 A/B verdict:** as part of the airtime+endgame composite at 64-seed,
# the lift was indistinguishable from baseline. Standalone (eg_only, no
# airtime) saw 40 draws / 64 games = stalemate. Default reverted to 1.0
# (identity). Constant kept for future research (e.g., size-conditional
# burn, neutrals-near-source-only).
ENDGAME_STEP = 470
ENDGAME_NEUTRAL_BONUS = 1.0

# Affordability filter (v3.5+): when True, propose a Mission only if the
# source planet can fund the base capture (target.ships + 1) ALONE. Phase-0
# idle-trace showed ~45% of all idle classifications are
# MECHANISM_DROP:validate, which fires on `intent.ships > src.ships`.
# Filtering at proposal time lets the source's runner-up affordable target
# win settle_plan's per-source greedy instead of being silently dropped
# downstream. Drawback: blocks gang-up (multiple sources contributing to
# one target) — but gang-up doesn't actually work today (each intent is
# independently sized by arrival_size), so the filter is a near-pure
# improvement to idle rate. Default OFF (= 0) until validated by A/B.
# Stored as int so scripts/ab_variants.py can patch it (its regex requires
# a numeric literal).
PROPOSER_AFFORDABILITY_FILTER = 0


def _player_totals(world: World) -> dict[int, float]:
    """Aggregate ships across planets + in-flight fleets for each player.

    Used by the 4P spoiler logic to identify the current leader.
    """
    totals: dict[int, float] = {}
    for p in world.planets_by_id.values():
        if p.owner == -1:
            continue
        totals[p.owner] = totals.get(p.owner, 0) + p.ships
    raw = world.obs_raw
    fleets_raw = (
        raw.get("fleets", []) if isinstance(raw, dict) else getattr(raw, "fleets", [])
    )
    for f in fleets_raw:
        # Fleet schema: [id, owner, x, y, angle, from_planet_id, ships].
        owner = f[1]
        ships = f[6]
        if owner == -1:
            continue
        totals[owner] = totals.get(owner, 0) + ships
    return totals


def _leader_pid(world: World) -> tuple[int | None, int | None]:
    """Return (leader_pid, our_rank) for 4P spoiler scoring.

    Rank is 0-indexed (0 = leader). If we're alone or only-vs-one
    other player, returns (None, None) — no spoiler applies in 2P.
    """
    totals = _player_totals(world)
    if len(totals) < 3:
        return None, None  # 2P or solo — no spoiler
    ordered = sorted(totals.items(), key=lambda kv: -kv[1])
    leader_pid = ordered[0][0]
    our_rank = None
    for i, (pid, _ships) in enumerate(ordered):
        if pid == world.my_id:
            our_rank = i
            break
    return leader_pid, our_rank


# Aggressive sizing (added 2026-05-12 for v3.5.1):
# Top-10 fingerprint analysis (knowledge-base/concepts/top-performer-strategies.md)
# shows mean fleet 38 vs midpack 29 (+33%) and mean garrison-at-launch 11
# vs midpack 22 (half). Translating: top-10 sends a higher FRACTION of
# source garrison per launch. When `aggressive=True` and the source has
# more than AGGRESSIVE_MIN_GARRISON ships, base_ships is set to
# `min(src.ships * AGGRESSIVE_FRACTION, src.ships - AGGRESSIVE_RESERVE)`
# capped above by target_min — so we always send at least what's needed
# to capture, and at most a fixed fraction of garrison.
#
# Parameter sweep (audit/tournaments/sizing-sweep-20260512T044157Z.json):
# 0.7 dominates 0.6 / 0.8 / 0.9 in both vs-baseline winrate and
# head-to-head. 32-seed 2P A/B vs v3_snipe: 68.8% Wilson lo 56.6% [PASS].
# 8-seed × 4-seat 4P FFA vs weak background: 96.9% (vs v3_snipe baseline
# 93.8% in same panel).
AGGRESSIVE_FRACTION = 0.7
AGGRESSIVE_RESERVE = 5
AGGRESSIVE_MIN_GARRISON = 12


def propose_snipe_missions(
    world: World,
    model: WorldModel,
    aggressive: bool = False,
) -> list[Mission]:
    """Build one snipe Mission per (our source, non-our target) pair.

    `aggressive=False` (default) uses the v3.4 minimum-viable formula
    `max(1, t.ships + 1)` — preserves the parity-gated v3_snipe bundle.
    `aggressive=True` uses the top-10-aligned sizing formula. v3.5.1
    is the first agent to pass aggressive=True.
    """
    if not world.planets_by_id:
        return []
    my_planets = [
        p for p in world.planets_by_id.values() if p.owner == world.my_id
    ]
    if not my_planets:
        return []
    targets = [
        p for p in world.planets_by_id.values() if p.owner != world.my_id
    ]
    if not targets:
        return []

    step_now = int(world.step)
    leader_pid, our_rank = _leader_pid(world)
    spoiler_on = leader_pid is not None and our_rank is not None and our_rank >= 2

    missions: list[Mission] = []
    for src in my_planets:
        for t in targets:
            d = math.hypot(t.x - src.x, t.y - src.y)
            target_min = max(1, int(t.ships) + 1)
            if aggressive and src.ships > AGGRESSIVE_MIN_GARRISON:
                fraction_size = max(1, int(src.ships * AGGRESSIVE_FRACTION))
                cap = max(1, int(src.ships) - AGGRESSIVE_RESERVE)
                base_ships = max(target_min, min(fraction_size, cap))
            else:
                base_ships = target_min
            if PROPOSER_AFFORDABILITY_FILTER and base_ships > src.ships:
                # Source can't fund this capture alone; let its smaller
                # affordable runner-up win settle_plan's per-source greedy.
                # OFF by default (regressed in 64-seed A/B); kept for
                # future ablation. See main's optimize-ship-strategy-tDPXx.
                continue
            v = fleet_speed(base_ships)
            eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
            pred_owner = model.owner_at(t.id, eta)
            pred_ships = model.ships_at(t.id, eta) or 0.0
            if pred_owner == world.my_id and pred_ships >= base_ships:
                # Target will be ours with surplus garrison; redundant.
                continue
            # Comet-lifetime correction: comets leave the board at
            # `len(path) - path_index` steps from now; capping time_to_hold
            # by remaining lifetime stops us scoring "long-run yield" on a
            # comet that's about to depart.
            is_comet = t.id in world.comet_ids
            if is_comet:
                rem = comet_remaining_lifetime(t.id, world)
                time_to_hold = max(0, (rem or 0) - eta)
            else:
                time_to_hold = max(1, EPISODE_STEPS - step_now - eta)
            value = t.production * time_to_hold

            # Cost-aware ROI baseline + priority modifiers.
            priority = 1.0
            if t.owner == -1:
                # Unclaimed: no garrison growth during flight, no opponent
                # competition. Bonus reflects the easier capture.
                priority *= COMET_BONUS if is_comet else NEUTRAL_BONUS
                if step_now >= ENDGAME_STEP:
                    # Late-game burn: neutrals stay launchable (no
                    # production growth → no arrival_size bump), so prefer
                    # them over high-growth enemy captures we likely can't
                    # afford in the remaining turn budget.
                    priority *= ENDGAME_NEUTRAL_BONUS
            if spoiler_on and t.owner == leader_pid:
                priority *= LEADER_MULTIPLIER
            # Cost-aware ROI denominator (legacy) + optional airtime term.
            # - base_ships + d + 1: original v3.4 form. Wave-1b's
            #   `0.5 × base_ships` rebalance was NEUTRAL at 50% in phys-only
            #   A/B (audit/2026-05-12-v3.5-stack-results.md); reverted on
            #   merge to preserve main's parity invariants. v3.5.1's
            #   value driver was the AGGRESSIVE_FRACTION ship sizing, not
            #   this denominator.
            # - AIRTIME_PENALTY_WEIGHT × eta: optional discount for far
            #   targets. Default weight=0 (identity).
            score = priority * value / (
                base_ships + d + AIRTIME_PENALTY_WEIGHT * eta + 1.0
            )

            missions.append(Mission(
                mission_class="snipe",
                src_id=src.id,
                target_id=t.id,
                ships=base_ships,
                score=score,
                eta=eta,
            ))
    return missions

# === inlined: lib/missions/reinforce.py ===


import math

fleet_speed = speed

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
            d_dist = math.hypot(d.x - s.x, d.y - s.y)
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

# === inlined: lib/missions/recapture.py ===


import math

fleet_speed = speed

EPISODE_STEPS = 500
RECAPTURE_WINDOW = 50          # turns after loss within which to recapture
RECAPTURE_BONUS_PEAK = 1.5     # multiplier at the moment of loss
RECENTLY_LOST_GARRISON_MAX = 50  # don't bother recapturing if enemy fortified
# Calibration fixes ported with the file from origin/main (post-revert).
# The 200-game A/B that triggered the revert (audit/...recapture-wireup-ab.md)
# identified three failure modes: score-scale mismatch with snipe, proposal-
# volume dilution (80-160 per turn), and infeasible commits. Knobs below let
# the v7.2 integration A/B re-test with corrected defaults; the gate is
# Wilson lo ≥ 55% at 24 seeds × both sides.
RECAPTURE_SCORE_DENOM_MATCHES_SNIPE = 1  # 1 = use (base_ships + d + 1)
                                          # (snipe-aligned); 0 = legacy
                                          # (0.5*base_ships + d + 1).
RECAPTURE_TOPK_PER_TURN = 5    # cap on proposals returned per turn
                                # (0 = no cap; replicates the regression).


# ---------------------------------------------------------------------------
# Module-level state — persists across turns of a single game.
# Key: planet_id; Value: step_lost (int).
# ---------------------------------------------------------------------------


class _RecaptureState:
    """Mutable state owned by THIS module. Reset when a step-0 obs arrives."""

    def __init__(self):
        self.last_step: int = -1
        self.last_ownership: dict[int, int] = {}
        # planet_id -> step we lost it
        self.lost_at: dict[int, int] = {}

    def reset(self):
        self.last_step = -1
        self.last_ownership = {}
        self.lost_at = {}

    def update(self, world: World) -> None:
        """Compare current planet ownership to last-call snapshot;
        record any planet that was ours and is now an enemy's."""
        step = int(world.step)
        if step == 0 or step < self.last_step:
            self.reset()
        current_ownership = {
            p.id: p.owner for p in world.planets_by_id.values()
        }
        # Detect losses since last call.
        my_id = world.my_id
        for pid, prev_owner in self.last_ownership.items():
            cur_owner = current_ownership.get(pid)
            if cur_owner is None:
                continue
            if prev_owner == my_id and cur_owner != my_id and cur_owner != -1:
                self.lost_at[pid] = step
            # If a planet flipped back to us, drop its lost record.
            if prev_owner != my_id and cur_owner == my_id and pid in self.lost_at:
                del self.lost_at[pid]
        # Also evict lost-records older than RECAPTURE_WINDOW.
        cutoff = step - RECAPTURE_WINDOW
        stale = [pid for pid, s in self.lost_at.items() if s < cutoff]
        for pid in stale:
            del self.lost_at[pid]
        # Update snapshot.
        self.last_step = step
        self.last_ownership = current_ownership


_STATE = _RecaptureState()


def _reset_state_for_tests() -> None:
    """Reset between independent test cases (the module-level state
    bleeds across pytest cases otherwise)."""
    _STATE.reset()


def propose_recapture_missions(world: World, model: WorldModel) -> list[Mission]:
    """One Mission per (recently-lost target, viable source) pair, with
    a time-decaying RECAPTURE_BONUS."""
    _STATE.update(world)
    if not _STATE.lost_at:
        return []
    step_now = int(world.step)
    my_planets = [
        p for p in world.planets_by_id.values() if p.owner == world.my_id
    ]
    if not my_planets:
        return []
    missions: list[Mission] = []
    for lost_pid, step_lost in _STATE.lost_at.items():
        t = world.planets_by_id.get(lost_pid)
        if t is None:
            continue
        # If target is back to us (race condition / mid-update), skip.
        if t.owner == world.my_id:
            continue
        # If target is fortified beyond what we'd consider, skip.
        if t.ships > RECENTLY_LOST_GARRISON_MAX:
            continue
        # Recapture urgency: 1.0 at loss-step → 0.0 at end of window.
        elapsed = step_now - step_lost
        urgency = max(0.0, 1.0 - elapsed / RECAPTURE_WINDOW)
        bonus = 1.0 + (RECAPTURE_BONUS_PEAK - 1.0) * urgency

        for src in my_planets:
            d = math.hypot(t.x - src.x, t.y - src.y)
            base_ships = max(1, int(t.ships) + 1)
            if base_ships >= src.ships:
                # Source can't afford this recapture.
                continue
            v = fleet_speed(base_ships)
            eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
            pred_owner = model.owner_at(t.id, eta)
            if pred_owner == world.my_id:
                continue
            time_to_hold = max(1, EPISODE_STEPS - step_now - eta)
            value = t.production * time_to_hold
            # Denominator: aligned with snipe by default (audit hypothesis
            # 1 fix). Set RECAPTURE_SCORE_DENOM_MATCHES_SNIPE=0 to
            # reproduce the legacy 0.5×base_ships denominator that
            # over-weighted recapture vs snipe in the original A/B.
            if RECAPTURE_SCORE_DENOM_MATCHES_SNIPE:
                denom = base_ships + d + 1.0
            else:
                denom = 0.5 * base_ships + d + 1.0
            score = bonus * value / denom
            missions.append(Mission(
                mission_class="recapture",
                src_id=src.id,
                target_id=t.id,
                ships=base_ships,
                score=score,
                eta=eta,
            ))
    # Audit hypothesis 2 fix: cap per-turn proposal volume so settle_plan
    # isn't drowned in low-value recapture variants. K=5 retains the
    # urgent / high-prod / nearby options; legacy was uncapped (80-160/turn).
    if RECAPTURE_TOPK_PER_TURN > 0 and len(missions) > RECAPTURE_TOPK_PER_TURN:
        missions.sort(key=lambda m: -m.score)
        missions = missions[:RECAPTURE_TOPK_PER_TURN]
    return missions

# === inlined: lib/planner.py ===


from collections import defaultdict



def settle_plan(
    missions: list[Mission],
    world: World,
    model: WorldModel,
    reasons: dict[int, str] | None = None,
) -> list[Intent]:
    """Pick at most one mission per source under a same-turn ledger.

    Algorithm:
    1. Bucket missions by source; sort each bucket by score descending.
    2. Order sources by their top-mission score (highest first).
    3. For each source in order, walk its ranked candidates and accept
       the first one whose target isn't already over-committed by
       prior this-turn picks.
    4. After accepting a mission, register its arrival in the ledger.

    Idle-source tracing (opt-in): pass a `reasons` dict to receive a
    classification of why each non-emitting owned-and-shipped source
    went idle this turn. Keys are planet ids; values are one of:

    - `"NO_PROPOSALS"` — no proposer emitted a Mission for this source.
    - `"LEDGER_LOSS"` — proposer(s) emitted Mission(s) but all were
      skipped because earlier this-turn picks already covered every
      candidate target.

    `MECHANISM_DROP` (intent built but dropped by the realize pipeline)
    is set by `lib.intent.realize`, not here, since this function returns
    before mechanisms run.
    """
    if reasons is None and not missions:
        return []

    by_src: dict[int, list[Mission]] = defaultdict(list)
    for m in missions:
        by_src[m.src_id].append(m)
    # σ-equiv tie-break REVERTED (v7.6 bisect: ~54pp regression of v7_0
    # drop-one architecture). Plain score sort.
    for src_id in by_src:
        by_src[src_id].sort(key=lambda m: -m.score)

    source_order = sorted(
        by_src.keys(),
        key=lambda s: -by_src[s][0].score,
    )

    # target_id -> list of (eta, ships) for this-turn pending arrivals.
    pending: dict[int, list[tuple[int, int]]] = defaultdict(list)
    chosen: list[Mission] = []
    for src_id in source_order:
        selected = False
        for m in by_src[src_id]:
            # Ships our prior this-turn picks have committed to land at
            # m.target_id by step m.eta (or earlier). A defender at
            # T_loss < m.eta will already be neutralised by earlier
            # arrivals; we only need to add to that pool.
            already = sum(
                s for (e, s) in pending[m.target_id] if e <= m.eta
            )
            pred_enemy = model.ships_at(m.target_id, m.eta)
            # For our own planets (reinforce target), the planet may
            # never be "enemy-held" at our arrival; pred_enemy is just
            # the garrison total. We use it as the "size needed to
            # contest" — if our prior picks already supply that much,
            # any additional ships are surplus.
            if pred_enemy is None:
                pred_enemy = 0.0
            # Skip when our prior this-turn picks already exceed
            # (enemy garrison + 1 buffer). The +1 matches the snipe /
            # reinforce ship-sizing convention.
            if already >= pred_enemy + 1:
                continue
            chosen.append(m)
            pending[m.target_id].append((m.eta, m.ships))
            selected = True
            break
        if reasons is not None and not selected and by_src[src_id]:
            reasons[src_id] = "LEDGER_LOSS"

    if reasons is not None:
        chosen_srcs = {m.src_id for m in chosen}
        for p in world.planets_by_id.values():
            if p.owner != world.my_id or p.ships <= 0:
                continue
            if p.id in chosen_srcs or p.id in reasons:
                continue
            reasons[p.id] = "NO_PROPOSALS"

    return [m.to_intent() for m in chosen]

# === inlined: lib/lookahead_planner.py ===


COMET_SPAWN_STEPS: tuple[int, ...] = (50, 150, 250, 350, 450)

# K bounds calibrated to the local box's measured per-step cost (~10 ms
# per forward step including 2x v3.5.1 policy calls). With env_from_obs
# ~105 ms one-time + ~24 ms clone() per candidate, the per-candidate
# wallclock is roughly (24 + 10*K) ms. At K_MAX=12 the budget supports
# ~5 candidates under 1 s; the watchdog in v4_planner cuts later
# candidates if needed. Phase 2 audit measurements (~5.6 ms/step) were
# on a faster 4-core box; these bounds adapt without changing the
# algorithm.
K_MIN = 6
K_MAX = 10


def evaluate_value(
    observation,
    my_id: int,
    *,
    denial_weight: float = 0.4,
    ships_weight: float = 0.05,
    survivor_bonus: float = 5.0,
) -> float:
    """V(s) = prod_share + denial * prod_denied + ships * ships_share + survivor.

    `observation` is the kaggle env's per-seat observation dict (the same
    shape `agent(obs)` receives). Fields used: `planets` (list of
    [id, owner, x, y, radius, ships, production]) and `fleets`
    (list of [id, owner, x, y, angle, from_planet_id, ships]).

    Empty world (no planets) → 0.0. Total production zero (all planets
    have prod=0, which doesn't happen in practice but bounds the math)
    → 0.0 share components, only survivor bonus can fire.
    """
    planets = observation.get("planets", []) if isinstance(observation, dict) else getattr(observation, "planets", [])
    if not planets:
        return 0.0

    total_prod = 0.0
    my_prod = 0.0
    opp_prod = 0.0
    owners_with_planets: set[int] = set()
    garrison: dict[int, float] = {}
    for p in planets:
        owner = int(p[1])
        ships = float(p[5])
        prod = float(p[6])
        total_prod += prod
        if owner >= 0:
            owners_with_planets.add(owner)
            garrison[owner] = garrison.get(owner, 0.0) + ships
            if owner == my_id:
                my_prod += prod
            else:
                opp_prod += prod

    fleets = observation.get("fleets", []) if isinstance(observation, dict) else getattr(observation, "fleets", [])
    fleet_totals: dict[int, float] = {}
    for f in fleets:
        owner = int(f[1])
        ships = float(f[6])
        if owner >= 0:
            fleet_totals[owner] = fleet_totals.get(owner, 0.0) + ships

    totals: dict[int, float] = dict(garrison)
    for owner, ships in fleet_totals.items():
        totals[owner] = totals.get(owner, 0.0) + ships
    total_ships = sum(totals.values())
    my_ships = totals.get(my_id, 0.0)

    prod_share = (my_prod / total_prod) if total_prod > 0 else 0.0
    prod_denied = ((total_prod - opp_prod) / total_prod) if total_prod > 0 else 0.0
    ships_share = (my_ships / total_ships) if total_ships > 0 else 0.0
    lone = 1.0 if owners_with_planets == {my_id} else 0.0

    return (
        prod_share
        + denial_weight * prod_denied
        + ships_weight * ships_share
        + survivor_bonus * lone
    )


def adaptive_K(world) -> int:
    """Entropy-adaptive rollout depth.

    entropy = fleets_in_flight + 0.5 * contested_planets
    K = clamp(round(8 + 1.5 * entropy), 8, 30)

    Empty boards bottom at K=8 (the floor). The 0.5 weighting on neutral
    planets reflects that they're potential rather than active conflict —
    they raise entropy less than an already-launched fleet.
    """
    raw = getattr(world, "obs_raw", None)
    if raw is None:
        return K_MIN
    fleets_raw = (
        raw.get("fleets", []) if isinstance(raw, dict) else getattr(raw, "fleets", [])
    )
    fleets_in_flight = len(fleets_raw)
    contested = 0
    for p in world.planets_by_id.values():
        if p.owner == -1:
            contested += 1
    entropy = fleets_in_flight + 0.5 * contested
    K = int(round(K_MIN + 0.5 * entropy))
    return max(K_MIN, min(K, K_MAX))


def truncate_K_to_comet_boundary(K: int, step: int) -> int:
    """Shorten K so rollout doesn't cross a comet spawn boundary.

    Spawn boundaries are at steps in `COMET_SPAWN_STEPS`. `env_from_obs`
    is bit-exact within an inter-boundary segment; crossing a boundary
    re-rolls the env's comet RNG, which diverges from the real game. We
    cap K so the clone's `step + K` stays strictly below the next
    boundary. Floor at 1 — we always apply at least our chosen action.
    """
    for boundary in COMET_SPAWN_STEPS:
        if boundary > step:
            allowed = boundary - step - 1
            return max(1, min(K, allowed))
    return K

# === inlined: lib/game/interpreter.py ===


import math
import random
from collections import namedtuple

import numpy as np


# Module-level aliases for the math builtins called inside the per-step
# hot loop. Saves the per-call `math.X` attribute lookup; ~12% on the
# 7 M trig/sqrt/log calls measured across a 2000-step random episode.
_cos = math.cos
_sin = math.sin
_sqrt = math.sqrt
_log = math.log


# Threshold below which scalar `swept_pair_hit` beats the numpy batch
# (numpy dispatch overhead exceeds savings). Measured empirically.
_BATCH_FLEETxPLANET_THRESHOLD = 60


def _fleet_planet_first_hits(
    fleets_local: list,
    planets_local: list,
    planet_paths: dict,
    max_speed: float,
    log1000: float,
):
    """For each fleet, return the index of the FIRST planet it collides with
    over this tick (preserves scalar early-break semantics), or -1.

    Vectorises across BOTH fleets and planets in one numpy pass. Falls back
    to a per-fleet scalar loop on tiny workloads where dispatch overhead
    exceeds the savings.

    Bit-exact with the scalar `swept_pair_hit`; parity-gated by
    `tests/test_game_parity.py`.
    """
    n_fleets = len(fleets_local)
    n_planets = len(planets_local)
    if n_fleets * n_planets < _BATCH_FLEETxPLANET_THRESHOLD:
        # Per-fleet scalar fallback; mirrors the original loop's order.
        results: list[int] = [-1] * n_fleets
        for f_idx, fleet in enumerate(fleets_local):
            f2 = fleet[2]; f3 = fleet[3]
            angle = fleet[4]; ships = fleet[6]
            speed = 1.0 + (max_speed - 1.0) * (_log(ships) / log1000) ** 1.5
            if speed > max_speed:
                speed = max_speed
            new_x = f2 + _cos(angle) * speed
            new_y = f3 + _sin(angle) * speed
            for p_idx, planet in enumerate(planets_local):
                path = planet_paths.get(planet[0])
                if path is None or not path[2]:
                    continue
                if swept_pair_hit(
                    (f2, f3), (new_x, new_y), path[0], path[1], planet[4]
                ):
                    results[f_idx] = p_idx
                    break
        return results

    # Build planet arrays (P,).
    pold_x = np.empty(n_planets); pold_y = np.empty(n_planets)
    pnew_x = np.empty(n_planets); pnew_y = np.empty(n_planets)
    pr_arr = np.empty(n_planets)
    pcheck_arr = np.zeros(n_planets, dtype=bool)
    for i, p in enumerate(planets_local):
        path = planet_paths.get(p[0])
        if path is None:
            continue
        pold_x[i] = path[0][0]; pold_y[i] = path[0][1]
        pnew_x[i] = path[1][0]; pnew_y[i] = path[1][1]
        pr_arr[i] = p[4]
        pcheck_arr[i] = path[2]

    # Build fleet arrays (F,) and compute new positions in vectorised form.
    fold_x = np.empty(n_fleets); fold_y = np.empty(n_fleets)
    angles = np.empty(n_fleets); ships_arr = np.empty(n_fleets)
    for i, f in enumerate(fleets_local):
        fold_x[i] = f[2]; fold_y[i] = f[3]
        angles[i] = f[4]; ships_arr[i] = f[6]
    speeds = 1.0 + (max_speed - 1.0) * (np.log(ships_arr) / log1000) ** 1.5
    np.minimum(speeds, max_speed, out=speeds)
    fnew_x = fold_x + np.cos(angles) * speeds
    fnew_y = fold_y + np.sin(angles) * speeds

    # AABB prune across the (F, P) grid.
    f_min_x = np.minimum(fold_x, fnew_x)[:, None]
    f_max_x = np.maximum(fold_x, fnew_x)[:, None]
    f_min_y = np.minimum(fold_y, fnew_y)[:, None]
    f_max_y = np.maximum(fold_y, fnew_y)[:, None]
    p_min_x = (np.minimum(pold_x, pnew_x) - pr_arr)[None, :]
    p_max_x = (np.maximum(pold_x, pnew_x) + pr_arr)[None, :]
    p_min_y = (np.minimum(pold_y, pnew_y) - pr_arr)[None, :]
    p_max_y = (np.maximum(pold_y, pnew_y) + pr_arr)[None, :]
    candidate = (
        pcheck_arr[None, :]
        & (f_max_x >= p_min_x) & (f_min_x <= p_max_x)
        & (f_max_y >= p_min_y) & (f_min_y <= p_max_y)
    )

    # Full discriminant check (F, P) over the AABB candidates.
    d0x = fold_x[:, None] - pold_x[None, :]
    d0y = fold_y[:, None] - pold_y[None, :]
    dvx = (fnew_x - fold_x)[:, None] - (pnew_x - pold_x)[None, :]
    dvy = (fnew_y - fold_y)[:, None] - (pnew_y - pold_y)[None, :]
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - (pr_arr * pr_arr)[None, :]

    a_small = a < 1e-12
    disc = b * b - 4.0 * a * c
    disc_ok = disc >= 0.0
    sq = np.sqrt(np.where(disc_ok, disc, 0.0))
    denom = np.where(a_small, 1.0, 2.0 * a)
    t1 = (-b - sq) / denom
    t2 = (-b + sq) / denom
    hit_full = disc_ok & (t2 >= 0.0) & (t1 <= 1.0)
    hit = candidate & np.where(a_small, c <= 0.0, hit_full)

    any_hit = hit.any(axis=1)
    first_hit = hit.argmax(axis=1)
    # argmax returns 0 when no True; mask those out.
    results_arr = np.where(any_hit, first_hit, -1)
    return results_arr.tolist()


Planet = namedtuple(
    "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
)
Fleet = namedtuple(
    "Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"]
)

BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
PLANET_CLEARANCE = 7
MIN_PLANET_GROUPS = 5
MAX_PLANET_GROUPS = 10
MIN_STATIC_GROUPS = 3
COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]


def _get(d, key, default):
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return _sqrt(dx * dx + dy * dy)


def point_to_segment_distance(p, v, w):
    vx = v[0]; vy = v[1]
    wx = w[0]; wy = w[1]
    dx = vx - wx
    dy = vy - wy
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        return distance(p, v)
    px = p[0]; py = p[1]
    t = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    projx = vx + t * (wx - vx)
    projy = vy + t * (wy - vy)
    dx2 = px - projx
    dy2 = py - projy
    return _sqrt(dx2 * dx2 + dy2 * dy2)


def swept_pair_hit(A, B, P0, P1, r):
    A0 = A[0]; A1 = A[1]
    B0 = B[0]; B1 = B[1]
    P00 = P0[0]; P01 = P0[1]
    P10 = P1[0]; P11 = P1[1]
    # Cheap AABB prune. The fleet (a point) traverses segment A→B; the
    # planet center traverses P0→P1, with the planet body inflating by
    # radius r in every direction. If the fleet's segment bbox and the
    # inflated planet-center segment bbox are disjoint on either axis,
    # no collision is possible — skip the discriminant math.
    if A0 < B0:
        fmin_x = A0; fmax_x = B0
    else:
        fmin_x = B0; fmax_x = A0
    if P00 < P10:
        pmin_x = P00 - r; pmax_x = P10 + r
    else:
        pmin_x = P10 - r; pmax_x = P00 + r
    if fmax_x < pmin_x or fmin_x > pmax_x:
        return False
    if A1 < B1:
        fmin_y = A1; fmax_y = B1
    else:
        fmin_y = B1; fmax_y = A1
    if P01 < P11:
        pmin_y = P01 - r; pmax_y = P11 + r
    else:
        pmin_y = P11 - r; pmax_y = P01 + r
    if fmax_y < pmin_y or fmin_y > pmax_y:
        return False

    d0x = A0 - P00; d0y = A1 - P01
    dvx = (B0 - A0) - (P10 - P00)
    dvy = (B1 - A1) - (P11 - P01)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = _sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0


def generate_planets(rng=None):
    if rng is None:
        rng = random
    planets = []
    num_q1 = rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS)
    id_counter = 0

    static_groups = 0
    for _ in range(5000):
        if static_groups >= MIN_STATIC_GROUPS:
            break
        prod = rng.randint(1, 5)
        r = 1 + math.log(prod)
        angle = rng.uniform(0, math.pi / 2)
        min_orbital = ROTATION_RADIUS_LIMIT - r
        max_orbital = (BOARD_SIZE - CENTER - r) / max(math.cos(angle), math.sin(angle))
        if min_orbital > max_orbital:
            continue
        orbital_r = rng.uniform(min_orbital, max_orbital)
        x = CENTER + orbital_r * math.cos(angle)
        y = CENTER + orbital_r * math.sin(angle)

        if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
            continue
        if (BOARD_SIZE - x) - r < 0 or (BOARD_SIZE - y) - r < 0:
            continue
        if (x - CENTER) < r + 5 or (y - CENTER) < r + 5:
            continue

        ships = min(rng.randint(5, 99), rng.randint(5, 99))
        temp_planets = [
            [id_counter, -1, y, x, r, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
            [id_counter + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ]

        valid = True
        for tp in temp_planets:
            for p in planets:
                if distance((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            planets.extend(temp_planets)
            id_counter += 4
            static_groups += 1

    attempts = 0
    max_attempts = 5000
    has_orbiting = False

    while len(planets) < num_q1 * 4 or (not has_orbiting and attempts < max_attempts):
        attempts += 1
        if attempts >= max_attempts:
            break
        prod = rng.randint(1, 5)
        r = 1 + math.log(prod)
        x = rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)
        y = rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)

        orbital_radius = distance((x, y), (CENTER, CENTER))

        if orbital_radius < SUN_RADIUS + r + 10:
            continue

        if orbital_radius + r >= ROTATION_RADIUS_LIMIT:
            if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
                continue

        valid = True
        ships = rng.randint(5, 30)
        temp_planets = [
            [id_counter, -1, y, x, r, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
            [id_counter + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ]

        for tp in temp_planets:
            tp_orbital = distance((tp[2], tp[3]), (CENTER, CENTER))
            tp_is_rotating = tp_orbital + tp[4] < ROTATION_RADIUS_LIMIT

            for p in planets:
                p_orbital = distance((p[2], p[3]), (CENTER, CENTER))
                p_is_rotating = p_orbital + p[4] < ROTATION_RADIUS_LIMIT

                if distance((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False
                    break

                if tp_is_rotating != p_is_rotating:
                    if abs(tp_orbital - p_orbital) < tp[4] + p[4] + PLANET_CLEARANCE:
                        valid = False
                        break

            if not valid:
                break

        if valid:
            if orbital_radius + r < ROTATION_RADIUS_LIMIT:
                has_orbiting = True
            planets.extend(temp_planets)
            id_counter += 4

    return planets


def generate_comet_paths(
    initial_planets,
    angular_velocity,
    spawn_step,
    comet_planet_ids=None,
    comet_speed=4.0,
    rng=None,
):
    if rng is None:
        rng = random
    if comet_planet_ids is None:
        comet_planet_ids = set()
    else:
        comet_planet_ids = set(comet_planet_ids)
    for _ in range(300):
        e = rng.uniform(0.75, 0.93)
        a = rng.uniform(60, 150)
        perihelion = a * (1 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            continue

        b = a * math.sqrt(1 - e**2)
        c_val = a * e
        phi = rng.uniform(math.pi / 6, math.pi / 3)

        # Vectorised dense-sample loop (formerly 5000 trig calls per iter
        # × 4 trig calls each). Bit-exact with the scalar version: numpy
        # uses the same libm cos/sin/sqrt and we keep float64 throughout.
        num = 5000
        idx = np.arange(num, dtype=np.float64)
        t = 0.3 * math.pi + 1.4 * math.pi * idx / (num - 1)
        ex = c_val + a * np.cos(t)
        ey = b * np.sin(t)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        x_arr = CENTER + ex * cos_phi - ey * sin_phi
        y_arr = CENTER + ex * sin_phi + ey * cos_phi
        dense_x = x_arr.tolist()
        dense_y = y_arr.tolist()
        dense = list(zip(dense_x, dense_y))

        # Vectorised re-sample at constant comet_speed arc-length intervals.
        # The scalar code maintains a running cumulative distance and
        # appends dense[i] whenever cum >= target = k*comet_speed for the
        # next k. Equivalent: for each k=1..K, find smallest i such that
        # cum_dist[i] >= k*comet_speed; append dense[i+1] (using cum_dist
        # being length len(dense)-1, indexed from dense[1] to dense[-1]).
        diffs = np.sqrt(
            (x_arr[1:] - x_arr[:-1]) ** 2 + (y_arr[1:] - y_arr[:-1]) ** 2
        )
        cum_dist = np.cumsum(diffs)
        total_dist = float(cum_dist[-1]) if cum_dist.size else 0.0
        max_k = int(total_dist // comet_speed)
        if max_k > 0:
            targets = comet_speed * np.arange(1, max_k + 1, dtype=np.float64)
            search_idx = np.searchsorted(cum_dist, targets, side="left")
            # cum_dist[j] >= target → append dense[j+1] in original indexing.
            picked = (search_idx + 1).tolist()
            path = [dense[0]] + [dense[i] for i in picked if i < num]
        else:
            path = [dense[0]]

        board_start = None
        board_end = None
        for i, (x, y) in enumerate(path):
            if 0 <= x <= BOARD_SIZE and 0 <= y <= BOARD_SIZE:
                if board_start is None:
                    board_start = i
                board_end = i

        if board_start is None:
            continue
        visible = path[board_start : board_end + 1]
        if not (5 <= len(visible) <= 40):
            continue

        paths = [
            [[y, x] for x, y in visible],
            [[BOARD_SIZE - x, y] for x, y in visible],
            [[x, BOARD_SIZE - y] for x, y in visible],
            [[BOARD_SIZE - y, BOARD_SIZE - x] for x, y in visible],
        ]

        static_planets = []
        orbiting_planets = []
        for planet in initial_planets:
            if planet[0] in comet_planet_ids:
                continue
            pr = distance((planet[2], planet[3]), (CENTER, CENTER))
            if pr + planet[4] < ROTATION_RADIUS_LIMIT:
                orbiting_planets.append(planet)
            else:
                static_planets.append(planet)

        valid = True
        buf = COMET_RADIUS + 0.5
        for k, (cx, cy) in enumerate(visible):
            if distance((cx, cy), (CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS:
                valid = False
                break

            sym_pts = [
                (cy, cx),
                (BOARD_SIZE - cx, cy),
                (cx, BOARD_SIZE - cy),
                (BOARD_SIZE - cy, BOARD_SIZE - cx),
            ]
            for planet in static_planets:
                for sp in sym_pts:
                    if distance(sp, (planet[2], planet[3])) < planet[4] + buf:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

            game_step = spawn_step - 1 + k
            for planet in orbiting_planets:
                dx = planet[2] - CENTER
                dy = planet[3] - CENTER
                orb_r = math.sqrt(dx**2 + dy**2)
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + angular_velocity * game_step
                px = CENTER + orb_r * math.cos(cur_angle)
                py = CENTER + orb_r * math.sin(cur_angle)
                for sp in sym_pts:
                    if distance(sp, (px, py)) < planet[4] + COMET_RADIUS:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if valid:
            return paths
    return None


def interpreter(state, env):
    configuration = env.configuration
    num_agents = len(state)
    obs0 = state[0].observation

    if not hasattr(obs0, "planets") or not obs0.planets:
        if not hasattr(env, "info") or env.info is None:
            env.info = {}
        seed = env.info.get("seed")
        if seed is None:
            seed = _get(configuration, "seed", None)
        if seed is None:
            seed = random.randrange(2**31)
        try:
            configuration.seed = None
        except (AttributeError, TypeError):
            configuration["seed"] = None
        env.info["seed"] = seed
        init_rng = random.Random(seed)

        angular_velocity = init_rng.uniform(0.025, 0.05)
        obs0.angular_velocity = angular_velocity
        obs0.planets = generate_planets(init_rng)
        obs0.initial_planets = [p.copy() for p in obs0.planets]
        obs0.fleets = []
        obs0.next_fleet_id = 0
        obs0.comets = []
        obs0.comet_planet_ids = []

        num_groups = len(obs0.planets) // 4
        if num_groups > 0:
            home_group = init_rng.randint(0, num_groups - 1)
            base = home_group * 4

            if num_agents == 2:
                obs0.planets[base][1] = 0
                obs0.planets[base][5] = 10
                obs0.planets[base + 3][1] = 1
                obs0.planets[base + 3][5] = 10
            elif num_agents == 4:
                for j in range(4):
                    obs0.planets[base + j][1] = j
                    obs0.planets[base + j][5] = 10

        for i in range(num_agents):
            state[i].observation.player = i
            if i > 0:
                state[i].observation.angular_velocity = obs0.angular_velocity
                state[i].observation.planets = obs0.planets
                state[i].observation.initial_planets = obs0.initial_planets
                state[i].observation.fleets = obs0.fleets
                state[i].observation.next_fleet_id = obs0.next_fleet_id
                state[i].observation.comets = obs0.comets
                state[i].observation.comet_planet_ids = obs0.comet_planet_ids

        return state

    if env.done:
        return state

    expired_comet_pids = []
    for group in obs0.comets:
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            if idx >= len(group["paths"][i]):
                expired_comet_pids.append(pid)
    if expired_comet_pids:
        expired_set = set(expired_comet_pids)
        obs0.planets = [p for p in obs0.planets if p[0] not in expired_set]
        obs0.initial_planets = [
            p for p in obs0.initial_planets if p[0] not in expired_set
        ]
        obs0.comet_planet_ids = [
            pid for pid in obs0.comet_planet_ids if pid not in expired_set
        ]
        for group in obs0.comets:
            group["planet_ids"] = [
                pid for pid in group["planet_ids"] if pid not in expired_set
            ]
        obs0.comets = [g for g in obs0.comets if g["planet_ids"]]

    step = _get(obs0, "step", 0)
    comet_speed = configuration.cometSpeed
    if (step + 1) in COMET_SPAWN_STEPS:
        env_info = getattr(env, "info", None) or {}
        episode_seed = env_info.get("seed", 0) or 0
        # Comet paths + ships are deterministic from (episode_seed,
        # spawn_step, initial_planets, comet_planet_ids). Inside one
        # lookahead, initial_planets and comet_planet_ids are fixed up
        # to the agent's "now", so all rollouts crossing the next spawn
        # boundary share the same result. Cache it on env.comet_path_cache
        # to amortise the ~100 ms generation across rollouts.
        cache = getattr(env, "comet_path_cache", None)
        cache_key = (episode_seed, step + 1)
        cached = cache.get(cache_key) if cache is not None else None
        if cached is not None:
            comet_paths, comet_ships = cached
        else:
            comet_rng = random.Random(
                f"orbit_wars-comet-{episode_seed}-{step + 1}"
            )
            comet_paths = generate_comet_paths(
                obs0.initial_planets,
                obs0.angular_velocity,
                step + 1,
                obs0.comet_planet_ids,
                comet_speed,
                rng=comet_rng,
            )
            if comet_paths:
                comet_ships = min(
                    comet_rng.randint(1, 99),
                    comet_rng.randint(1, 99),
                    comet_rng.randint(1, 99),
                    comet_rng.randint(1, 99),
                )
            else:
                comet_ships = None
            if cache is not None:
                cache[cache_key] = (comet_paths, comet_ships)
        if comet_paths:
            next_id = max(p[0] for p in obs0.planets) + 1
            group = {"planet_ids": [], "paths": comet_paths, "path_index": -1}
            for i, p_path in enumerate(comet_paths):
                pid = next_id + i
                group["planet_ids"].append(pid)
                obs0.comet_planet_ids.append(pid)
                planet = [
                    pid,
                    -1,
                    -99,
                    -99,
                    COMET_RADIUS,
                    comet_ships,
                    COMET_PRODUCTION,
                ]
                obs0.planets.append(planet)
                obs0.initial_planets.append(planet[:])
            obs0.comets.append(group)

    def process_moves(player_id, action):
        if not action or not isinstance(action, list):
            return
        for move in action:
            if len(move) != 3:
                continue
            from_id, angle, ships = move
            ships = int(ships)

            from_planet = next((p for p in obs0.planets if p[0] == from_id), None)

            if from_planet and from_planet[1] == player_id:
                if from_planet[5] >= ships and ships > 0:
                    from_planet[5] -= ships
                    start_x = from_planet[2] + math.cos(angle) * (from_planet[4] + 0.1)
                    start_y = from_planet[3] + math.sin(angle) * (from_planet[4] + 0.1)
                    obs0.fleets.append(
                        [
                            obs0.next_fleet_id,
                            player_id,
                            start_x,
                            start_y,
                            angle,
                            from_id,
                            ships,
                        ]
                    )
                    obs0.next_fleet_id += 1

    for i in range(num_agents):
        process_moves(i, state[i].action)

    for planet in obs0.planets:
        if planet[1] != -1:
            planet[5] += planet[6]

    angular_velocity = obs0.angular_velocity
    step = _get(obs0, "step", 1)
    comet_pid_set = set(obs0.comet_planet_ids)
    initial_by_id = {p[0]: p for p in obs0.initial_planets}

    planets_local = obs0.planets
    comets_local = obs0.comets
    fleets_local = obs0.fleets

    planet_paths = {}
    expired_comet_pids = []

    _atan2 = math.atan2
    # Optional pre-computed planet positions (populated by fast_sim).
    pos_cache = getattr(env, "planet_position_cache", None)
    for planet in planets_local:
        if planet[0] in comet_pid_set:
            continue
        old_pos = (planet[2], planet[3])
        new_pos = old_pos
        # Cache hit: read pre-computed (x, y) for this step. Falls through
        # to the trig path on miss (non-rotating planets, or absent cache).
        if pos_cache is not None:
            cached = pos_cache.get(planet[0])
            if cached is not None and step < len(cached):
                new_pos = cached[step]
                planet_paths[planet[0]] = (old_pos, new_pos, True)
                continue
        initial_p = initial_by_id.get(planet[0])
        if initial_p is not None:
            dx = initial_p[2] - CENTER
            dy = initial_p[3] - CENTER
            r = _sqrt(dx * dx + dy * dy)
            if r + planet[4] < ROTATION_RADIUS_LIMIT:
                current_angle = _atan2(dy, dx) + angular_velocity * step
                new_pos = (
                    CENTER + r * _cos(current_angle),
                    CENTER + r * _sin(current_angle),
                )
        planet_paths[planet[0]] = (old_pos, new_pos, True)

    # planet_by_id once for the comet update lookup (replaces per-comet
    # linear scan over all planets).
    planet_by_id = {p[0]: p for p in planets_local}
    for group in comets_local:
        group["path_index"] += 1
        idx = group["path_index"]
        group_paths = group["paths"]
        for i, pid in enumerate(group["planet_ids"]):
            planet = planet_by_id.get(pid)
            if planet is None:
                continue
            p_path = group_paths[i]
            old_pos = (planet[2], planet[3])
            if idx >= len(p_path):
                expired_comet_pids.append(pid)
                planet_paths[pid] = (old_pos, old_pos, True)
            else:
                pp = p_path[idx]
                new_pos = (pp[0], pp[1])
                check = old_pos[0] >= 0
                planet_paths[pid] = (old_pos, new_pos, check)

    max_speed = configuration.shipSpeed
    log1000 = _log(1000)
    fleets_to_remove = []
    combat_lists = {p[0]: [] for p in planets_local}

    _swept = swept_pair_hit
    _seg_dist = point_to_segment_distance
    _ftr_append = fleets_to_remove.append
    sun_center = (CENTER, CENTER)
    for fleet in fleets_local:
        angle = fleet[4]
        ships = fleet[6]
        speed = 1.0 + (max_speed - 1.0) * (_log(ships) / log1000) ** 1.5
        if speed > max_speed:
            speed = max_speed
        f2 = fleet[2]; f3 = fleet[3]
        old_pos = (f2, f3)
        new_x = f2 + _cos(angle) * speed
        new_y = f3 + _sin(angle) * speed
        fleet[2] = new_x
        fleet[3] = new_y
        new_pos = (new_x, new_y)

        hit_planet = False
        for planet in planets_local:
            path = planet_paths.get(planet[0])
            if path is None or not path[2]:
                continue
            if _swept(old_pos, new_pos, path[0], path[1], planet[4]):
                combat_lists[planet[0]].append(fleet)
                _ftr_append(fleet)
                hit_planet = True
                break
        if hit_planet:
            continue

        if not (0 <= new_x <= BOARD_SIZE and 0 <= new_y <= BOARD_SIZE):
            _ftr_append(fleet)
            continue

        if _seg_dist(sun_center, old_pos, new_pos) < SUN_RADIUS:
            _ftr_append(fleet)
            continue

    for planet in obs0.planets:
        path = planet_paths.get(planet[0])
        if path is not None:
            planet[2], planet[3] = path[1]

    if expired_comet_pids:
        expired_set = set(expired_comet_pids)
        obs0.planets = [p for p in obs0.planets if p[0] not in expired_set]
        obs0.initial_planets = [
            p for p in obs0.initial_planets if p[0] not in expired_set
        ]
        obs0.comet_planet_ids = [
            pid for pid in obs0.comet_planet_ids if pid not in expired_set
        ]
        for group in obs0.comets:
            group["planet_ids"] = [
                pid for pid in group["planet_ids"] if pid not in expired_set
            ]
        obs0.comets = [g for g in obs0.comets if g["planet_ids"]]

    # Identity-based removal: `f not in list` triggers element-wise list
    # equality (O(N·M·7)); id() membership is O(N+M) on a hash set.
    if fleets_to_remove:
        remove_ids = {id(f) for f in fleets_to_remove}
        obs0.fleets = [f for f in obs0.fleets if id(f) not in remove_ids]

    for pid, planet_fleets in combat_lists.items():
        planet = next((p for p in obs0.planets if p[0] == pid), None)
        if not planet or not planet_fleets:
            continue

        player_ships = {}
        for fleet in planet_fleets:
            owner = fleet[1]
            player_ships[owner] = player_ships.get(owner, 0) + fleet[6]

        if not player_ships:
            continue

        sorted_players = sorted(
            player_ships.items(), key=lambda item: item[1], reverse=True
        )
        top_player, top_ships = sorted_players[0]

        if len(sorted_players) > 1:
            second_ships = sorted_players[1][1]
            survivor_ships = top_ships - second_ships

            if sorted_players[0][1] == sorted_players[1][1]:
                survivor_ships = 0

            survivor_owner = top_player if survivor_ships > 0 else -1
        else:
            survivor_owner = top_player
            survivor_ships = top_ships

        if survivor_ships > 0:
            if planet[1] == survivor_owner:
                planet[5] += survivor_ships
            else:
                planet[5] -= survivor_ships
                if planet[5] < 0:
                    planet[1] = survivor_owner
                    planet[5] = abs(planet[5])

    for i in range(1, num_agents):
        state[i].observation.planets = obs0.planets
        state[i].observation.initial_planets = obs0.initial_planets
        state[i].observation.fleets = obs0.fleets
        state[i].observation.next_fleet_id = obs0.next_fleet_id
        state[i].observation.comets = obs0.comets
        state[i].observation.comet_planet_ids = obs0.comet_planet_ids

    terminated = False
    step = _get(obs0, "step", 0)
    if step >= configuration.episodeSteps - 2:
        terminated = True

    alive_players = set()
    for p in obs0.planets:
        if p[1] != -1:
            alive_players.add(p[1])
    for f in obs0.fleets:
        alive_players.add(f[1])

    if len(alive_players) <= 1:
        terminated = True

    if terminated:
        for s in state:
            s.status = "DONE"

        scores = [0] * num_agents
        for p in obs0.planets:
            if p[1] != -1:
                scores[p[1]] += p[5]
        for f in obs0.fleets:
            scores[f[1]] += f[6]

        max_score = max(scores)
        for i in range(num_agents):
            if scores[i] == max_score and max_score > 0:
                state[i].reward = 1
            else:
                state[i].reward = -1

    return state

# === inlined: lib/fast_sim.py ===


import copy
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from kaggle_environments.utils import Struct

_orbit_wars_interpreter = interpreter
_GAME_BOARD_SIZE = BOARD_SIZE
_GAME_CENTER = CENTER
_GAME_RRL = ROTATION_RADIUS_LIMIT


# Default configuration matching kaggle_environments.envs.orbit_wars's
# `orbit_wars.json` defaults; kept here so fast_sim doesn't depend on
# any I/O at import time.
DEFAULT_CONFIG: dict[str, Any] = {
    "episodeSteps": 500,
    "shipSpeed": 6.0,
    "sunRadius": 10.0,
    "boardSize": 100.0,
    "cometSpeed": 4.0,
    "actTimeout": 1.0,
    "agentTimeout": 60.0,
    "runTimeout": 1200.0,
}


class _FakeEnv:
    """Minimal env shim that satisfies what `interpreter()` reads.

    The interpreter touches `env.configuration`, `env.info`, `env.done`
    only (orbit_wars.py:335, 348, 363, 406, 438, 570, 686). We provide
    just those three attributes. `info["seed"]` is read once per comet
    spawn (orbit_wars.py:438-440) — passing it through keeps comet RNG
    deterministic.

    `comet_path_cache` is a dict {(episode_seed, spawn_step):
    (comet_paths_or_None, comet_ships_or_None)} populated lazily by the
    interpreter. It's SHARED across clones so all branches of a
    lookahead inherit the same cache and amortise the ~100 ms
    generate_comet_paths cost across rollouts.

    `planet_position_cache` is a dict {planet_id: list[(x, y)]} keyed
    by initial-planet id, indexed by absolute step. Pre-computed once
    in `from_obs()` for every rotating planet; the interpreter uses it
    instead of recomputing `atan2/cos/sin` each step. SHARED across
    clones too. ~240 KB at 30 rotating planets × 500 steps.
    """
    __slots__ = (
        "configuration", "info", "done",
        "comet_path_cache", "planet_position_cache",
    )

    def __init__(self, configuration: Struct, episode_seed: int) -> None:
        self.configuration = configuration
        self.info = {"seed": episode_seed}
        self.done = False
        self.comet_path_cache = {}
        self.planet_position_cache = {}


@dataclass
class Snapshot:
    """Forward-simulator state. `state` is a list of `Struct` per seat,
    each with `observation` (also a Struct), `action`, `status`,
    `reward`. This matches what the env's interpreter expects to mutate.

    Don't mutate directly; use `step()` / `clone()`. `from_obs()` is the
    only sanctioned constructor for production code.
    """
    state: list[Struct]
    fake_env: _FakeEnv
    episode_seed: int

    @property
    def obs(self) -> Struct:
        """Primary observation (seat 0). The interpreter mutates this in
        place; the other seats hold the same list/dict references."""
        return self.state[0].observation

    @property
    def step_idx(self) -> int:
        return int(self.obs.get("step", 0))

    @property
    def done(self) -> bool:
        return self.fake_env.done

    @property
    def num_seats(self) -> int:
        return len(self.state)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


# The mutable fields on `state[i].observation` that fast_sim cares about.
# Anything not in this tuple is either immutable across the episode
# (e.g. `player`) or not read by the interpreter.
_OBS_KEYS = (
    "planets", "fleets", "comets", "comet_planet_ids",
    "initial_planets", "angular_velocity", "step", "next_fleet_id",
)


def _configuration_struct(configuration) -> Struct:
    """Coerce a configuration (dict / SimpleNamespace / Struct) into a
    Struct with the defaults filled in.

    The env's interpreter reads `configuration.shipSpeed`,
    `configuration.cometSpeed`, `configuration.episodeSteps` — Struct's
    dual attr+dict access satisfies both styles.
    """
    cfg = dict(DEFAULT_CONFIG)
    if configuration is not None:
        if isinstance(configuration, dict):
            cfg.update(configuration)
        else:
            for k in DEFAULT_CONFIG:
                v = getattr(configuration, k, None)
                if v is not None:
                    cfg[k] = v
    return Struct(**cfg)


def _read_obs_field(obs: Any, key: str, default: Any = None) -> Any:
    """Dual dict-or-attr read, matching the env's `get()` helper."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def from_obs(
    obs: Any,
    configuration: Any = None,
    *,
    episode_seed: int = 0,
    num_seats: int = 2,
) -> Snapshot:
    """Build a Snapshot from an agent-visible observation.

    `obs` accepts both dict-style (live ladder) and Struct-style (env
    in-memory). Pulls each mutable field via `_read_obs_field` and
    deep-copies it so the Snapshot is independent of the source obs.

    `configuration` is folded into the env defaults; pass the agent's
    `configuration` argument unchanged.

    `episode_seed` is the env's `info["seed"]` (or any deterministic
    int). Required for bit-exact comet spawns; if unknown, pass 0 and
    accept divergence past a spawn boundary (see module docstring).

    `num_seats` defaults to 2P; pass 4 for FFA. Single-seat (`num_seats
    = 1`) is allowed for tests but isn't a real game configuration.
    """
    config_struct = _configuration_struct(configuration)

    obs_data: dict[str, Any] = {}
    for k in _OBS_KEYS:
        v = _read_obs_field(obs, k)
        if v is not None:
            obs_data[k] = copy.deepcopy(v)
    # Make sure required scalar fields are present (interpreter reads
    # `step` and `next_fleet_id` with `get(..., default)`; the defaults
    # are safe but having them explicit avoids attribute fallbacks).
    obs_data.setdefault("step", 0)
    obs_data.setdefault("next_fleet_id", 0)
    obs_data.setdefault("fleets", [])
    obs_data.setdefault("comets", [])
    obs_data.setdefault("comet_planet_ids", [])
    obs_data.setdefault("initial_planets", [list(p) for p in obs_data.get("planets", [])])
    obs_data.setdefault("angular_velocity", 0.0)

    state: list[Struct] = []
    obs0 = Struct(**obs_data)
    obs0.player = 0
    state.append(Struct(
        observation=obs0,
        action=None,
        status="ACTIVE",
        reward=0,
        info={},
    ))
    # Other seats share the same mutable references — same aliasing
    # behaviour the env's `interpreter()` sets up on init
    # (orbit_wars.py:393-402) and re-applies after each step
    # (orbit_wars.py:676-682).
    for i in range(1, num_seats):
        obs_i = Struct(**{k: getattr(obs0, k) for k in obs_data})
        obs_i.player = i
        state.append(Struct(
            observation=obs_i,
            action=None,
            status="ACTIVE",
            reward=0,
            info={},
        ))

    fake_env = _FakeEnv(config_struct, episode_seed)
    _populate_planet_position_cache(fake_env, obs0)
    return Snapshot(state=state, fake_env=fake_env, episode_seed=episode_seed)


def _populate_planet_position_cache(fake_env, obs0) -> None:
    """Pre-compute orbital positions for every rotating planet at every
    step of the episode. Eliminates per-step atan2/cos/sin for the planet
    path computation. Storage ~240 KB at 30 planets × 500 steps.
    """
    import math as _math
    cache = fake_env.planet_position_cache
    angular_velocity = float(obs0.angular_velocity)
    episode_steps = int(fake_env.configuration.episodeSteps)
    initial_planets = obs0.initial_planets
    comet_pid_set = set(obs0.comet_planet_ids)
    sqrt = _math.sqrt
    cos = _math.cos
    sin = _math.sin
    atan2 = _math.atan2
    for ip in initial_planets:
        pid = ip[0]
        if pid in comet_pid_set:
            continue
        dx = ip[2] - _GAME_CENTER
        dy = ip[3] - _GAME_CENTER
        r = sqrt(dx * dx + dy * dy)
        if r + ip[4] >= _GAME_RRL:
            # Non-rotating; no cache entry needed (interpreter keeps the
            # static position).
            continue
        initial_angle = atan2(dy, dx)
        # Index 0 is the position AT step 0 (which equals initial position).
        positions = []
        for s in range(episode_steps + 1):
            theta = initial_angle + angular_velocity * s
            positions.append((_GAME_CENTER + r * cos(theta), _GAME_CENTER + r * sin(theta)))
        cache[pid] = positions


# ---------------------------------------------------------------------------
# Cloning
# ---------------------------------------------------------------------------


def clone(snap: Snapshot) -> Snapshot:
    """Targeted deep-copy of mutating fields. Faster than `copy.deepcopy`.

    What gets copied (the interpreter mutates these):
    - `planets`: list + each inner list (planet ships incremented,
      positions overwritten — `orbit_wars.py:514, 615`).
    - `fleets`: list + each inner list (positions advanced —
      `orbit_wars.py:580-581`).
    - `initial_planets`: list (filtered on comet expiration), inner
      lists are read-only after init so shared safely.
    - `comets`: list + per-group dict + `planet_ids` list. `paths` is
      built once and only read; we share it.
    - `comet_planet_ids`: list (filtered on expiration / appended on
      spawn).
    - Scalars: `step`, `next_fleet_id`, `angular_velocity`, `player`.

    What gets shared (immutable across the episode):
    - `fake_env.configuration` (Struct).
    - Comet `paths`.
    """
    src0 = snap.state[0].observation

    obs0 = Struct(
        planets=[list(p) for p in src0.planets],
        fleets=[list(f) for f in src0.fleets],
        initial_planets=[list(p) for p in src0.initial_planets],
        comet_planet_ids=list(src0.comet_planet_ids),
        comets=[
            {
                "planet_ids": list(g["planet_ids"]),
                "paths": g["paths"],
                "path_index": g["path_index"],
            }
            for g in src0.comets
        ],
        angular_velocity=src0.angular_velocity,
        step=int(src0.get("step", 0)),
        next_fleet_id=int(src0.next_fleet_id),
        player=0,
    )
    new_state: list[Struct] = [Struct(
        observation=obs0,
        action=None,
        status=snap.state[0].status,
        reward=snap.state[0].reward,
        info={},
    )]
    for i in range(1, snap.num_seats):
        obs_i = Struct(
            planets=obs0.planets,
            fleets=obs0.fleets,
            initial_planets=obs0.initial_planets,
            comet_planet_ids=obs0.comet_planet_ids,
            comets=obs0.comets,
            angular_velocity=obs0.angular_velocity,
            step=obs0.step,
            next_fleet_id=obs0.next_fleet_id,
            player=i,
        )
        new_state.append(Struct(
            observation=obs_i,
            action=None,
            status=snap.state[i].status,
            reward=snap.state[i].reward,
            info={},
        ))

    fake_env = _FakeEnv(snap.fake_env.configuration, snap.episode_seed)
    fake_env.done = snap.fake_env.done
    # Share both caches with the parent so all lookahead branches benefit
    # from work done by any one of them (comet generation, planet orbits).
    fake_env.comet_path_cache = snap.fake_env.comet_path_cache
    fake_env.planet_position_cache = snap.fake_env.planet_position_cache
    return Snapshot(state=new_state, fake_env=fake_env, episode_seed=snap.episode_seed)


# ---------------------------------------------------------------------------
# Stepping
# ---------------------------------------------------------------------------


def step(
    snap: Snapshot,
    actions_per_seat: Sequence[list],
    *,
    in_place: bool = False,
) -> Snapshot:
    """Advance the snapshot by one tick.

    `actions_per_seat` is a list of `[[src_id, angle, ships], ...]` per
    seat (matches the env action format). Pass `[]` for seats that
    don't launch this turn.

    Returns a new Snapshot unless `in_place=True`, in which case `snap`
    is mutated and returned.

    The actual game logic comes from
    `kaggle_environments.envs.orbit_wars.orbit_wars.interpreter` — same
    physics, RNG, combat resolution the real env uses. The fast path
    skips: action-schema validation (`core.py:262`), structify wrapping
    (`core.py:600`), state-history append (`core.py:277`), and the
    redundant per-step state deepcopy that Environment.clone() pays.
    """
    if snap.fake_env.done:
        # Mirror Environment.step()'s "cannot step a done env"
        # convention without raising (rollout loops check `snap.done`
        # to exit; raising would surprise them).
        if not in_place:
            snap = clone(snap)
        return snap

    if not in_place:
        snap = clone(snap)

    # Wire per-seat actions.
    for i, action in enumerate(actions_per_seat):
        snap.state[i].action = action
    # Any seat we didn't get an action for gets [] (no-op).
    for i in range(len(actions_per_seat), snap.num_seats):
        snap.state[i].action = []

    # Call the env's interpreter directly. Mutates state in place.
    _orbit_wars_interpreter(snap.state, snap.fake_env)

    # Post-interpreter bookkeeping that core.py handles for us:
    #   1. Increment observation.step (core.py:602).
    #   2. Update fake_env.done if any seat went DONE.
    obs0 = snap.state[0].observation
    obs0.step = int(obs0.get("step", 0)) + 1
    for i in range(1, snap.num_seats):
        snap.state[i].observation.step = obs0.step

    if any(s.status == "DONE" for s in snap.state):
        snap.fake_env.done = True

    return snap


# ---------------------------------------------------------------------------
# Scoring head
# ---------------------------------------------------------------------------


def ship_totals(snap: Snapshot) -> dict[int, float]:
    """Sum ships on owned planets + in-flight fleets, per owner.

    Same scoring head as `lib/lookahead.py::_ship_total_by_owner` and
    the Phase 2 probe (audit:69-76); kept here so future consumers
    don't import from `lookahead`.
    """
    totals: dict[int, float] = {}
    obs0 = snap.state[0].observation
    for p in obs0.planets:
        owner = int(p[1])
        if owner >= 0:
            totals[owner] = totals.get(owner, 0.0) + float(p[5])
    for f in obs0.fleets:
        owner = int(f[1])
        if owner >= 0:
            totals[owner] = totals.get(owner, 0.0) + float(f[6])
    return totals


def delta_us_minus_them(snap: Snapshot, my_id: int) -> float:
    """`(our ships) - (sum of other seats' ships)`. The Phase 2 scoring
    scalar — the value that the AUC = 0.952 probe at K=50 measured.
    """
    t = ship_totals(snap)
    ours = t.get(my_id, 0.0)
    theirs = sum(v for k, v in t.items() if k != my_id)
    return ours - theirs


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


Policy = Callable[[Any], list]


def rollout(
    snap: Snapshot,
    K: int,
    policies: Sequence[Policy],
    *,
    in_place: bool = False,
) -> Snapshot:
    """Roll forward up to `K` ticks under per-seat policies.

    `policies[i]` is `Callable(obs) -> action`. Each tick:
    1. For each seat, call `policies[i](snap.state[i].observation)` to
       get the action.
    2. `step(snap, actions, in_place=True)`.
    3. If `snap.done`, exit early.

    Returns the terminal Snapshot. Equivalent to
    `lib/lookahead.py::score_action`'s inner loop but operating on a
    Snapshot, with the env-overhead stripped.
    """
    if not in_place:
        snap = clone(snap)
    if len(policies) != snap.num_seats:
        raise ValueError(
            f"need {snap.num_seats} policies, got {len(policies)}"
        )
    for _ in range(K):
        if snap.fake_env.done:
            break
        actions = [policies[i](snap.state[i].observation) for i in range(snap.num_seats)]
        snap = step(snap, actions, in_place=True)
    return snap

# === inlined: lib/opp_model.py ===


from typing import Any, Callable



Policy = Callable[[Any], list]


# ---------------------------------------------------------------------------
# Tier 0 — mirror self (= v3_snipe pipeline, aggressive sizing OFF)
# ---------------------------------------------------------------------------


def mirror_self_policy(obs: Any) -> list:
    """Run the v3_snipe pipeline against `obs`. Bit-exact equivalent of
    `agents/v3_snipe/main.py`'s `agent(obs)` body.

    Drop-in replacement for `lib.lookahead.score_action`'s `policy`
    argument when you want the Phase 2 default ("opponent plays v2").
    """
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    model = WorldModel.from_world(world)
    missions = (
        propose_snipe_missions(world, model, aggressive=False)
        + propose_reinforce_missions(world, model)
    )
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)


# ---------------------------------------------------------------------------
# Tier 1 — top-tier mirror (= v3.5.1 pipeline, aggressive sizing ON)
# ---------------------------------------------------------------------------


def top_tier_mirror_policy(obs: Any) -> list:
    """Run the v3.5.1 pipeline against `obs`. Mirrors `agents/v3.5.1/main.py`.

    Why this and not Tier 0: top-10 fingerprints show mean fleet 38 vs
    midpack 29 and mean garrison-at-launch 10.6 vs 22 (knowledge-base/
    concepts/top-performer-strategies.md:171-184). The single behavioural
    change that captures most of that gap is `aggressive=True` in the
    snipe builder — it sizes launches as a fraction of source garrison
    (0.7) rather than minimum-viable target.ships+1.

    Use Tier 1 as the rollout policy when modelling opponents above the
    μ≈1100 band; use Tier 0 for parity with prior probes / lower-ladder
    self-play.
    """
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    model = WorldModel.from_world(world)
    missions = (
        propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)


# ---------------------------------------------------------------------------
# Tier 2 — placeholder for the trained launch-decision classifier
# ---------------------------------------------------------------------------


def trained_logreg_policy(obs: Any) -> list:
    """Reserved for the trained launch-decision classifier.

    Will read a model artifact (≤200-float logistic regression weights)
    from a fixed path under the submission bundle, score each candidate
    mission with the 24-dim feature schema at
    `data/shot_validator/schema.json`, and emit the argmax-launch set.

    Not implemented in this branch — see plan section "Deliverable 2 /
    Tier 2". Fallback to Tier 1 so downstream consumers can wire it up
    without crashing.
    """
    return top_tier_mirror_policy(obs)


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


_TIER_REGISTRY: dict[int, Policy] = {
    0: mirror_self_policy,
    1: top_tier_mirror_policy,
    2: trained_logreg_policy,
}


def make_opp_policy(tier: int = 1) -> Policy:
    """Return a `Callable(obs) -> action` for the given tier.

    Tier defaults to 1 (top-tier mirror) because that's the better
    proxy for the average ladder opponent above μ1100; downgrade to
    Tier 0 in unit tests / parity replays where the legacy Phase 2
    behavior is wanted.
    """
    if tier not in _TIER_REGISTRY:
        raise ValueError(f"unknown opp_model tier: {tier}")
    return _TIER_REGISTRY[tier]


def predict_opponent_action(obs: Any, tier: int = 1) -> list:
    """One-shot prediction — convenience wrapper around `make_opp_policy`.

    The opp's `player` field on `obs` determines which seat is acting;
    we don't override it here. Caller should ensure `obs.player ==
    opp_id`."""
    return make_opp_policy(tier)(obs)


def opponent_action_distribution(
    obs: Any, *, tier: int = 1, samples: int = 1,
) -> list[list]:
    """Return `samples` plausible action sets, weighted by prior probability.

    Stubbed at Tiers 0/1: returns `[deterministic_action] * samples`
    because the underlying policy is deterministic. Reserved for the
    Tier-2 (trained) consumer, which will sample from the
    logistic-regression score distribution. PIMC-style rollout
    consumers (next session) call this; for now it's a single-point
    distribution.
    """
    if samples < 1:
        raise ValueError("samples must be >= 1")
    base = predict_opponent_action(obs, tier=tier)
    return [base for _ in range(samples)]

# === inlined: lib/v7_search.py ===


import math
import time
from typing import Any, Callable, Iterable

fs_clone = clone
fs_from_obs = from_obs
fs_step = step
fleet_speed = speed


# ---------------------------------------------------------------------------
# Archetype presets — frozen constants from the top-10 fingerprint
# (knowledge-base/concepts/top-performer-strategies.md).
# ---------------------------------------------------------------------------

# Each preset is (aggressive_fraction, max_targets_per_source, reinforce_priority_boost).
# Reading: concentrated artillery (Isaiah / bowwowforeach) empties the
# source onto one big target; saturation skirmisher (flg / Ebi) spreads
# medium fleets over multiple targets; defensive boosts reinforce
# missions over snipe.
ARCHETYPE_PRESETS = {
    "baseline":     (0.7,  1, 1.0),   # v3.5.1 default
    "concentrated": (0.95, 1, 1.0),   # Isaiah-style: full source onto top-1
    "saturation":   (0.5,  3, 1.0),   # flg-style: medium fleets, 3 targets
    "defensive":    (0.7,  1, 3.0),   # reinforce priority × 3
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ranked_snipe_missions_by_source(
    world: World, model: WorldModel,
) -> dict[int, list[Mission]]:
    """{src_id: [Mission descending by score]} for every owned source.

    Uses the existing `propose_snipe_missions(aggressive=True)` — same
    set v3.5.1 considers. We re-rank per source so each source has its
    own top-K ordering."""
    missions = propose_snipe_missions(world, model, aggressive=True)
    by_src: dict[int, list[Mission]] = {}
    for m in missions:
        by_src.setdefault(m.src_id, []).append(m)
    for src_id in by_src:
        by_src[src_id].sort(key=lambda m: -m.score)
    return by_src


def _action_from_intents(
    intents: list[Intent], obs: Any, model: WorldModel | None = None,
) -> list[list]:
    """Run the realize pipeline to convert Intents into env-format actions."""
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)


def _override_one_source(
    incumbent_intents: list[Intent], override: Intent,
) -> list[Intent]:
    """Replace incumbent's intent from `override.src_id` with `override`.
    If the incumbent had no launch from that source, append."""
    out = [i for i in incumbent_intents if i.src_id != override.src_id]
    out.append(override)
    return out


def _aggressive_size(
    src_ships: int, target_min: int,
    *, fraction: float, reserve: int = 5,
) -> int:
    """v3.5.1's aggressive sizing formula, parameterized by fraction.

    Mirrors `lib/missions/snipe.py::AGGRESSIVE_FRACTION` semantics:
    base = min(src.ships * fraction, src.ships - reserve), clamped
    above target_min. With fraction=0.7 / reserve=5 this matches
    v3.5.1 exactly. Other fractions produce the concentrated /
    saturation archetypes."""
    if src_ships <= 12:
        return target_min
    fraction_size = max(1, int(src_ships * fraction))
    cap = max(1, int(src_ships) - reserve)
    return max(target_min, min(fraction_size, cap))


def _build_incumbent_intents(
    world: World, model: WorldModel, *, include_recapture: bool = False,
) -> list[Intent]:
    """v3.5.1's mission set: aggressive snipe + reinforce, run through
    settle_plan. Used as the parity-floor candidate.

    `include_recapture=True` (v7.2+) also adds recapture missions.
    The recapture proposer's score is calibrated to snipe scale and
    top-K capped (see lib/missions/recapture.py); without those fixes
    recapture dominates settle_plan and regresses
    (audit/2026-05-12-recapture-wireup-ab.md).
    """
    missions = (
        propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )
    if include_recapture:
        missions = missions + propose_recapture_missions(world, model)
    chosen = settle_plan(missions, world, model)
    return chosen


# ---------------------------------------------------------------------------
# Enumerators (one per mode)
# ---------------------------------------------------------------------------


def _enumerate_drop_one(incumbent_action: list[list]) -> list[list[list]]:
    """Incumbent + each drop-one variant. Floor for v7 framework lift."""
    if not incumbent_action:
        return [[]]
    cands: list[list[list]] = [list(incumbent_action)]
    for i in range(len(incumbent_action)):
        cands.append([m for j, m in enumerate(incumbent_action) if j != i])
    return cands


def _enumerate_target_swap(
    world: World, model: WorldModel,
    incumbent_intents: list[Intent], incumbent_action: list[list],
    obs: Any,
) -> list[list[list]]:
    """For each owned source in the incumbent, swap to its runner-up
    snipe target. Generates at most N additional candidates."""
    cands: list[list[list]] = [list(incumbent_action)]
    by_src = _ranked_snipe_missions_by_source(world, model)
    incumbent_by_src = {i.src_id: i for i in incumbent_intents}
    for src_id, ranked in by_src.items():
        if len(ranked) < 2:
            continue
        # Find the top mission that's NOT the incumbent's target choice.
        cur_target = (
            incumbent_by_src[src_id].target_id
            if src_id in incumbent_by_src else None
        )
        alt = next((m for m in ranked if m.target_id != cur_target), None)
        if alt is None:
            continue
        new_intents = _override_one_source(incumbent_intents, alt.to_intent())
        cands.append(_action_from_intents(new_intents, obs, model))
    return cands


def _enumerate_ship_sweep(
    world: World, model: WorldModel,
    incumbent_intents: list[Intent], incumbent_action: list[list],
    obs: Any,
) -> list[list[list]]:
    """For each owned source's top-target mission, sweep ships in
    {min-viable, half, 0.95×}. At most 3N additional candidates."""
    cands: list[list[list]] = [list(incumbent_action)]
    by_src = _ranked_snipe_missions_by_source(world, model)
    incumbent_by_src = {i.src_id: i for i in incumbent_intents}
    for src_id, ranked in by_src.items():
        if not ranked:
            continue
        src = world.planets_by_id.get(src_id)
        if src is None or src.ships <= 1:
            continue
        # Use the incumbent's chosen target if it exists for this source,
        # otherwise the top-ranked target from the missions.
        chosen_target_id = (
            incumbent_by_src[src_id].target_id
            if src_id in incumbent_by_src
            else ranked[0].target_id
        )
        target = world.planets_by_id.get(chosen_target_id)
        if target is None:
            continue
        target_min = max(1, int(target.ships) + 1)
        for fraction in (0.5, 0.95):
            ships = _aggressive_size(
                int(src.ships), target_min, fraction=fraction,
            )
            if ships <= 0 or ships > src.ships:
                continue
            # Build the swap intent at this fraction.
            new_intent = Intent(
                src_id=src_id, target_id=chosen_target_id, ships=ships,
            )
            new_intents = _override_one_source(incumbent_intents, new_intent)
            cand = _action_from_intents(new_intents, obs, model)
            if cand and cand != incumbent_action:
                cands.append(cand)
    return cands


def _enumerate_archetype(
    world: World, model: WorldModel, obs: Any,
) -> list[list[list]]:
    """Generate four full-action bundles under preset archetypes.

    Each preset re-derives the snipe+reinforce mission set and runs it
    through settle_plan. The realize pipeline is the same; only the
    per-mission scoring weights and ship-sizing change.
    """
    cands: list[list[list]] = []
    # The "baseline" preset == v3.5.1 incumbent, which the caller
    # always includes first via enumerate_candidates.
    for name, (fraction, max_per_src, reinforce_boost) in ARCHETYPE_PRESETS.items():
        if name == "baseline":
            continue
        # Re-derive missions with the preset's aggressive_fraction.
        snipe = _snipe_missions_with_fraction(
            world, model, fraction=fraction, max_targets_per_source=max_per_src,
        )
        reinforce = propose_reinforce_missions(world, model)
        if reinforce_boost != 1.0:
            for r in reinforce:
                r.score = r.score * reinforce_boost
        chosen = settle_plan(snipe + reinforce, world, model)
        cand = _action_from_intents(chosen, obs, model)
        cands.append(cand)
    return cands


def _snipe_missions_with_fraction(
    world: World, model: WorldModel,
    *, fraction: float, max_targets_per_source: int,
) -> list[Mission]:
    """Reuse propose_snipe_missions(aggressive=True) for ship sizing,
    then post-filter each source to only its top-N missions to enforce
    the archetype's `max_targets_per_source` policy.

    A full re-implementation would pass `fraction` into the proposer,
    but the proposer's `AGGRESSIVE_FRACTION` is a module constant. We
    re-size each emitted Mission's `ships` here to match the preset's
    fraction — same effect, no patching of the lib module.
    """
    base = propose_snipe_missions(world, model, aggressive=True)
    # Re-size to the preset's fraction.
    resized: list[Mission] = []
    for m in base:
        src = world.planets_by_id.get(m.src_id)
        if src is None:
            continue
        target = world.planets_by_id.get(m.target_id)
        if target is None:
            continue
        target_min = max(1, int(target.ships) + 1)
        ships = _aggressive_size(int(src.ships), target_min, fraction=fraction)
        m.ships = ships
        resized.append(m)
    # Per-source top-N filter.
    by_src: dict[int, list[Mission]] = {}
    for m in resized:
        by_src.setdefault(m.src_id, []).append(m)
    out: list[Mission] = []
    for src_id, ranked in by_src.items():
        ranked.sort(key=lambda m: -m.score)
        out.extend(ranked[:max_targets_per_source])
    return out


def _enumerate_hungarian(
    world: World, model: WorldModel,
    incumbent_action: list[list], obs: Any,
) -> list[list[list]]:
    """Globally-coordinated (source × target) assignment as one
    additional candidate. Uses `scipy.optimize.linear_sum_assignment`.

    The score matrix is `propose_snipe_missions` scores filtered to
    our owned sources × non-owned targets. The assignment forces each
    source to ONE target (no double-commit), unlike settle_plan's
    per-source greedy with a same-turn ledger.

    Returns `[incumbent, hungarian_alternative]` — caller appends.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        # Bundle environment doesn't ship scipy — fall back to incumbent
        # only. This is an acceptable degradation.
        return [list(incumbent_action)]

    cands: list[list[list]] = [list(incumbent_action)]

    missions = propose_snipe_missions(world, model, aggressive=True)
    if not missions:
        return cands
    sources = sorted({m.src_id for m in missions})
    targets = sorted({m.target_id for m in missions})
    if not sources or not targets:
        return cands

    src_to_row = {s: i for i, s in enumerate(sources)}
    tgt_to_col = {t: j for j, t in enumerate(targets)}
    # Initialise with a very negative score (linear_sum_assignment minimises
    # cost; we negate scores so high score = low cost).
    NEG = 1e6
    cost = [[NEG for _ in targets] for _ in sources]
    by_pair: dict[tuple[int, int], Mission] = {}
    for m in missions:
        i = src_to_row[m.src_id]
        j = tgt_to_col[m.target_id]
        if cost[i][j] > -m.score:
            cost[i][j] = -m.score
            by_pair[(m.src_id, m.target_id)] = m

    # Pad to square matrix if rectangular — linear_sum_assignment supports
    # rectangular but a small problem is fine.
    row_ind, col_ind = linear_sum_assignment(cost)
    chosen: list[Mission] = []
    for i, j in zip(row_ind, col_ind):
        if cost[i][j] >= NEG / 2:
            continue
        src_id = sources[i]
        tgt_id = targets[j]
        m = by_pair.get((src_id, tgt_id))
        if m is not None:
            chosen.append(m)
    if not chosen:
        return cands
    # settle_plan still applies per-source uniqueness + same-turn ledger;
    # run it on `chosen` so the bundle uses the same gating discipline.
    intents = settle_plan(chosen, world, model)
    cand = _action_from_intents(intents, obs, model)
    if cand:
        cands.append(cand)
    return cands


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enumerate_candidates(
    world: World,
    model: WorldModel,
    *,
    enumerator_mode: str,
    incumbent_intents: list[Intent],
    incumbent_action: list[list],
    obs: Any,
) -> list[list[list]]:
    """Generate candidate action bundles. Incumbent is ALWAYS index 0
    so the watchdog fallback never regresses below v3.5.1."""
    if enumerator_mode == "drop_one":
        return _enumerate_drop_one(incumbent_action)
    if enumerator_mode == "target_swap":
        return _enumerate_target_swap(
            world, model, incumbent_intents, incumbent_action, obs,
        )
    if enumerator_mode == "ship_sweep":
        return _enumerate_ship_sweep(
            world, model, incumbent_intents, incumbent_action, obs,
        )
    if enumerator_mode == "archetype":
        archetypes = _enumerate_archetype(world, model, obs)
        # baseline first, then the other three.
        return [list(incumbent_action)] + archetypes
    if enumerator_mode == "hungarian":
        return _enumerate_hungarian(world, model, incumbent_action, obs)
    if enumerator_mode == "combined":
        # Union of every mode's candidates, with the incumbent only once.
        seen: list[list[list]] = [list(incumbent_action)]
        seen_keys = {_action_key(incumbent_action)}
        for mode in ("drop_one", "target_swap", "ship_sweep",
                     "archetype", "hungarian"):
            for cand in enumerate_candidates(
                world, model, enumerator_mode=mode,
                incumbent_intents=incumbent_intents,
                incumbent_action=incumbent_action,
                obs=obs,
            ):
                k = _action_key(cand)
                if k not in seen_keys:
                    seen.append(cand)
                    seen_keys.add(k)
        return seen
    raise ValueError(f"unknown enumerator_mode: {enumerator_mode}")


def _action_key(action: list[list]) -> tuple:
    """Hashable key for deduplicating action bundles. Coarse rounding on
    the angle so jittered duplicates aren't double-counted."""
    return tuple(
        (int(m[0]), round(float(m[1]), 5), int(m[2])) for m in action
    )


def _infer_num_seats(world: World) -> int:
    """Best-effort player-count inference from the obs.

    The kaggle_environments obs doesn't directly expose `num_agents` —
    only `player` (our seat). We infer from the highest owner ID seen
    across planets + fleets. 2P games yield owners in {0, 1} ∪ {-1};
    4P yields {0, 1, 2, 3} ∪ {-1}.
    """
    max_owner = world.my_id
    for p in world.planets_by_id.values():
        if p.owner > max_owner:
            max_owner = p.owner
    raw = world.obs_raw
    fleets = (
        raw.get("fleets", [])
        if isinstance(raw, dict)
        else getattr(raw, "fleets", [])
    )
    for f in fleets or []:
        owner = int(f[1])
        if owner > max_owner:
            max_owner = owner
    return max_owner + 1 if max_owner >= 0 else 1


def score_candidate(
    snap: Snapshot,
    action: list[list],
    *,
    my_id: int = 0,
    K: int = 10,
    opp_tier: int = 1,
    value_fn: Callable | None = None,
) -> float:
    """Rollout score for `action` under our seat.

    The opponent plays the requested tier policy throughout the
    rollout. Our seat plays `action` on the first tick, then the
    top-tier mirror policy thereafter.

    `value_fn(observation, my_id) -> float` is the leaf-state scoring
    head. Defaults to `delta_us_minus_them` (our minus their total
    ships) — the Phase-2-validated scalar. v7.3+ passes
    `lib.lookahead_planner.evaluate_value` for production-share +
    denial + survivor bonus.
    """
    if snap.num_seats != 2:
        raise ValueError(f"v7 score_candidate is 2P only (got {snap.num_seats})")
    clone = fs_clone(snap)
    opp_id = 1 - my_id

    opp_policy = make_opp_policy(opp_tier)
    our_followup_policy = top_tier_mirror_policy

    # First step: forced action for us; opp plays its policy.
    a_opp = opp_policy(clone.state[opp_id].observation)
    actions = [None, None]
    actions[my_id] = action
    actions[opp_id] = a_opp
    if not clone.done:
        clone = fs_step(clone, actions, in_place=True)

    # Remaining K-1 steps: both seats play top-tier mirror.
    for _ in range(max(0, K - 1)):
        if clone.done:
            break
        a0 = our_followup_policy(clone.state[0].observation)
        a1 = our_followup_policy(clone.state[1].observation)
        clone = fs_step(clone, [a0, a1], in_place=True)

    if value_fn is None:
        return delta_us_minus_them(clone, my_id)
    return value_fn(clone.state[my_id].observation, my_id)


def score_candidate_symmetric(
    snap: Snapshot,
    action: list[list],
    *,
    K: int = 10,
    opp_tier: int = 1,
) -> float:
    """Seat-symmetric variant of `score_candidate`.

    Runs the rollout twice — once with us at seat 0 (opp at seat 1)
    and once with us at seat 1 (opp at seat 0) — and averages the
    `delta_us_minus_them` results from our POV in each. Cancels the
    env's documented P1-favoring tie-break bias that otherwise leaks
    into the maximin payoff matrix.

    Ported from `score_joint_action_symmetric` in
    `origin/claude/game-theory-strategy-analysis-0oH4N` but adapted
    to operate on Snapshots (so we keep fast_sim's 183× speedup).

    Cost: 2× score_candidate.
    """
    a = score_candidate(snap, action, my_id=0, K=K, opp_tier=opp_tier)
    b = score_candidate(snap, action, my_id=1, K=K, opp_tier=opp_tier)
    return (a + b) / 2.0


def score_joint(
    snap: Snapshot,
    our_action: list[list],
    opp_action: list[list],
    *,
    my_id: int = 0,
    K: int = 10,
    value_fn: Callable | None = None,
) -> float:
    """Snapshot variant of `lib/lookahead.score_joint_action`.

    Both first-turn actions are forced; turns 2..K both seats play
    top_tier_mirror. Returns the leaf-state value via `value_fn`
    (default: `delta_us_minus_them`).
    """
    if snap.num_seats != 2:
        raise ValueError(f"score_joint is 2P only (got {snap.num_seats})")
    clone = fs_clone(snap)
    opp_id = 1 - my_id
    actions = [None, None]
    actions[my_id] = our_action
    actions[opp_id] = opp_action
    if not clone.done:
        clone = fs_step(clone, actions, in_place=True)
    for _ in range(max(0, K - 1)):
        if clone.done:
            break
        a0 = top_tier_mirror_policy(clone.state[0].observation)
        a1 = top_tier_mirror_policy(clone.state[1].observation)
        clone = fs_step(clone, [a0, a1], in_place=True)
    if value_fn is None:
        return delta_us_minus_them(clone, my_id)
    return value_fn(clone.state[my_id].observation, my_id)


def score_joint_symmetric(
    snap: Snapshot,
    our_action: list[list],
    opp_action: list[list],
    *,
    K: int = 10,
    value_fn: Callable | None = None,
) -> float:
    """Seat-symmetric joint scorer. Used by the maximin overlay."""
    a = score_joint(snap, our_action, opp_action, my_id=0, K=K, value_fn=value_fn)
    b = score_joint(snap, our_action, opp_action, my_id=1, K=K, value_fn=value_fn)
    return (a + b) / 2.0


def _drop_smallest(action: list[list]) -> list[list]:
    """Return `action` with its smallest-ship launch removed.

    Mirrors the drop_smallest function in v7_minimax (ported from
    `origin/claude/game-theory-strategy-analysis-0oH4N`'s
    agents/v7_minimax/main.py:98-117). Ties broken by removing the
    EARLIEST launch among smallest, which is σ-deterministic given
    upstream ordering.
    """
    if not action:
        return []
    if len(action) == 1:
        return []
    min_idx = 0
    min_ships = int(action[0][2])
    for i, la in enumerate(action[1:], start=1):
        if int(la[2]) < min_ships:
            min_ships = int(la[2])
            min_idx = i
    return [la for i, la in enumerate(action) if i != min_idx]


def _opp_incumbent_action(world: World, obs: Any, opp_id: int) -> list[list]:
    """Compute the opponent's incumbent action via v3.5.1 pipeline
    from the opp's POV.

    We don't have a clean way to swap `world.my_id` (it's frozen at
    construction), so we rebuild World from a copy of obs with
    `player=opp_id`. This is the same technique v7_minimax uses
    (`_swap_obs_player` in their main.py).
    """
    if isinstance(obs, dict):
        obs2 = dict(obs)
        obs2["player"] = opp_id
    else:
        keys = (
            "player", "planets", "fleets", "angular_velocity",
            "initial_planets", "comet_planet_ids", "comets",
            "step", "next_fleet_id",
        )
        obs2 = {}
        for k in keys:
            v = getattr(obs, k, None)
            if v is not None:
                obs2[k] = v
        obs2["player"] = opp_id
    opp_world = World.from_obs(obs2)
    if not opp_world.planets_by_id:
        return []
    opp_model = WorldModel.from_world(opp_world)
    missions = (
        propose_snipe_missions(opp_world, opp_model, aggressive=True)
        + propose_reinforce_missions(opp_world, opp_model)
    )
    intents = settle_plan(missions, opp_world, opp_model)
    return realize(intents, obs2, mechanisms=DEFAULT_MECHANISMS, model=opp_model)


def choose_maximin(
    obs: Any,
    configuration: Any = None,
    *,
    K: int = 10,
    wallclock_ms: float = 700.0,
    my_id: int | None = None,
    use_symmetric: bool = True,
    include_recapture: bool = False,
    value_fn: Callable | None = None,
) -> list[list]:
    """v7.1 maximin overlay.

    Per turn:
      1. Build N=N+1 our candidates via `_enumerate_drop_one(incumbent)`
         (incumbent + drop-each-launch).
      2. Build M=2 opp candidates: opp's v3.5.1 incumbent + drop-smallest.
      3. Score every (our_i, opp_j) cell via `score_joint_symmetric`
         (Snapshot, K-step rollout, symmetric average).
      4. Pick i* = argmax_i (min_j P[i,j]). Tie-break: prefer row 0
         (= our incumbent) — σ-equivariant fallback.

    Wallclock watchdog (`wallclock_ms`, default 700) bails the inner
    loop if budget exhausted. Row 0 (incumbent) is ALWAYS evaluated
    in full first so its worst-case is honest. 4P games fall back to
    the incumbent (no maximin guarantee at n>2).
    """
    t_start = time.perf_counter()

    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    if my_id is None:
        my_id = world.my_id
    model = WorldModel.from_world(world)

    incumbent_intents = _build_incumbent_intents(
        world, model, include_recapture=include_recapture,
    )
    incumbent_action = _action_from_intents(incumbent_intents, obs, model)

    # 4P fallback — maximin is 2P-only.
    if _infer_num_seats(world) != 2:
        return incumbent_action

    opp_id = 1 - my_id
    # Our candidate class: incumbent + each drop-one variant.
    C = _enumerate_drop_one(incumbent_action)
    if len(C) <= 1:
        return incumbent_action
    # Opp candidate class M=2.
    O_inc = _opp_incumbent_action(world, obs, opp_id)
    O_drop = _drop_smallest(O_inc)
    O = [O_inc] if not O_drop or O_drop == O_inc else [O_inc, O_drop]

    snap = fs_from_obs(obs, configuration, episode_seed=0, num_seats=2)
    if use_symmetric:
        def score_fn(s, ours, opps, *, K=K):
            return score_joint_symmetric(s, ours, opps, K=K, value_fn=value_fn)
    else:
        def score_fn(s, ours, opps, *, K=K):
            return score_joint(s, ours, opps, my_id=my_id, K=K, value_fn=value_fn)

    N = len(C)
    M = len(O)
    P: list[list[float]] = [[float("-inf")] * M for _ in range(N)]
    unfilled: list[list[bool]] = [[True] * M for _ in range(N)]

    # Row 0 (incumbent) first, full row. Then i>=1 row-by-row with bail.
    for i in range(N):
        for j in range(M):
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            if i > 0 and elapsed_ms > wallclock_ms:
                break
            try:
                P[i][j] = score_fn(snap, C[i], O[j], K=K)
                unfilled[i][j] = False
            except Exception:
                P[i][j] = float("-inf")
                unfilled[i][j] = False
        else:
            continue
        break  # exited inner via budget bail

    # Maximin: argmax_i (min_j P[i,j]) over evaluated cells, tie → row 0.
    best_i = 0
    best_worst = float("-inf")
    for i in range(N):
        evaluated = [P[i][j] for j in range(M) if not unfilled[i][j]]
        if not evaluated:
            worst = float("-inf")
        else:
            worst = min(evaluated)
        if worst > best_worst:
            best_worst = worst
            best_i = i
    return C[best_i]


def score_candidate_4p(
    snap: Snapshot,
    action: list[list],
    *,
    my_id: int,
    K: int = 8,
    value_fn: Callable | None = None,
) -> float:
    """Rollout score for a 4P candidate action.

    All 3 non-pov seats play `top_tier_mirror_policy`. Our seat plays
    `action` on tick 0, then `top_tier_mirror_policy` for the rest.
    Scoring head: `value_fn(state[my_id].observation, my_id)` at
    terminal — defaults to "our ships − max(other seat ships)" which
    rewards keeping the lead vs the best-remaining-opponent (better
    proxy for 4P first-place than total-sum-of-them).
    """
    if snap.num_seats != 4:
        raise ValueError(f"score_candidate_4p needs num_seats=4 (got {snap.num_seats})")
    clone = fs_clone(snap)

    # First step: forced action for us; all 3 opps play top_tier_mirror.
    actions: list[list[list]] = [[] for _ in range(4)]
    for seat in range(4):
        if seat == my_id:
            actions[seat] = action
        else:
            actions[seat] = top_tier_mirror_policy(clone.state[seat].observation)
    if not clone.done:
        clone = fs_step(clone, actions, in_place=True)

    # Remaining K-1 steps: all 4 seats play top_tier_mirror.
    for _ in range(max(0, K - 1)):
        if clone.done:
            break
        acts = [top_tier_mirror_policy(clone.state[seat].observation) for seat in range(4)]
        clone = fs_step(clone, acts, in_place=True)

    if value_fn is not None:
        return value_fn(clone.state[my_id].observation, my_id)

    # Default 4P scoring: our ships − max(other seat ships).
    # Better proxy for "did we keep the lead vs the best-remaining-
    # opponent" than (our − sum_others), which is dominated by total
    # ship counts.
    from collections import defaultdict
    totals: dict[int, float] = defaultdict(float)
    obs0 = clone.state[my_id].observation
    for p in obs0.planets:
        if int(p[1]) >= 0:
            totals[int(p[1])] += float(p[5])
    for f in obs0.fleets:
        if int(f[1]) >= 0:
            totals[int(f[1])] += float(f[6])
    ours = totals.get(my_id, 0.0)
    others = [v for k, v in totals.items() if k != my_id and k >= 0]
    best_opp = max(others) if others else 0.0
    return ours - best_opp


def choose_4p(
    obs: Any,
    configuration: Any = None,
    *,
    K: int = 8,
    wallclock_ms: float = 700.0,
    my_id: int | None = None,
    include_recapture: bool = True,
    value_fn: Callable | None = None,
) -> list[list]:
    """v7.4 — 4P drop-one chooser.

    No maximin (no Nash guarantee at n>2). All 3 opps modeled as
    top_tier_mirror; we score drop-one candidates and pick argmax.
    Falls back to incumbent if the watchdog trips or no candidate
    strictly beats it.
    """
    t_start = time.perf_counter()

    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    if my_id is None:
        my_id = world.my_id
    model = WorldModel.from_world(world)

    incumbent_intents = _build_incumbent_intents(
        world, model, include_recapture=include_recapture,
    )
    incumbent_action = _action_from_intents(incumbent_intents, obs, model)

    # Build candidate set: incumbent + drop-each-launch.
    candidates = _enumerate_drop_one(incumbent_action)
    if len(candidates) <= 1:
        return incumbent_action

    snap = fs_from_obs(obs, configuration, episode_seed=0, num_seats=4)

    best_action = incumbent_action
    best_score = float("-inf")
    incumbent_scored = False
    for cand in candidates:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if elapsed_ms > wallclock_ms:
            break
        try:
            score = score_candidate_4p(
                snap, cand, my_id=my_id, K=K, value_fn=value_fn,
            )
        except Exception:
            continue
        if not incumbent_scored:
            incumbent_scored = True
            best_score = score
            best_action = list(cand)
            continue
        if score > best_score:
            best_score = score
            best_action = list(cand)
    return best_action


def choose_simple_2p(
    obs: Any,
    configuration: Any = None,
    *,
    K: int = 10,
    wallclock_ms: float = 700.0,
    my_id: int | None = None,
    include_recapture: bool = True,
    value_fn: Callable | None = None,
) -> list[list]:
    """2P drop-one chooser WITHOUT maximin overlay.

    This is what v7.1 maximin should have been but wasn't: pure
    argmax over drop-one candidates with σ-equiv-enabled incumbent.
    The maximin variant (`choose_maximin`) lost the A/B because
    its 2×N × symmetric-scoring budget blew the wallclock; the
    simple variant has the same per-candidate cost as v7_0 (proven
    fast enough at 746-816 ms p95) while still getting σ-equiv,
    recapture, and value_fn for free.
    """
    t_start = time.perf_counter()

    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    if my_id is None:
        my_id = world.my_id
    model = WorldModel.from_world(world)

    incumbent_intents = _build_incumbent_intents(
        world, model, include_recapture=include_recapture,
    )
    incumbent_action = _action_from_intents(incumbent_intents, obs, model)

    candidates = _enumerate_drop_one(incumbent_action)
    if len(candidates) <= 1:
        return incumbent_action

    snap = fs_from_obs(obs, configuration, episode_seed=0, num_seats=2)
    best_action = incumbent_action
    best_score = float("-inf")
    incumbent_scored = False
    for cand in candidates:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if elapsed_ms > wallclock_ms:
            break
        try:
            score = score_candidate(
                snap, cand, my_id=my_id, K=K, opp_tier=1, value_fn=value_fn,
            )
        except Exception:
            continue
        if not incumbent_scored:
            incumbent_scored = True
            best_score = score
            best_action = list(cand)
            continue
        if score > best_score:
            best_score = score
            best_action = list(cand)
    return best_action


def choose_simple_with_4p(
    obs: Any,
    configuration: Any = None,
    *,
    K_2p: int = 10,
    K_4p: int = 8,
    wallclock_ms: float = 700.0,
    include_recapture: bool = True,
    value_fn: Callable | None = None,
) -> list[list]:
    """v7.5 entry — auto-routes 2P→choose_simple_2p, 4P→choose_4p.

    No maximin overlay (which regressed at v7.1 A/B). σ-equiv layer
    is library-level (lib/planner + lib/geometry + lib/missions/snipe)
    so it's automatically present. Recapture + value_fn pluggable.
    """
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    n_seats = _infer_num_seats(world)
    if n_seats == 2:
        return choose_simple_2p(
            obs, configuration,
            K=K_2p, wallclock_ms=wallclock_ms,
            include_recapture=include_recapture,
            value_fn=value_fn,
        )
    if n_seats == 4:
        return choose_4p(
            obs, configuration,
            K=K_4p, wallclock_ms=wallclock_ms,
            include_recapture=include_recapture,
            value_fn=value_fn,
        )
    # 3P or 1P: rare; fall back to incumbent.
    model = WorldModel.from_world(world)
    intents = _build_incumbent_intents(world, model, include_recapture=include_recapture)
    return _action_from_intents(intents, obs, model)


def choose_with_4p(
    obs: Any,
    configuration: Any = None,
    *,
    K_2p: int = 10,
    K_4p: int = 8,
    wallclock_ms: float = 700.0,
    use_symmetric: bool = True,
    include_recapture: bool = True,
    value_fn: Callable | None = None,
) -> list[list]:
    """Auto-routes 2P → choose_maximin, 4P → choose_4p.

    Combines the v7.1 maximin overlay (with σ-equiv + symmetric
    scoring) for 2P and the v7.4 4-seat drop-one rollout for 4P. v7.5
    final agent uses this entry point.
    """
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    n_seats = _infer_num_seats(world)
    if n_seats == 2:
        return choose_maximin(
            obs, configuration,
            K=K_2p, wallclock_ms=wallclock_ms,
            use_symmetric=use_symmetric,
            include_recapture=include_recapture,
            value_fn=value_fn,
        )
    if n_seats == 4:
        return choose_4p(
            obs, configuration,
            K=K_4p, wallclock_ms=wallclock_ms,
            include_recapture=include_recapture,
            value_fn=value_fn,
        )
    # 3P or 1P: rare; fall back to incumbent (safest).
    model = WorldModel.from_world(world)
    intents = _build_incumbent_intents(world, model, include_recapture=include_recapture)
    return _action_from_intents(intents, obs, model)


def choose(
    obs: Any,
    configuration: Any = None,
    *,
    enumerator_mode: str,
    K: int = 10,
    wallclock_ms: float = 700.0,
    my_id: int | None = None,
) -> list[list]:
    """End-to-end: build incumbent, enumerate, score with watchdog,
    return argmax. Always returns the incumbent if no candidate
    scores strictly higher (parity floor).

    `my_id` defaults to `obs.player`. `configuration` is forwarded to
    `fs_from_obs`; if `None`, defaults are used (live ladder
    scrubs the episode seed anyway).
    """
    t_start = time.perf_counter()

    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    if my_id is None:
        my_id = world.my_id
    model = WorldModel.from_world(world)

    incumbent_intents = _build_incumbent_intents(world, model)
    incumbent_action = _action_from_intents(incumbent_intents, obs, model)

    # 2P-only guard: the rollout's opp_model assumes a single opponent
    # seat. In 4P games we'd need to model 3 opponents simultaneously,
    # and the 2P Snapshot of a 4P obs systematically prefers "do
    # nothing" (the other 2 opponents are invisible to the simulator).
    # Fall back to the v3.5.1 incumbent — parity floor preserved.
    if _infer_num_seats(world) != 2:
        return incumbent_action

    # Snapshot for rollout. Episode seed unknown on the live ladder; this
    # is the same caveat documented in lib/lookahead.py and lib/fast_sim.
    snap = fs_from_obs(obs, configuration, episode_seed=0, num_seats=2)

    candidates = enumerate_candidates(
        world, model,
        enumerator_mode=enumerator_mode,
        incumbent_intents=incumbent_intents,
        incumbent_action=incumbent_action,
        obs=obs,
    )
    if len(candidates) <= 1:
        return incumbent_action

    best_action = incumbent_action  # incumbent is always candidates[0]
    best_score = float("-inf")
    incumbent_scored = False
    for cand in candidates:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if elapsed_ms > wallclock_ms:
            break
        score = score_candidate(snap, cand, my_id=my_id, K=K, opp_tier=1)
        if not incumbent_scored:
            # The first candidate is the incumbent — pin its score so
            # ties prefer the parity floor.
            incumbent_scored = True
            best_score = score
            best_action = list(cand)
            continue
        if score > best_score:
            best_score = score
            best_action = list(cand)
    return best_action

# === inlined: lib/candidate_portfolios.py ===


from collections import defaultdict
from dataclasses import dataclass



@dataclass
class Portfolio:
    """A labelled mission list to be ranked by the Sim<K> scorer."""

    label: str
    missions: list[Mission]


def _incumbent_missions(world: World, model: WorldModel) -> list[Mission]:
    """v3.5.1's mission set — aggressive snipe + reinforce."""
    return (
        propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )


def _conservative_missions(world: World, model: WorldModel) -> list[Mission]:
    """v3_snipe's mission set — minimum-viable snipe + reinforce."""
    return (
        propose_snipe_missions(world, model, aggressive=False)
        + propose_reinforce_missions(world, model)
    )


def _per_source_swap(missions: list[Mission]) -> list[Mission] | None:
    """Drop top-1 for the source with the smallest top-1 / top-2 score gap.

    Returns None if no source has at least 2 missions (no swap possible)
    or if there are no missions at all.
    """
    if not missions:
        return None
    by_src: dict[int, list[Mission]] = defaultdict(list)
    for m in missions:
        by_src[m.src_id].append(m)
    # Sort each bucket high → low by score, find the source with
    # the smallest (top1 - top2) gap.
    candidates = []
    for src_id, bucket in by_src.items():
        if len(bucket) < 2:
            continue
        bucket.sort(key=lambda m: -m.score)
        gap = bucket[0].score - bucket[1].score
        candidates.append((gap, src_id, bucket[0]))
    if not candidates:
        return None
    # Smallest gap = most-marginal greedy decision.
    _gap, swap_src, top1 = min(candidates, key=lambda t: t[0])
    return [m for m in missions if not (m.src_id == swap_src and m is top1)]


def _drop_weakest_source(missions: list[Mission]) -> list[Mission] | None:
    """Drop ALL missions from the source whose best score is the lowest.

    Equivalent to "idle the weakest source this turn." Returns None if
    there are fewer than 2 sources (dropping the only source = no-op
    duplicate, the noop portfolio covers it).
    """
    if not missions:
        return None
    by_src: dict[int, list[Mission]] = defaultdict(list)
    for m in missions:
        by_src[m.src_id].append(m)
    if len(by_src) < 2:
        return None
    # Best score per source; weakest is the one whose best is lowest.
    weakest_src = min(
        by_src.keys(), key=lambda s: max(m.score for m in by_src[s])
    )
    return [m for m in missions if m.src_id != weakest_src]


def generate_portfolios(
    world: World,
    model: WorldModel,
    incumbent_missions: list[Mission] | None = None,
) -> list[Portfolio]:
    """Build ≤ 5 mission portfolios for the lookahead scorer to rank.

    `incumbent_missions` may be passed in if the caller already built
    them (avoiding a duplicate proposer call); otherwise this rebuilds
    them. The incumbent is always portfolios[0] so the scorer's "score
    incumbent first" loop has a safe fallback.
    """
    incumbent = (
        incumbent_missions
        if incumbent_missions is not None
        else _incumbent_missions(world, model)
    )
    portfolios: list[Portfolio] = [Portfolio("incumbent", incumbent)]

    conservative = _conservative_missions(world, model)
    # Only add conservative if it differs in at least one ship count from
    # incumbent — otherwise it would settle to the same action.
    if _missions_differ(conservative, incumbent):
        portfolios.append(Portfolio("conservative", conservative))

    swap = _per_source_swap(incumbent)
    if swap is not None and swap != incumbent:
        portfolios.append(Portfolio("per_source_swap", swap))

    drop_weak = _drop_weakest_source(incumbent)
    if drop_weak is not None and drop_weak != incumbent:
        portfolios.append(Portfolio("drop_weakest_source", drop_weak))

    portfolios.append(Portfolio("noop", []))
    return portfolios


def _missions_differ(a: list[Mission], b: list[Mission]) -> bool:
    """Cheap structural compare on (src, target, ships) keys.

    score / mission_class metadata is irrelevant for whether settle_plan
    would emit a different action — only the launch tuple matters.
    """
    def key(ms: list[Mission]):
        return sorted((m.src_id, m.target_id, m.ships) for m in ms)
    return key(a) != key(b)

# === inlined: lib/value_heads.py ===


from typing import Any



# Phase 2 audit established AUC ≈ oracle at K=50. K=10 + 30 extra of
# static substrate ≈ K=40 effective; close enough.
INFLIGHT_EXTRA_HORIZON: int = 30

# How much weight to give the in-flight production credit relative to
# ship-delta. 0.5 chosen so a captured 3-production planet (worth
# ~3*30=90 production-points) approximately balances 90 ships of
# delta. Calibration knob.
INFLIGHT_WEIGHT: float = 0.5


def delta_us_minus_them_obs(obs: Any, my_id: int) -> float:
    """Plain `(our ships) − (their ships)` from a Snapshot's primary
    observation. Phase 2 validated this at AUC ≈ oracle for K=50.

    Renamed from `delta_us_minus_them` to avoid bundle-shadow collision
    with the identically-named `lib.fast_sim.delta_us_minus_them(snap, ...)`.
    The fast_sim version takes a Snapshot; this one takes an obs.
    Same logic, different first-arg type.

    `obs` is `snap.state[my_id].observation` (a `Struct`). Sums
    planet garrisons + in-flight fleet ship counts for owned planets/
    fleets; subtracts each other seat's total.
    """
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    fleets = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])
    ours = 0.0
    theirs = 0.0
    for p in planets:
        owner = int(p[1])
        if owner == my_id:
            ours += float(p[5])
        elif owner >= 0:
            theirs += float(p[5])
    for f in fleets:
        owner = int(f[1])
        if owner == my_id:
            ours += float(f[6])
        elif owner >= 0:
            theirs += float(f[6])
    return ours - theirs


def inflight_value(
    obs: Any, my_id: int,
    *, extra_horizon: int = INFLIGHT_EXTRA_HORIZON,
    weight: float = INFLIGHT_WEIGHT,
) -> float:
    """Composite scoring head: `delta_us_minus_them + weight × inflight_credit`.

    The credit term reads the predicted owner of each planet at
    `step + extra_horizon` from the terminal Snapshot's WorldModel
    (which integrates in-flight fleets). For planets that flip TO
    us within the extended horizon, the credit is `production`.
    Sum across all such planets, weight by `weight`.

    The default weight=0.5 is the calibration knob the v9_inflight
    A/B is gated on. Phase 2 said AUC≈oracle at K=50; this head
    effectively extends K from 10 to ~40 via the static substrate
    while keeping the same rollout cost.

    Empty world (no planets) → returns the base ship-delta only
    (which is 0).
    """
    base = delta_us_minus_them_obs(obs, my_id)
    # Build World from the terminal observation. fast_sim's Snapshot
    # uses Struct, so World.from_obs accepts it.
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return base

    # Build a per-planet timeline that integrates current in-flight
    # fleets out to `extra_horizon`. WorldModel.simulate_planet_timeline
    # is O(horizon) per planet and ~1ms total at horizon=30 for a
    # typical board (see audit/2026-05-12-fast-sim-bench.md).
    model = WorldModel.from_world(world, horizon=extra_horizon)

    bonus = 0.0
    for p in world.planets_by_id.values():
        if p.owner == my_id:
            # Already ours; no in-flight credit needed (counted in base).
            continue
        pred_owner = model.owner_at(p.id, extra_horizon)
        if pred_owner == my_id:
            # We'll own it within extra_horizon → credit the production.
            bonus += float(p.production)
    return base + weight * bonus

# === agent ===
"""v7.0 — drop-one enumerator under the new fast_sim scorer.

Re-runs the v3_lookahead MVP architecture (drop-one candidate set)
but with `lib/fast_sim` (183× faster than env.clone) and
`lib/opp_model.top_tier_mirror_policy` (v3.5.1 mirror, better
opponent proxy than v3_snipe). Measures the FRAMEWORK's contribution
alone; lift here would mean the previous Phase 2 result was budget-
limited, not enumeration-limited.

Fallback: if no candidate scores strictly above the incumbent (or the
watchdog trips), returns v3.5.1's action — parity floor preserved.
"""


# from lib.v7_search import choose  # inlined by bundle_agent.py


def agent(obs, configuration=None):
    return choose(
        obs, configuration,
        enumerator_mode="drop_one",
        K=10,
        wallclock_ms=700.0,
    )
