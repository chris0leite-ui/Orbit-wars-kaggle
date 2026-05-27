# Bundled by scripts/bundle_agent.py from agents/baseline + lib/{geometry,fleet,orbit,aim,combat,world_model,intent,kinematic_table,trajectory,mechanism,mission,scoring,missions/snipe,missions/reinforce,missions/recapture,missions/opening,missions/drain,missions/gang_up,missions/opp_archetypes,planner,lookahead,lookahead_planner,game/interpreter,fast_sim,opp_model,v7_search,candidate_portfolios,value_heads,joint_solver/opening_planner}.
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


def danger_3nn(
    target_xy: Point, target_id: int, planets: list, my_id: int
) -> int:
    """Signed allegiance count over the 3 planets nearest `target_xy`.

    Skips the target itself (matched by `target_id`). Returns an int in
    [-3, +3]: +1 per ally planet, -1 per enemy, 0 per neutral. Used as
    a stepwise spatial-danger feature for snipe / reinforce scoring
    (H17 / TID 699003). The discussion-reported result was that a
    count-based 3-NN hardcoded scoring beat a distance- and ship-weighted
    gradient form 16-0; this is the count-based form.

    `planets` is any iterable yielding objects with `.id`, `.x`, `.y`,
    and `.owner` attrs (e.g. our `lib.intent.Planet` view). Owner
    convention matches the env: -1 = neutral, otherwise = player id.
    """
    tx, ty = target_xy
    others = [p for p in planets if p.id != target_id]
    if not others:
        return 0
    others.sort(key=lambda p: math.hypot(p.x - tx, p.y - ty))
    score = 0
    for p in others[:3]:
        if p.owner == my_id:
            score += 1
        elif p.owner != -1:
            score -= 1
    return score

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


def predict_relative_cached(current_planet, angular_velocity: float,
                            lead_turns: float, *, table=None) -> Point:
    """Lookup-aware wrapper around `predict_relative`.

    When `table` is provided and the planet is in the table's current
    obs snapshot, returns the cached lookup (O(1), no trig). On any
    miss — `table is None`, planet pid not in table, lookup past
    `max_lead` — falls through to the slow-path `predict_relative` call.

    Bit-parity guarantee: the cached path is bit-identical to the
    fallback IFF `planet` is the same instance from
    `world.planets_by_id` that `table.begin_turn(world)` saw. Synthetic
    or predicted planet states (e.g. inside an aim-orbiting fixed-point
    loop where the "planet" position is a hypothetical future tick)
    MUST pass `table=None` to force the slow path.
    """
    if table is None:
        return predict_relative(current_planet, angular_velocity, lead_turns)
    pid_obj = None
    try:
        pid_obj = getattr(current_planet, "id", None)
        if pid_obj is None:
            pid_obj = current_planet[0]
        pid = int(pid_obj)
    except (TypeError, IndexError, KeyError):
        return predict_relative(current_planet, angular_velocity, lead_turns)
    if not table.has(pid):
        return predict_relative(current_planet, angular_velocity, lead_turns)
    try:
        return table.lookup_relative(pid, lead_turns)
    except (IndexError, KeyError):
        return predict_relative(current_planet, angular_velocity, lead_turns)


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


def aim_comet(src_xy, src_radius, target_tuple, target_radius, ships,
              comet_path, comet_path_index):
    """5-iter fixed-point lead for COMET targets — path-indexed, NOT orbital.

    Sibling to `aim_orbiting`. Comets follow pre-computed polynomial paths
    at `cometSpeed=4` board-units/turn, not orbital rotation around the
    sun. Using `predict_relative` for comets mis-aims by 20-40 board
    units within ~7 turns (ep 77087563 / sub 52811320: 40 ships from
    planet 12 → comet 31, fleet OOB).

    Algorithm (mirrors `aim_orbiting`):
    1. Start with target's CURRENT position (path[index]).
    2. Estimate ETA to current target position.
    3. Predict target position at that ETA via path[index + ceil(eta)].
       Returns None if the comet exits the path before arrival.
    4. Re-estimate ETA to the predicted position.
    5. Repeat up to MAX_ITERATIONS. Convergence = |dx|, |dy| < TOL.
    6. If converged, return; else fall back to last-iteration guess.

    Returns (aim_angle, arrival_xy, eta) or None if the comet exits
    before any reachable intercept.

    `comet_path` is a list of `[x, y]` pairs; `comet_path_index` is the
    current path position (advances by 1 per turn in the env, see
    `orbit_wars.py:550`). Caller is responsible for fetching these via
    `lib.world_model.comet_position_at` or `_comet_paths_by_id`.
    """
    path_len = len(comet_path)
    base_idx = int(comet_path_index)
    if base_idx < 0 or base_idx >= path_len:
        return None

    # Start at the comet's current position (path[base_idx]).
    cur_pt = comet_path[base_idx]
    tx, ty = float(cur_pt[0]), float(cur_pt[1])
    last_eta: float | None = None

    for _ in range(MAX_ITERATIONS):
        eta = estimate_eta(src_xy, src_radius, (tx, ty), target_radius, ships)
        if eta is None:
            return None
        # Path-indexed position lookup at the predicted arrival step.
        lead = int(math.ceil(eta))
        future_idx = base_idx + lead
        if future_idx < 0 or future_idx >= path_len:
            # Comet has exited the board before we'd arrive — abort.
            return None
        future_pt = comet_path[future_idx]
        ntx, nty = float(future_pt[0]), float(future_pt[1])
        if (
            last_eta is not None
            and abs(ntx - tx) < CONVERGENCE_XY_TOL
            and abs(nty - ty) < CONVERGENCE_XY_TOL
        ):
            angle = math.atan2(nty - src_xy[1], ntx - src_xy[0])
            return angle, (ntx, nty), eta
        tx, ty = ntx, nty
        last_eta = eta

    # Non-convergence: return the last iteration's guess. The trajectory
    # filter / cost-parity filter will catch downstream misfires.
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

# Bug #12 fix (2026-05-18): width of the in-flight-enemy summation
# window used when computing combined threat against a single planet.
# A staggered multi-wave attack (e.g. f1 at eta=2 + f2 at eta=4)
# should be accounted for as one coordinated threat; pre-fix the
# window was `enemy_eta + 1` of the EARLIEST inbound, which silently
# excluded later waves and zeroed the shortfall. Anchored on the
# asdf-game (76947663) step 37 trace. Promoted to lib so both the
# proposer (`agents/baseline/proposer.py`) and the in-rollout
# defensive policy (`lib/opp_model.me_defensive_action`) import it
# from one location. The principled v2 of this fix is a full
# timeline simulation to find the max shortfall over time; this
# constant is the cheap version.
WAVE_LOOKAHEAD = 12


def _position_at(planet, omega: float, lead_turns: int) -> tuple[float, float]:
    """Return predicted `(x, y)` of `planet` `lead_turns` from now.

    Current position for static planets, when `omega == 0`, or when
    `lead_turns <= 0`. Otherwise routes through
    `lib.orbit.predict_relative` after the `is_orbiting` gate.

    Used by `time_to_enemy_threat` and sibling orbital-safety call sites
    to keep the predict-position-at-arrival pattern in one tested place.
    """
    if lead_turns <= 0 or omega == 0.0:
        return float(planet.x), float(planet.y)
    tup = [planet.id, planet.owner, planet.x, planet.y,
           planet.radius, planet.ships, planet.production]
    if not is_orbiting(tup):
        return float(planet.x), float(planet.y)
    return predict_relative(tup, omega, lead_turns)


def fleet_target_planet(fleet, planets, omega: float = 0.0,
                        max_horizon: int = DEFAULT_HORIZON):
    """Trace `fleet` along its angle, find first planet it'd hit.

    Returns `(target_planet, eta_turns)` or `(None, None)` if no planet
    intersects the fleet's trajectory within `max_horizon` steps.

    For STATIC (non-orbiting) planets: straight-line ray-cast (cheap;
    closed-form). For ORBITING planets: per-tick collision check using
    `lib.orbit.predict_relative` to predict the planet's position at
    each tick, then test fleet-vs-planet point-in-circle.

    The `omega` argument is the environment's angular velocity from the
    obs. When `omega == 0.0`, behaviour matches the previous static-only
    ray-cast (orbiting check is short-circuited since rotation is zero).

    Used to build the arrival ledger from in-flight fleets — the env
    doesn't expose a fleet's intended target, only its angle.

    Bug fix 2026-05-18 (#11): pre-fix the static ray-cast missed
    orbiting targets that rotate INTO the fleet's path mid-flight.
    Asdf game (76947663) step 37: 65-ship fleet aimed at orbiting P15
    returned target=None until P15 had rotated into the straight line
    at step 40 — by then too late to defend.
    """
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    spd = fleet_speed(fleet.ships)
    if spd <= 0:
        return None, None

    # Partition planets: static (closed-form fast path) vs orbiting
    # (per-tick scan). The partition is cheap; typical boards have
    # ~12-20 planets total with 5-8 orbiting.
    static_planets = []
    orbiting_planets = []
    for p in planets:
        # Build minimal tuple for is_orbiting (only x, y, radius used)
        p_tuple = (int(p.id), int(p.owner), float(p.x), float(p.y),
                   float(p.radius), 0, 0)
        if omega != 0.0 and is_orbiting(p_tuple):
            orbiting_planets.append((p, p_tuple))
        else:
            static_planets.append(p)

    best_planet = None
    best_turns = None

    # Fast path: static planets — straight-line ray-cast (unchanged
    # math from pre-fix behaviour).
    for p in static_planets:
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

    # Orbital path: per-tick collision scan.
    if orbiting_planets:
        # Discretize: check at integer ticks up to max_horizon. We use
        # the int-ceil semantics that the ledger eventually buckets to,
        # so checking at integer ticks is sufficient precision.
        for t in range(1, int(max_horizon) + 1):
            # Pruning: if we already have a static hit at eta T, no
            # orbital hit beyond T can win.
            if best_turns is not None and t > best_turns:
                break
            fx = fleet.x + dir_x * spd * t
            fy = fleet.y + dir_y * spd * t
            for p, p_tuple in orbiting_planets:
                px, py = predict_relative(p_tuple, omega, t)
                # Point-in-circle: fleet position within planet radius
                # at tick t. Matches the ledger's step-bucket precision.
                if math.hypot(fx - px, fy - py) <= float(p.radius):
                    if best_turns is None or t < best_turns:
                        best_turns = t
                        best_planet = p
                    break  # found a hit at this tick; advance to next tick

    if best_planet is None:
        return None, None
    return best_planet, int(math.ceil(best_turns))


def build_arrival_ledger(fleets, planets, omega: float = 0.0,
                         horizon: int = DEFAULT_HORIZON):
    """{planet_id: [(eta, owner, ships), ...]} for in-flight fleets.

    Fleets that won't hit any planet within `horizon` are dropped (they
    will exit the board or die in sun/non-target collision — out of
    scope for the timeline).

    `omega` is the env's angular velocity; passed through to
    `fleet_target_planet` for correct orbiting-target attribution.
    Defaults to 0 for backward compatibility (callers that don't pass
    it get the previous static-only behaviour).
    """
    ledger: dict[int, list[tuple[int, int, int]]] = {p.id: [] for p in planets}
    for fleet in fleets:
        target, eta = fleet_target_planet(fleet, planets, omega, horizon)
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


def predict_garrison_at(planet, eta: int,
                        arrivals: list[tuple[int, int, int]],
                        ) -> tuple[int, float]:
    """Single-tick combat prediction: `(owner, garrison)` at exactly `eta`
    ticks from now. O(eta) walk, O(arrivals) total work.

    Cheaper alternative to `simulate_planet_timeline` when callers only
    need state at one specific tick (e.g. a candidate's arrival). Same
    combat rules (production tick → resolve_arrivals per step), just
    doesn't build the full dict timeline.

    `arrivals` matches the per-planet entry in `build_arrival_ledger`:
    list of `(eta_arrival, owner, ships)`.

    Origin: trajectory-first chooser (2026-05-17). The chooser scores
    each candidate by predicting the arrival outcome at exactly the
    candidate's eta; building a 40-step timeline per planet per call
    was the dominant cost of the K-step rollout we're replacing.
    """
    eta = max(0, int(math.ceil(eta)))
    if eta == 0:
        return planet.owner, max(0.0, float(planet.ships))

    # Bucket arrivals by tick.
    by_turn: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for arrival_eta, arrival_owner, arrival_ships in arrivals:
        if arrival_ships <= 0:
            continue
        bucket = max(1, int(math.ceil(arrival_eta)))
        if bucket > eta:
            continue
        by_turn[bucket].append((arrival_owner, int(arrival_ships)))

    owner = planet.owner
    garrison = float(planet.ships)
    for t in range(1, eta + 1):
        if owner != -1:
            garrison += planet.production
        group = by_turn.get(t, [])
        if group:
            owner, garrison = resolve_arrivals(owner, garrison, group)
    return owner, max(0.0, garrison)


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
        directly from the raw obs because `World` doesn't materialise them.

        Threads the env's `angular_velocity` through to the ledger build
        so inbound fleets aimed at orbiting planets are correctly
        attributed (bug #11 fix, 2026-05-18).
        """
        raw = world.obs_raw
        if isinstance(raw, dict):
            fleets_raw = raw.get("fleets", [])
            omega = float(raw.get("angular_velocity", 0.0) or 0.0)
        else:
            fleets_raw = getattr(raw, "fleets", [])
            omega = float(getattr(raw, "angular_velocity", 0.0) or 0.0)

        from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet  # local import — keeps lib/ env-free
        fleets = [Fleet(*f) for f in fleets_raw]
        planets = list(world.planets_by_id.values())
        ledger = build_arrival_ledger(fleets, planets, omega, horizon)
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

    def incoming_enemy_eta_after(self, planet_id: int, my_id: int,
                                  after: int) -> int | None:
        """Min ETA among in-flight enemy fleets arriving STRICTLY AFTER
        `after`. None if no qualifying fleet exists.

        Used by `time_to_enemy_threat` with `arrival_eta > 0`: pre-arrival
        and same-step inbound fleets are resolved by `owner_at` /
        `ships_at` combat at our arrival, so they must NOT double-count
        as future threats; AND, the earliest inbound fleet may itself be
        pre-arrival while a LATER wave is a real threat — `incoming_enemy_eta`
        would silently drop the later wave because it only returns the
        minimum. This method surfaces the earliest post-`after` wave.
        Origin: B6 fix (2026-05-22 audit of f1774a7 orbital-safety patch).
        """
        arrivals = self.ledger.get(planet_id)
        if not arrivals:
            return None
        candidates = [
            eta for (eta, owner, ships) in arrivals
            if owner != my_id and ships > 0 and eta > after
        ]
        if not candidates:
            return None
        return min(candidates)

    def time_to_enemy_threat(self, planet_id: int, my_id: int, world,
                              arrival_eta: int = 0) -> int | None:
        """Earliest turn at which an enemy could have a fleet at
        `planet_id`. Considers BOTH (a) in-flight enemy fleets
        currently inbound, and (b) potential launches from every
        currently-stationary enemy-owned planet at its present
        garrison.

        Returns `None` if no enemy can plausibly threaten the planet
        (caller should treat as "saturate at game horizon").

        H22 helper for Hold-Aware Value scoring. See plan file
        2026-05-14 HAV section.

        `arrival_eta` (PI 2026-05-21 bug fix, completed 2026-05-22) —
        when > 0, the target and enemy planet positions are predicted
        at that future turn via `predict_relative`. This fixes a silent
        scoring bug where an orbiting target that rotates INTO enemy
        territory by our arrival was scored as safe (long expected_hold)
        because the threat ETA was computed from the CURRENT target
        position. Default 0 preserves the original "current position"
        semantics for source-safety callers (drain checks etc).

        Coverage notes (B5/B6/B7, completed in this audit pass):
        - B5: in-flight fleets that arrive at-or-before our arrival are
          resolved by combat at our arrival; only fleets arriving
          STRICTLY AFTER `arrival_eta` count as future threats.
        - B6: `incoming_enemy_eta` returns only the earliest inbound;
          when that earliest is pre-arrival, a later wave can be the
          real threat. Use `incoming_enemy_eta_after` to find it.
        - B7: enemy fleet aims at target-at-our-arrival, but target
          keeps rotating during enemy travel. Iterate a 5-step
          fixed-point on `enemy_eta_travel` for orbiting targets;
          fall through to the seed estimate on non-convergence.
        """
        target = world.planets_by_id.get(planet_id)
        if target is None:
            return None

        omega = float(getattr(world, "omega", 0.0))
        target_is_orbital = (
            arrival_eta > 0 and omega != 0.0
            and is_orbiting([target.id, target.owner, target.x, target.y,
                             target.radius, target.ships, target.production])
        )

        # Target position at our arrival.
        tx, ty = _position_at(target, omega, arrival_eta)

        best: int | None = None

        # (a) in-flight enemy fleets — B5 + B6 fix. Filter strictly to
        # fleets arriving AFTER our arrival; the earliest qualifying
        # fleet (not just the earliest overall) becomes the in-flight
        # threat ETA.
        if arrival_eta > 0:
            inbound = self.incoming_enemy_eta_after(planet_id, my_id,
                                                     arrival_eta)
        else:
            inbound = self.incoming_enemy_eta(planet_id, my_id)
        if inbound is not None:
            best = inbound

        # (b) potential launches from each enemy planet. When
        # arrival_eta > 0, predict the enemy's position at our arrival
        # too (assumes enemy launches immediately upon our capture).
        for p in world.planets_by_id.values():
            if p.id == planet_id:
                continue
            if p.owner == my_id or p.owner == -1:
                continue
            if p.ships <= 0:
                continue
            px, py = _position_at(p, omega, arrival_eta)
            dx = tx - px
            dy = ty - py
            dist = (dx * dx + dy * dy) ** 0.5
            v = fleet_speed(int(p.ships))
            if v <= 0:
                continue
            eta_travel = int(-(-dist // v))  # math.ceil without import

            # B7 — 5-iteration fixed-point on `eta_travel` for orbiting
            # targets. The enemy fleet aims at target-at-our-arrival, but
            # during its travel the target keeps rotating; the actual
            # rendezvous point shifts. Iterate target_pos_at(arrival +
            # eta_travel) → recompute dist → recompute eta_travel until
            # |Δ| ≤ 1 (mirror of lib/aim.py:aim_orbiting). The enemy
            # planet's position at arrival_eta stays fixed (the assumed
            # launch moment).
            if target_is_orbital and eta_travel > 0:
                for _ in range(5):
                    tx_k, ty_k = _position_at(
                        target, omega, arrival_eta + eta_travel,
                    )
                    dist_k = ((tx_k - px) ** 2 + (ty_k - py) ** 2) ** 0.5
                    new_eta = int(-(-dist_k // v))
                    if abs(new_eta - eta_travel) <= 1:
                        eta_travel = new_eta
                        break
                    eta_travel = new_eta

            threat_arrival = arrival_eta + eta_travel
            if best is None or threat_arrival < best:
                best = threat_arrival

        return best

    def predicted_threat_force(self, planet_id: int, my_id: int,
                                world, lookahead: int) -> int:
        """Force of the WORST single threat vector at `planet_id`
        within `lookahead` steps:
          - all in-flight ledger arrivals with 0 < eta <= lookahead
            (these are committed; sum is correct — they ARE all coming),
          - PLUS the MAX single stationary enemy planet's garrison
            among opps whose travel ETA from current position <= lookahead.
        Mental model: at most one opp can mount a coordinated wave in
        the relevant window; sizing reserve against the SUM of every
        opp's garrison (initial design 2026-05-27 AM) treated all opps
        as a single coalition and bricked launches (0/32 vs HEAD anchor
        smoke). MAX-single + committed-in-flight matches the legacy
        `threat_force` mental model in `_source_survives_launch_legacy`
        (sums in-flight ledger only).
        """
        committed = 0
        for (eta, owner, sh) in self.ledger.get(planet_id, []):
            if owner != my_id and 0 < eta <= lookahead and sh > 0:
                committed += int(sh)
        src_planet = world.planets_by_id.get(planet_id)
        if src_planet is None:
            return committed
        omega = float(getattr(world, "omega", 0.0))
        sx, sy = _position_at(src_planet, omega, 0)
        worst_potential = 0
        for p in world.planets_by_id.values():
            if p.id == planet_id or p.owner == my_id or p.owner == -1:
                continue
            if p.ships <= 0:
                continue
            px, py = _position_at(p, omega, 0)
            dist = ((sx - px) ** 2 + (sy - py) ** 2) ** 0.5
            v = fleet_speed(int(p.ships))
            if v <= 0:
                continue
            eta_travel = int(math.ceil(dist / v))
            if eta_travel <= lookahead and int(p.ships) > worst_potential:
                worst_potential = int(p.ships)
        return committed + worst_potential


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


def comet_position_at(planet_id: int, world, lead_turns: int) -> tuple[float, float] | None:
    """Position of comet `planet_id` at `lead_turns` from now.

    Returns `(x, y)` from the comet's pre-computed path at index
    `path_index + lead_turns`, or `None` if the comet has exited the
    board by then (index past the end of the path) or if `planet_id`
    isn't a comet.

    Comets travel along polynomial paths at `cometSpeed=4` board
    units/turn (env: `orbit_wars.py::generate_comet_paths`), NOT around
    the central sun like orbital planets. So `lib.orbit.predict_relative`
    is wrong for comets — use this instead.
    """
    paths_by_id = _comet_paths_by_id(world)
    entry = paths_by_id.get(int(planet_id))
    if entry is None:
        return None
    path, path_index = entry
    idx = int(path_index) + int(lead_turns)
    if idx < 0 or idx >= len(path):
        return None
    point = path[idx]
    return float(point[0]), float(point[1])

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

# === inlined: lib/kinematic_table.py ===


from dataclasses import dataclass
from typing import Any, Iterable, Optional

import math



# Sentinel reused from `lib/trajectory.py:127`; we DO NOT import to avoid
# a circular dependency with the call site we'll later modify.
OFF_BOARD: tuple[float, float] = (-1e6, -1e6)

# Default lookup window. predict_fleet_fate uses `max_steps=200` as the
# ray-cast horizon; callers may also pass wait_N (fire-offset) up to ~50.
# 500 gives generous headroom (covers wait_N up to ~300 + max_steps=200)
# at ~250 KB total memory cost — negligible. Phase γ relies on this
# default being large enough that the table covers any predict_fleet_fate
# call without falling through to the slow path.
DEFAULT_MAX_LEAD: int = 500


# ---------------------------------------------------------------------------
# Class — per-instance container; tests instantiate directly.
# ---------------------------------------------------------------------------


@dataclass
class _PlanetEntry:
    """Per-planet position cache for one turn."""

    pid: int
    kind: str  # "static" | "orbital" | "comet"
    # For static: positions == None, static_pos holds the constant.
    static_pos: Optional[tuple[float, float]] = None
    # For orbital + comet: positions[t] is (x, y) at `lead = t` from
    # current obs; len(positions) == max_lead + 1.
    positions: Optional[list[tuple[float, float]]] = None
    # For comets only: the raw path + path_index from obs["comets"],
    # surfaced via `comet_paths_view` for callers replacing
    # `lib.world_model._comet_paths_by_id`.
    comet_path: Optional[list] = None
    comet_path_index: Optional[int] = None


class KinematicTable:
    """Per-instance kinematic position cache.

    One instance is held as a module-level singleton; tests can create
    isolated instances for parity assertions. Lifecycle:

        table.begin_turn(world)           # rebuild from current obs
        table.lookup_relative(pid, lead)  # (x, y) at `lead` ticks ahead
        table.window(pids, off, n)        # dict of position lists

    `begin_turn` is idempotent within a turn: if the (step, omega,
    planets) fingerprint matches the last build, no rebuild happens.
    """

    def __init__(self, max_lead: int = DEFAULT_MAX_LEAD) -> None:
        self._entries: dict[int, _PlanetEntry] = {}
        self._fingerprint: Any = None
        self._omega: float = 0.0
        self._step: int = -1
        self._max_lead: int = int(max_lead)

    # ---- lifecycle ----

    def reset(self) -> None:
        """Drop all state. Tests use this; production callers shouldn't."""
        self._entries = {}
        self._fingerprint = None
        self._omega = 0.0
        self._step = -1

    def begin_turn(self, world, *, max_lead: Optional[int] = None) -> bool:
        """Rebuild the table from `world` if the turn fingerprint changed.

        Returns True iff a rebuild fired (caller can log this for
        observability). Fingerprint:

            (step, omega, n_planets, sorted-tuple of (pid, id(planet_obj)))

        The `id(planet_obj)` term cheaply detects the per-turn obs
        rebuild — `World.from_obs` constructs fresh `Planet` instances
        every turn, so identities never repeat. On game boundary
        (`step` drops to 0 with different planet ids), fingerprint
        differs and we wipe.
        """
        if max_lead is not None and int(max_lead) != self._max_lead:
            # max_lead change forces rebuild even if obs is unchanged.
            self._max_lead = int(max_lead)
            self._fingerprint = None

        new_fp = self._build_fingerprint(world)
        if self._fingerprint == new_fp:
            return False
        self._rebuild(world)
        self._fingerprint = new_fp
        return True

    @staticmethod
    def _build_fingerprint(world) -> tuple:
        planets = world.planets_by_id
        pid_ids = tuple(sorted((int(pid), id(p)) for pid, p in planets.items()))
        return (int(world.step), float(world.omega), len(planets), pid_ids)

    def _rebuild(self, world) -> None:
        """Materialise per-planet position lists from `world`.

        For orbital planets, calls `predict_relative` per lead-tick — the
        SAME function `predict_fleet_fate`'s inner loop calls, with the
        SAME planet tuple shape, so bit-parity is guaranteed by
        construction. Static planets store a single constant; comets
        consult the obs path array with the off-board sentinel.
        """
        self._entries = {}
        self._omega = float(world.omega)
        self._step = int(world.step)
        comet_paths = _extract_comet_paths(world)
        max_lead = self._max_lead

        for pid, p in world.planets_by_id.items():
            pid_i = int(pid)
            if pid_i in comet_paths:
                path, path_index = comet_paths[pid_i]
                positions: list[tuple[float, float]] = []
                for t in range(max_lead + 1):
                    path_t = int(path_index) + t
                    if 0 <= path_t < len(path):
                        pt = path[path_t]
                        positions.append((float(pt[0]), float(pt[1])))
                    else:
                        positions.append(OFF_BOARD)
                self._entries[pid_i] = _PlanetEntry(
                    pid=pid_i, kind="comet",
                    positions=positions,
                    comet_path=path,
                    comet_path_index=int(path_index),
                )
                continue

            p_tuple = [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
            if is_orbiting(p_tuple) and self._omega != 0.0:
                # Orbital. Per-lead call to predict_relative — identical
                # arithmetic path to the inline call site (scalar
                # math.cos/sin under the hood). Bit-parity by
                # construction.
                positions = [
                    predict_relative(p_tuple, self._omega, t)
                    for t in range(max_lead + 1)
                ]
                self._entries[pid_i] = _PlanetEntry(
                    pid=pid_i, kind="orbital",
                    positions=positions,
                )
            else:
                # Static (outer planet OR omega == 0). Single constant.
                self._entries[pid_i] = _PlanetEntry(
                    pid=pid_i, kind="static",
                    static_pos=(float(p.x), float(p.y)),
                )

    # ---- queries ----

    def has(self, pid: int) -> bool:
        return int(pid) in self._entries

    @property
    def max_lead(self) -> int:
        """Maximum `lead` value the table can answer without falling
        through. Use this to gate calls that need a large window."""
        return self._max_lead

    @property
    def step(self) -> int:
        """The absolute env step the table was last built for."""
        return self._step

    @property
    def n_planets(self) -> int:
        return len(self._entries)

    def covers(self, pids, max_needed_lead: int) -> bool:
        """Return True iff every pid is in the table AND the table's
        max_lead is >= max_needed_lead. Cheap pre-flight for the Phase γ
        predict_fleet_fate swap — on False, caller falls through to the
        slow inline build."""
        if max_needed_lead > self._max_lead:
            return False
        entries = self._entries
        for pid in pids:
            if int(pid) not in entries:
                return False
        return True

    def kind(self, pid: int) -> Optional[str]:
        entry = self._entries.get(int(pid))
        return entry.kind if entry is not None else None

    def lookup_relative(self, pid: int, lead: int) -> tuple[float, float]:
        """Return (x, y) at `lead` ticks after current obs.

        Bit-identical to
        `predict_relative(world.planets_by_id[pid], world.omega, lead)`
        for orbital planets, and to `(p.x, p.y)` for static planets.
        For comets, returns `path[path_index + lead]` if in range, else
        `OFF_BOARD`. Raises KeyError if `pid` is not in the table.
        """
        entry = self._entries.get(int(pid))
        if entry is None:
            raise KeyError(f"kinematic_table: pid={pid} not in current obs")
        if entry.kind == "static":
            return entry.static_pos  # type: ignore[return-value]
        positions = entry.positions
        if positions is None:
            raise RuntimeError(f"kinematic_table: pid={pid} has no positions cache")
        n = len(positions)
        i = int(lead)
        if 0 <= i < n:
            return positions[i]
        # Beyond the precomputed window: for orbital, this is a usage
        # bug (caller asked past max_lead). For comet, this is a real
        # case — the path may extend beyond max_lead. We fall through
        # to a slow-path computation that matches the inline behaviour.
        if entry.kind == "comet":
            path = entry.comet_path
            path_index = entry.comet_path_index
            assert path is not None and path_index is not None
            path_t = int(path_index) + i
            if 0 <= path_t < len(path):
                pt = path[path_t]
                return (float(pt[0]), float(pt[1]))
            return OFF_BOARD
        # Orbital out-of-range: compute on demand (bit-parity preserved
        # because we use the same predict_relative call).
        # Caller is asking past max_lead — re-derive from omega + the
        # stored first-position. We don't store the source `p_tuple`,
        # so we reconstruct from positions[0] which is the obs-step
        # position. NOTE: positions[0] == predict_relative(p_tuple, omega, 0)
        # which for static omega=0 case returns (p.x, p.y) exactly, and
        # for orbital case may differ from the raw obs (p.x, p.y) by ULPs
        # because of the atan2(cos(.), sin(.)) round-trip. To preserve
        # bit-parity we instead raise — out-of-range orbital lookups are
        # a contract violation and we want them surfaced, not silently
        # answered with possibly-drifted floats.
        raise IndexError(
            f"kinematic_table: lead={i} past max_lead={n - 1} for orbital "
            f"pid={pid}; increase max_lead at begin_turn"
        )

    def window(
        self,
        pids: Iterable[int],
        start_offset: int,
        length: int,
    ) -> dict[int, list[tuple[float, float]]]:
        """Return {pid: [position at lead=start_offset+t for t in range(length)]}.

        Mirrors the `planet_positions` dict built inline at
        `lib/trajectory.py:137-159`. Use the SAME `length = max_steps + 1`
        the inline code uses; `start_offset = wait_N` for the predict-
        fleet-fate use case.
        """
        out: dict[int, list[tuple[float, float]]] = {}
        for pid in pids:
            pid_i = int(pid)
            entry = self._entries.get(pid_i)
            if entry is None:
                # Match the inline behaviour: missing planet → skip
                # (callers iterate over world.planets_by_id, so this
                # shouldn't fire in practice).
                continue
            if entry.kind == "static":
                pos = entry.static_pos  # type: ignore[assignment]
                out[pid_i] = [pos] * int(length)
                continue
            assert entry.positions is not None
            positions = entry.positions
            n = len(positions)
            row: list[tuple[float, float]] = []
            for t in range(int(length)):
                k = int(start_offset) + t
                if 0 <= k < n:
                    row.append(positions[k])
                elif entry.kind == "comet":
                    # Slow-path lookup past max_lead.
                    path = entry.comet_path
                    path_index = entry.comet_path_index
                    assert path is not None and path_index is not None
                    path_t = int(path_index) + k
                    if 0 <= path_t < len(path):
                        pt = path[path_t]
                        row.append((float(pt[0]), float(pt[1])))
                    else:
                        row.append(OFF_BOARD)
                else:
                    # Orbital past max_lead — see note in lookup_relative.
                    raise IndexError(
                        f"kinematic_table: start_offset+t={k} past "
                        f"max_lead={n - 1} for orbital pid={pid_i}"
                    )
            out[pid_i] = row
        return out

    def comet_paths_view(self) -> dict[int, tuple[list, int]]:
        """{pid: (path, path_index)} for every comet in the current obs.

        Schema-identical to `lib/world_model._comet_paths_by_id(world)`;
        the integration in Phase γ swaps that function's body to read
        from here when the table is populated.
        """
        out: dict[int, tuple[list, int]] = {}
        for pid, entry in self._entries.items():
            if entry.kind == "comet":
                assert entry.comet_path is not None and entry.comet_path_index is not None
                out[pid] = (entry.comet_path, entry.comet_path_index)
        return out

    # ---- diagnostics ----

    def stats(self) -> dict:
        kinds = {"static": 0, "orbital": 0, "comet": 0}
        for e in self._entries.values():
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        return {
            "n_planets": len(self._entries),
            "kinds": kinds,
            "step": self._step,
            "omega": self._omega,
            "max_lead": self._max_lead,
        }


def _extract_comet_paths(world) -> dict[int, tuple[list, int]]:
    """Inline copy of `lib.world_model._comet_paths_by_id`'s body.

    Duplicated here to avoid a circular import at module-load time;
    Phase γ reverses this by having `_comet_paths_by_id` consult the
    table when populated.
    """
    raw = getattr(world, "obs_raw", None)
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


# ---------------------------------------------------------------------------
# Module-level singleton + thin function wrappers.
# ---------------------------------------------------------------------------

_DEFAULT = KinematicTable()


def clear() -> None:
    """Reset the module-level singleton (tests + the legacy entry point)."""
    _DEFAULT.reset()


def begin_turn(world, *, max_lead: Optional[int] = None) -> bool:
    return _DEFAULT.begin_turn(world, max_lead=max_lead)


def lookup_relative(pid: int, lead: int) -> tuple[float, float]:
    return _DEFAULT.lookup_relative(pid, lead)


def window(
    pids: Iterable[int],
    start_offset: int,
    length: int,
) -> dict[int, list[tuple[float, float]]]:
    return _DEFAULT.window(pids, start_offset, length)


def comet_paths_view() -> dict[int, tuple[list, int]]:
    return _DEFAULT.comet_paths_view()


def get_default() -> KinematicTable:
    """Accessor for the module-level singleton."""
    return _DEFAULT

# === inlined: lib/trajectory.py ===


import math
import os
from dataclasses import dataclass

fleet_speed = speed


def _kinematic_table_enabled() -> bool:
    """Phase γ opt-in: when `KINEMATIC_TABLE_ENABLED=1` AND the
    module-level singleton has been primed via begin_turn(world) AND
    its window is large enough, predict_fleet_fate's position build
    uses the cached lookup instead of re-computing predict_relative
    per (planet, step).
    """
    return os.environ.get("KINEMATIC_TABLE_ENABLED", "").strip().lower() in (
        "1", "true", "on", "yes",
    )

# Max steps we simulate before giving up. A 1-ship fleet at speed 1.0
# can cross the 141.4-unit board diagonal in 142 steps; 200 covers
# every realistic case with comfortable margin.
DEFAULT_MAX_STEPS = 200

# Safety margin around the sun (units). The env's sun-check uses
# point-to-segment distance < SUN_RADIUS strict; we MUST match exactly
# or we false-reject trajectories that pass within 10.0-10.5 units of
# centre (the engine accepts them).
#
# Origin: 2026-05-17 Direction A A/B. v4 with this cushion at 0.5 cost
# ~14pp of winrate vs v15 (n=64: 31/64 filter-OFF → 22/64 filter-ON,
# ~10pp difference; the 14pp is including some non-sun rejections).
# The 0.5 cushion was filed in 2026-05-11 as a "float drift cushion"
# but in fact created systematic false-rejections.
SUN_SAFETY = 0.0


@dataclass(frozen=True)
class FleetFate:
    outcome: str               # "target" | "planet" | "sun" | "oob" | "timeout"
    hit_planet_id: int | None  # set when outcome in {"target", "planet"}
    step: int                  # 1-based step at which the event occurred


def predict_fleet_fate(
    src, target, aim_angle: float, ships: int,
    world, max_steps: int = DEFAULT_MAX_STEPS,
    wait_N: int = 0,
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

    `wait_N>0`: the fleet is scheduled to launch wait_N ticks from now.
    Source position, planet positions, and the spawn point are all
    advanced by wait_N orbital ticks before the ray-cast begins. Use this
    when validating wait-then-fire candidates whose fire-time geometry
    differs from the current world snapshot.

    O(max_steps * planets) per call. On a 24-planet mid-game board with
    max_steps=200 that's ~4800 swept_pair_hit calls = ~1-2 ms.
    """
    omega = world.omega

    # Source position at fire time (t + wait_N).
    src_tuple = [src.id, src.owner, src.x, src.y, src.radius,
                 src.ships, src.production]
    if wait_N > 0 and is_orbiting(src_tuple) and omega != 0.0:
        src_x_fire, src_y_fire = predict_relative(src_tuple, omega, wait_N)
    else:
        src_x_fire, src_y_fire = src.x, src.y

    # Spawn position (env: src.center + (radius + 0.1) * direction).
    cos_a = math.cos(aim_angle)
    sin_a = math.sin(aim_angle)
    spawn_x = src_x_fire + cos_a * (src.radius + 0.1)
    spawn_y = src_y_fire + sin_a * (src.radius + 0.1)
    speed_val = fleet_speed(ships)
    if speed_val <= 0:
        # Shouldn't happen (fleet_speed is monotonically >= 1.0 for ships >= 1).
        return FleetFate("oob", None, 0)

    # Pre-compute per-planet positions at every step from t+wait_N onward.
    #
    # COMET HANDLING: comets follow discrete paths from `obs["comets"]`,
    # NOT orbital paths. Predicting them with `predict_relative` is
    # wrong — the prior bug produced 47 OOB events in seed 42 self-play
    # (all post-step-50 when comets enter): fleets aimed at "comet at
    # predicted orbital position" missed the real comet and flew off
    # the board. Look up the comet's actual path and use it; for steps
    # past the path's end, mark the comet as "gone" with sentinel
    # positions far outside the board so swept_pair_hit can't match.
    OFF_BOARD = (-1e6, -1e6)  # sentinel for "comet has left the board"
    # Env semantics (verified against
    # kaggle_environments/envs/orbit_wars/orbit_wars.py lines 480-595):
    # at env step T+1's fleet-movement check, the planet's old_pos is
    # the position from obs T (planet[2], planet[3]) and new_pos is the
    # advanced position. positions[0] is therefore the obs-T position
    # (path[path_index] for comets; predict_relative(.., 0) for orbital);
    # positions[1] is the obs-T+1 position. With wait_N>0 the fleet
    # appears at env step T+1+wait_N and positions[0] = obs-T+wait_N.
    #
    # Phase γ — when KINEMATIC_TABLE_ENABLED=1 and the table is primed
    # for this world AND covers our (wait_N + max_steps) window, the
    # `planet_positions` dict comes from a one-call lookup into the
    # table's per-turn cache. On any miss — env-var off, table not
    # primed, max_lead too small — fall through to the inline build.
    planet_positions = _table_window_or_none(world, wait_N, max_steps + 1)
    if planet_positions is None:
        comet_paths = _comet_paths_by_id(world) if world.comet_ids else {}
        planet_positions = {}
        for pid, p in world.planets_by_id.items():
            if int(pid) in comet_paths:
                # Comet: use its discrete path.
                path, path_index = comet_paths[int(pid)]
                positions: list[tuple[float, float]] = []
                for t in range(max_steps + 1):
                    path_t = int(path_index) + int(wait_N) + t
                    if 0 <= path_t < len(path):
                        pt = path[path_t]
                        positions.append((float(pt[0]), float(pt[1])))
                    else:
                        positions.append(OFF_BOARD)
                planet_positions[pid] = positions
                continue
            p_tuple = [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
            if is_orbiting(p_tuple) and omega != 0.0:
                planet_positions[pid] = [
                    predict_relative(p_tuple, omega, wait_N + t)
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
            # NB: the env DOES check fleet-vs-source at step 0 (see
            # orbit_wars.py L588-597 — no exclusion of from_id). For
            # STATIC sources the geometry handles it: spawn is at
            # `src.center + (radius + 0.1) * direction`, the fleet
            # moves AWAY, swept_pair_hit never matches.
            # For MOVING sources (comets, fast-orbiting planets), the
            # source can catch the fleet within 1 step — that's a real
            # collision the env applies. The earlier `if pid == src_id
            # and step == 0: continue` skip falsely declared "target
            # reached" for drain trajectories whose comet-source
            # caught up and absorbed the fleet (root cause of stranded
            # ships on captured comets).
            p_old = positions[step]
            p_new = positions[step + 1]
            # Comet expiry guard: if EITHER endpoint is the off-board
            # sentinel, the comet has expired during this step — skip
            # the collision check entirely. Without this guard,
            # swept_pair_hit would treat the comet as moving along the
            # huge sentinel-going segment, falsely matching any fleet
            # trajectory (the env, however, removes expired comets from
            # collision resolution — see orbit_wars.py L558-561). This
            # was the cause of the residual seed-13 OOB: fleet aimed at
            # "comet 38 at fleet_step 20" — but the comet's path ended
            # at index 33 (path[14] + 20 == 34), so positions[20] is
            # OFF_BOARD and the swept check produced a phantom hit;
            # the env had no comet there, so the fleet sailed past and
            # exited the board.
            if (p_old[0] < 0 or p_old[0] > BOARD_SIZE
                    or p_old[1] < 0 or p_old[1] > BOARD_SIZE
                    or p_new[0] < 0 or p_new[0] > BOARD_SIZE
                    or p_new[1] < 0 or p_new[1] > BOARD_SIZE):
                continue
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


def _table_window_or_none(world, wait_N: int, length: int):
    """Phase γ: pull positions from the kinematic table when enabled
    and primed for this world. Returns the planet_positions dict the
    inline build would have produced, or None to signal "fall through
    to inline build".

    Bit-parity contract: the table is rebuilt every turn from
    `world.planets_by_id` using the SAME `predict_relative` calls the
    inline build makes (orbital), the SAME `(p.x, p.y)` constants
    (static), and the SAME path lookups (comets).
    """
    if not _kinematic_table_enabled():
        return None
    # Lazy import keeps default-path module-load time unchanged.
    table = get_default()
    pids = list(world.planets_by_id.keys())
    if not pids:
        return None
    needed_lead = int(wait_N) + int(length) - 1
    if not table.covers(pids, needed_lead):
        return None
    # Sanity: the table must be primed for THIS turn's obs. Trust
    # begin_turn fingerprinting to keep this fresh.
    return table.window(pids, start_offset=int(wait_N), length=int(length))

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
# Fleet-size over-commit (H19 / TID 697397). Per Gemini's Day-2 writeup,
# sending 10 % more ships than the minimum capture amount gives a
# log-curve speed boost (fleet_speed = 1.0 + (max_speed-1.0) ·
# (log(ships)/log(1000))^1.5) — earlier arrival, more reliable captures,
# small extra defender on arrival. Off by default (= 1.0 identity).
# 2026-05-13 falsified on v7 (3-variant Rule-21 sweep) — the K=10 rollout
# already sizes optimally; pre-rollout inflation drains source garrisons
# without producing the lift Gemini's heuristic-only setup observed.
# Kept as a flag for future use on simpler agents.
FLEET_OVERCOMMIT = 1.0


# ---------------------------------------------------------------------------
# Pre-reinforce window (H21 / TID 698478). The discussion thread describes
# the "you take it, they take it back at no cost" pattern: we capture a
# planet at t=eta with the minimum garrison, then enemy fleet arrives at
# t=eta+1 and recaptures because our post-capture garrison is ~0. The fix
# is to query the WorldModel arrival ledger for enemy fleets landing in
# the window (eta, eta+window] and add buffer to our intent so the
# post-capture garrison survives them. Off by default (= 0). A/B
# candidate values: 2 (one production tick + one arrival), 3, 5.
PRE_REINFORCE_WINDOW = 0


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
        if target.owner == world.my_id:
            # Reinforce — pass through unchanged (boost would just crowd
            # the home combat resolution).
            out.append(intent)
            continue
        if target.owner == -1:
            # Neutral target — no garrison growth to integrate, but still
            # apply the over-commit boost so the fleet-speed bonus kicks
            # in and we deposit a non-trivial defender on arrival.
            if FLEET_OVERCOMMIT > 1.0:
                boosted = math.ceil(FLEET_OVERCOMMIT * intent.ships)
                intent.ships = min(boosted, int(src.ships))
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
        # H21 / [F] pre-reinforce: scan the WorldModel arrival ledger
        # for ENEMY fleets landing in (eta, eta + PRE_REINFORCE_WINDOW]
        # and add buffer to our intent so the post-capture garrison
        # survives the strongest such follow-up (TID 698478, "they
        # take it back at no cost" pattern). Per-target production
        # during the window offsets some of the threat.
        if PRE_REINFORCE_WINDOW > 0 and model is not None:
            strongest = 0
            strongest_eta = eta + 1
            for f_eta, f_owner, f_ships in model.ledger.get(target.id, ()):
                if f_owner == world.my_id:
                    continue
                if eta < f_eta <= eta + PRE_REINFORCE_WINDOW and f_ships > strongest:
                    strongest = int(f_ships)
                    strongest_eta = int(f_eta)
            if strongest > 0:
                # During (eta, strongest_eta] we own + grow the planet.
                prod_during = int(target.production) * (strongest_eta - eta)
                deficit = strongest - prod_during
                if deficit > 0:
                    intent.ships = max(intent.ships, intent.ships + deficit + 1)
        # 1.1× over-commit AFTER the production-aware sizing. Falsified
        # at v7; preserved as a dormant flag.
        if FLEET_OVERCOMMIT > 1.0:
            boosted = math.ceil(FLEET_OVERCOMMIT * intent.ships)
            intent.ships = min(boosted, int(src.ships))
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

# === inlined: lib/scoring.py ===


import math
import os

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

fleet_speed = speed

T_TOTAL_DEFAULT: int = 500


def eta_proxy(mine: Planet, target: Planet) -> int:
    """Conservative integer-turn ETA for a launch from `mine` to `target`.

    Uses `fleet_speed(target.ships + 1)` as the speed proxy — this is the
    speed of the minimum-cover fleet a strategy would send before
    `arrival_size` inflates it. Returns 0 for zero-distance pairs.
    """
    d = dist((mine.x, mine.y), (target.x, target.y))
    if d == 0.0:
        return 0
    v = fleet_speed(target.ships + 1)
    return int(math.ceil(d / v))


def projected_garrison(target: Planet, eta: int) -> int:
    """Predicted target garrison at arrival.

    Neutral targets (`owner == -1`) don't produce; their garrison stays
    flat. Owned targets (ours or enemy) grow by `production * eta`.
    """
    if target.owner == -1:
        return target.ships
    return target.ships + target.production * eta


def s_needed(target: Planet, eta: int) -> int:
    """Minimum fleet size to capture `target` at arrival.

    Mirrors `lib.mechanism.arrival_size`'s formula so strategy-side gates
    use the same number the mechanism layer will end up sizing fleets
    against. The +1 ensures strict win on the combat resolver.
    """
    return projected_garrison(target, eta) + 1


def horizon(step: int, eta: int, t_total: int = T_TOTAL_DEFAULT) -> int:
    """Remaining turns the captured planet can produce for us.

    `H = max(0, T_total - step - eta)`. Late-game this collapses toward 0
    so production-weighted scores naturally pivot to cheap snipes /
    denial in the closing turns.
    """
    return max(0, t_total - step - eta)


# Discount factor for present-value horizon valuation. With γ < 1, future
# production is discounted at γ per turn from the arrival step. At γ = 1.0
# (the default) the function reduces to the linear horizon above; that
# preserves the pre-PV scoring shape so existing snipe/reinforce tests pass
# unchanged. A/B candidates set γ < 1 (typically 0.99 per discussion-thread
# TID 699003) via `scripts/ab_variants.py --variant pv PV_GAMMA=0.99`.
#
# Env-var override (PV_GAMMA): an agent that wants PV-aware proposers can
# `os.environ.setdefault("PV_GAMMA", "0.99")` BEFORE this module is
# imported. v7_pv (live μ=1064.4) is v7_0_drop_one + this single config.
PV_GAMMA = float(os.environ.get("PV_GAMMA", "1.0"))


# Sensitivity coefficient for the 3-NN allegiance danger field (H17 /
# TID 699003). Multiplicative on snipe + reinforce score:
#     score *= max(MIN_DANGER3_MULT, 1.0 + DANGER3_KAPPA · danger_3nn(target))
# At κ=0 the field has no effect — preserves the snipe/reinforce score
# numerics for the existing parity tests. Typical A/B candidate values
# are 0.1-0.3 (each ally-neighbour boosts score by 10-30 %, each enemy
# discounts it by the same). `MIN_DANGER3_MULT` clamps the multiplier
# above zero so a 3-enemy neighbourhood at κ ≥ 1/3 doesn't zero the score.
DANGER3_KAPPA = 0.0
MIN_DANGER3_MULT = 0.05


def pv_horizon(
    step: int, eta: int, gamma: float = PV_GAMMA,
    t_total: int = T_TOTAL_DEFAULT,
) -> float:
    """Present-value of a unit production stream starting at `step + eta`.

    With γ < 1: `γ^eta · Σ_{k=0}^{h-1} γ^k = γ^eta · (1-γ^h)/(1-γ)` where
    `h = t_total - step - eta`. The early arrival turns count for full
    γ^eta weight; far-future production is exponentially discounted. At
    γ = 1.0 the formula degenerates to the linear horizon (`h` turns of
    equal weight), matching `horizon()` above modulo floating-point cast.
    Returns 0 when no production turns remain.
    """
    h = t_total - step - eta
    if h <= 0:
        return 0.0
    if gamma >= 1.0:
        return float(h)
    return (gamma ** eta) * (1.0 - gamma ** h) / (1.0 - gamma)


def margin_multiplier(target: Planet, my_id: int) -> int:
    """Owner-flip multiplier for margin-based scoring.

    Captures contribute to the margin (my_ships - their_ships):
    - Self (already ours): 0 — reinforce moves don't change margin.
    - Neutral: 1 — we gain P/turn going forward.
    - Enemy: 2 — we gain P/turn AND deny them P/turn (zero-sum).
    """
    if target.owner == my_id:
        return 0
    if target.owner == -1:
        return 1
    return 2


def expected_hold(
    target_id: int, eta: int, world, model,
    t_total: int = T_TOTAL_DEFAULT,
) -> int:
    """Predicted turns we'll own `target_id` after capturing at our
    arrival turn `now + eta`. Capped above by remaining-game-end and
    below at 0.

    Computed as `min(remaining_game, threat_eta − eta)` where
    `threat_eta = WorldModel.time_to_enemy_threat(target, my_id)`.
    `None` from the helper (no enemy threat) → saturate at the
    remaining-game-end cap.

    Used by HAV-1: drives the `pv_horizon` `t_total` per-target,
    instead of the flat `EPISODE_STEPS − step − eta` form. Targets
    deep in enemy space get short `expected_hold` (the enemy retakes
    them quickly); targets in our cluster saturate.
    """
    step_now = int(world.step)
    remaining = max(0, t_total - step_now - eta)
    if remaining == 0:
        return 0
    # PI 2026-05-21 fix — when `BASELINE_ORBITAL_SAFETY=1`, pass
    # `arrival_eta=eta` so enemy threat is computed from the target's
    # predicted position at our arrival, not its current position.
    # Without this, orbiting targets that rotate INTO enemy territory
    # by our arrival are silently scored as safe (long hold), pushing
    # us to capture planets we will immediately lose. Gated for A/B
    # testing; default OFF preserves backwards compat with sub 52882014.
    import os as _os
    if _os.environ.get("BASELINE_ORBITAL_SAFETY", "0") == "1":
        threat = model.time_to_enemy_threat(target_id, world.my_id, world,
                                             arrival_eta=int(eta))
    else:
        threat = model.time_to_enemy_threat(target_id, world.my_id, world)
    if threat is None:
        # No enemy can plausibly reach — saturate at remaining game.
        return remaining
    # The threat fleet arrives at `threat` from now; we arrive at
    # `eta`; we hold from arrival to threat-arrival.
    hold = max(0, int(threat) - int(eta))
    return min(remaining, hold)


__all__ = [
    "T_TOTAL_DEFAULT",
    "PV_GAMMA",
    "eta_proxy",
    "projected_garrison",
    "s_needed",
    "horizon",
    "pv_horizon",
    "expected_hold",
    "margin_multiplier",
]

# === inlined: lib/missions/snipe.py ===


import math
import os

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
# NON_LEADER_MULTIPLIER — H20 / [E] / TID 697397: Gemini's Day-2 4P
# kingmaker logic boosts the leader (×1.5, above) AND down-weights
# non-leader opponents (×0.8). Default 1.0 = no down-weighting (current
# behaviour). A/B candidate: 0.8. Like LEADER_MULTIPLIER, only fires
# when our_rank >= 2 — 2P unaffected.
NON_LEADER_MULTIPLIER = 1.0

# H10 (2026-05-14): enemy-target multiplier. Top-10 replay analysis
# (knowledge-base/concepts/top-performer-strategies.md §H10) finds
# enemy-target picks at 32% vs midpack 14% — a ×2.3 gap. Multiplying
# the snipe priority by ENEMY_MULTIPLIER when `t.owner ≠ ourselves
# AND t.owner ≠ -1` shifts the scorer toward enemy captures over
# neutral expansion. Default 1.0 (identity) so v7_0 / v3.5.1 baselines
# stay unchanged; v7_7 sets it to 1.3 inside `agent(obs)` per the
# safe-monkey-patch friction pattern (set per-agent-call, never
# persists across processes; see `module-mutation-patching-has-
# worker-reuse-race`, 2026-05-12).
ENEMY_MULTIPLIER = 1.0

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


# Drop comet chasing entirely (H15 / H18). The top-10 capture-rate
# fingerprint is 3.4 % vs midpack 13.4 %; emanuellcs's public spoofing
# agent formalises a break-even filter. Additionally, our [C] audit on
# 2026-05-13 confirmed that `lib/trajectory.predict_fleet_fate` treats
# comets as STATIC (the `is_orbiting` gate excludes comets — they
# follow path indices, not orbital math), so the fleet's aim assumes
# the comet stays put for the entire flight. By the time the fleet
# arrives, the comet has moved `eta * cometSpeed ≈ eta * 4` units
# along its path. Most comet captures silently fail this way.
# DROP_COMET_TARGETS = 1 filters comet targets out of the proposer
# entirely. Default 0 = current behaviour.
DROP_COMET_TARGETS = 0

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


# HAV (Hold-Aware Value) — see plan file 2026-05-14 section.
# USE_HAV=1 caps each target's PV horizon by the time-to-enemy-threat
# at that target. Targets in enemy territory get shortened hold, often
# dropping value to zero (proposer skips). Default 0 = identity (PV
# at full remaining-game horizon).
USE_HAV = 0

# Tiered Mission emission. When USE_HOLDING_TIER=1, additionally emit
# a "holding" Mission per (src, target) sized to absorb expected
# enemy counter-attack within HOLD_WINDOW turns. When
# USE_OPERATIONAL_TIER=1, additionally emit an "operational" Mission
# sized to also project a follow-on capture from the captured target
# to its cheapest reachable nearby unowned planet within
# FOLLOWON_RADIUS. settle_plan picks the highest-scoring tier per
# source. Defaults 0 = no extra tiers, current behaviour.
USE_HOLDING_TIER = 0
USE_OPERATIONAL_TIER = 0

# Tier constants — see plan file for derivation. All tunable via
# ab_variants. The defaults are conservative starting points; expect
# to sweep them after a binary on/off PASS.
SOURCE_DEFENSE_RESERVE = 8     # never strand the source below this
OP_RESERVE = 5                 # extra ships in operational fleet
MIN_FOLLOWON_HOLD = 10         # don't propose op-tier if followon is shaky
FOOTHOLD_DISCOUNT = 0.5        # follow-on value weighted at half-credit
HOLD_WINDOW = 10               # look this many turns past arrival for counter-attack
FOLLOWON_RADIUS = 40.0         # max distance from target → followon
HAV_MIN_HOLD = 5               # floor on HAV expected-hold (turns).
                               # `time_to_enemy_threat` is conservative
                               # (any nearby enemy "could" launch);
                               # the floor prevents over-pruning of
                               # contested-but-still-valuable targets.


def _max_enemy_arrival_within(
    ledger_entries: list, my_id: int, eta_lo: int, eta_hi: int,
) -> int:
    """Sum of enemy ship counts in the ledger arriving within
    `[eta_lo, eta_hi]` (inclusive). Used by the holding tier to
    estimate the counter-attack the post-capture garrison must
    survive."""
    if not ledger_entries:
        return 0
    total = 0
    for f_eta, f_owner, f_ships in ledger_entries:
        if f_owner == my_id:
            continue
        if eta_lo <= f_eta <= eta_hi:
            total += int(f_ships)
    return total


def _followon_hold_estimate(
    followon, target, world: World, model: WorldModel, my_id: int, f_eta: int,
    arrival_eta: int = 0,
) -> int:
    """Estimate how many turns we'd hold `followon` after capturing it
    from `target` (the about-to-be-captured forward base).

    Like `expected_hold` but explicitly EXCLUDES `target` from the
    enemy threat set, because we're about to flip target to our side.

    `arrival_eta` (B3, 2026-05-22) — when > 0 AND
    `BASELINE_ORBITAL_SAFETY=1`, predict `followon` and each enemy planet
    position at our arrival to `followon` (in turns from now) via
    `_position_at`. Returns `incoming_enemy_eta_after(arrival_eta - 1)`
    so simultaneous-at-arrival fleets count as future threats (the
    `expected_hold` site uses `incoming_enemy_eta_after(arrival_eta)`
    instead — followon semantics differ because the followon ETA spans
    *from now* through the target capture; we want any inbound that
    survives arrival). Default 0 preserves prior behavior.
    """
    step_now = int(world.step)
    remaining = max(0, EPISODE_STEPS - step_now - f_eta)
    if remaining == 0:
        return 0

    orbital_safety = os.environ.get("BASELINE_ORBITAL_SAFETY", "0") == "1"
    omega = float(getattr(world, "omega", 0.0))
    use_predict = (
        orbital_safety and omega != 0.0 and arrival_eta > 0
    )

    # Semantic anchor: legacy treats `f_eta` (target → followon travel)
    # as "turns from now" (assumes we're at target now); the fixed path
    # anchors at our TARGET ARRIVAL turn (arrival_eta from now), so
    # `best` and `f_eta` are both relative to that anchor. We convert
    # in-flight ETAs by subtracting `arrival_eta` so the final
    # `best - f_eta` subtraction stays unit-consistent.
    best: int | None
    if use_predict:
        inbound_abs = model.incoming_enemy_eta_after(
            followon.id, my_id, arrival_eta - 1,
        )
        best = (inbound_abs - arrival_eta) if inbound_abs is not None else None
    else:
        best = model.incoming_enemy_eta(followon.id, my_id)

    # Followon's position at our arrival.
    if use_predict:
        fx, fy = _position_at(followon, omega, arrival_eta)
    else:
        fx, fy = float(followon.x), float(followon.y)

    # Potential launches from each enemy planet EXCEPT the target.
    for p in world.planets_by_id.values():
        if p.id == followon.id or p.id == target.id:
            continue
        if p.owner == my_id or p.owner == -1:
            continue
        if p.ships <= 0:
            continue
        if use_predict:
            px, py = _position_at(p, omega, arrival_eta)
        else:
            px, py = float(p.x), float(p.y)
        dx = fx - px
        dy = fy - py
        d = math.hypot(dx, dy)
        v = fleet_speed(int(p.ships))
        if v <= 0:
            continue
        # Travel time from enemy planet (at anchor moment) to followon
        # — same units (turns from anchor) in both modes.
        eta_travel = int(math.ceil(d / v))
        if best is None or eta_travel < best:
            best = eta_travel

    if best is None:
        return remaining
    hold = max(0, int(best) - int(f_eta))
    return min(remaining, hold)


def _best_followon(target, world: World, model: WorldModel, my_id: int,
                   radius: float, arrival_eta: int = 0):
    """Find the cheapest reachable nearby unowned planet from `target`,
    returning `(followon_planet, capture_cost, eta_from_target,
    expected_hold)` or `None` if no follow-on qualifies.

    Used by the operational tier: the captured `target` becomes a
    forward base; the follow-on is the next move from there. Only
    considers planets that are NOT ours, NOT comets, within `radius`
    units of `target`, and predicted to be holdable for at least
    `MIN_FOLLOWON_HOLD` turns after the follow-on arrives (computed
    AS IF target were already ours).

    `arrival_eta` (B4, 2026-05-22) — when > 0 AND `BASELINE_ORBITAL_SAFETY=1`,
    the target's position and each followon candidate's position rotate
    by our arrival_eta; the launch-from-target distance, hence `f_eta`,
    is computed from predicted positions at `arrival_eta`. The downstream
    `_followon_hold_estimate` is then called with `arrival_eta=arrival_eta
    + f_eta` (when WE arrive at the followon, from now). Default 0
    preserves prior behavior.
    """
    orbital_safety = os.environ.get("BASELINE_ORBITAL_SAFETY", "0") == "1"
    omega = float(getattr(world, "omega", 0.0))
    use_predict = (
        orbital_safety and omega != 0.0 and arrival_eta > 0
    )
    if use_predict:
        tgt_x, tgt_y = _position_at(target, omega, arrival_eta)
    else:
        tgt_x, tgt_y = float(target.x), float(target.y)

    candidates = []
    for n in world.planets_by_id.values():
        if n.id == target.id:
            continue
        if n.owner == my_id:
            continue
        if n.id in world.comet_ids:
            continue
        if use_predict:
            nx, ny = _position_at(n, omega, arrival_eta)
        else:
            nx, ny = float(n.x), float(n.y)
        dx = nx - tgt_x
        dy = ny - tgt_y
        d = math.hypot(dx, dy)
        if d > radius:
            continue
        cost = max(1, int(n.ships) + 1)
        v = fleet_speed(cost)
        if v <= 0:
            continue
        f_eta = int(math.ceil(d / v))
        followon_arrival_eta = (arrival_eta + f_eta) if use_predict else 0
        eh = _followon_hold_estimate(
            n, target, world, model, my_id, f_eta,
            arrival_eta=followon_arrival_eta,
        )
        if eh < MIN_FOLLOWON_HOLD:
            continue
        candidates.append((n, cost, f_eta, eh))
    if not candidates:
        return None
    # Pick the highest production/cost ratio — best per-ship payoff.
    candidates.sort(key=lambda x: x[0].production / max(1, x[1]), reverse=True)
    return candidates[0]


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
            if DROP_COMET_TARGETS and t.id in world.comet_ids:
                # H15/H18 — drop comet targets entirely (see flag docstring).
                continue
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
            # comet that's about to depart. `pv_horizon` with PV_GAMMA=1.0
            # is identity to the prior linear `max(0, lifetime − eta)`
            # form; PV_GAMMA<1.0 discounts future production geometrically
            # (TID 699003).
            is_comet = t.id in world.comet_ids
            if is_comet:
                rem = comet_remaining_lifetime(t.id, world)
                if (rem or 0) <= eta:
                    # H15 (main 2026-05-13 cb02fd9): comet leaves the
                    # board before our fleet arrives — don't emit a
                    # Mission. Lets the source's runner-up win the
                    # per-source slot in settle_plan instead of
                    # consuming it with a degenerate score≈0 candidate.
                    continue
                # PV horizon over the remaining-lifetime budget; identity
                # to `max(0, rem - eta)` at PV_GAMMA=1.0.
                time_to_hold = max(0.0, pv_horizon(0, eta, PV_GAMMA, rem or 0))
            else:
                if USE_HAV:
                    # HAV-1: cap PV horizon by predicted hold window
                    # (time-to-enemy-threat at target). Soft floor at
                    # HAV_MIN_HOLD turns rather than dropping the
                    # Mission — `time_to_enemy_threat` is over-pessimistic
                    # for centrally-located targets (it assumes enemy
                    # will dedicate full garrison to this target which
                    # isn't realistic). Floor lets settle_plan pick a
                    # contested target if it's still the best option.
                    eh = expected_hold(t.id, eta, world, model, EPISODE_STEPS)
                    eh = max(HAV_MIN_HOLD, eh)
                    time_to_hold = max(1.0, pv_horizon(0, 0, PV_GAMMA, eh))
                else:
                    time_to_hold = max(
                        1.0, pv_horizon(step_now, eta, PV_GAMMA, EPISODE_STEPS)
                    )
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
            if spoiler_on:
                if t.owner == leader_pid:
                    priority *= LEADER_MULTIPLIER
                elif t.owner != -1 and t.owner != world.my_id:
                    # Non-leader other player's planet — Gemini-style 4P
                    # kingmaker down-weight (H20). Neutrals + our own
                    # planets are unaffected.
                    priority *= NON_LEADER_MULTIPLIER
            # H10 enemy-target multiplier (default 1.0 = no change).
            # Applies whenever the target is owned by an enemy (not us,
            # not neutral). v7_7 sets ENEMY_MULTIPLIER=1.3 to bias
            # toward enemy snipes. Composes multiplicatively with the
            # H20 NON_LEADER_MULTIPLIER above when both fire.
            if t.owner >= 0 and t.owner != world.my_id:
                priority *= ENEMY_MULTIPLIER
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
            # 3-NN allegiance danger map (H17). At κ=0 (default) the
            # multiplier is exactly 1.0 — no effect on existing tests.
            if DANGER3_KAPPA != 0.0:
                d3 = danger_3nn(
                    (t.x, t.y), t.id,
                    list(world.planets_by_id.values()),
                    world.my_id,
                )
                score *= max(MIN_DANGER3_MULT, 1.0 + DANGER3_KAPPA * d3)

            missions.append(Mission(
                mission_class="snipe",
                src_id=src.id,
                target_id=t.id,
                ships=base_ships,
                score=score,
                eta=eta,
            ))

            # --------- Tier 2: Holding (2026-05-14 plan, HAV-2) ----------
            # Size the fleet to absorb the strongest expected enemy
            # arrival within HOLD_WINDOW turns of our capture. Skip
            # comets (their threat model is dominated by lifetime, not
            # counter-attack).
            S_hold: int | None = None
            if USE_HOLDING_TIER and not is_comet:
                counter = _max_enemy_arrival_within(
                    model.ledger.get(t.id, []),
                    my_id=world.my_id,
                    eta_lo=eta + 1, eta_hi=eta + HOLD_WINDOW,
                )
                if counter > 0:
                    prod_during = int(t.production) * HOLD_WINDOW
                    deficit = counter - prod_during
                    if deficit > 0:
                        S_hold = base_ships + int(deficit) + 1
                if S_hold is not None and S_hold > base_ships:
                    if int(src.ships) - S_hold >= SOURCE_DEFENSE_RESERVE:
                        src_threat = model.incoming_enemy_eta(src.id, world.my_id)
                        if src_threat is None or src_threat > eta:
                            # Holding tier permanently denies the
                            # planet to enemy → use full remaining-game
                            # horizon (not the HAV-capped one).
                            hold_t = max(
                                1.0,
                                pv_horizon(step_now, eta, PV_GAMMA, EPISODE_STEPS),
                            )
                            hold_value = t.production * hold_t
                            hold_score = priority * hold_value / (
                                S_hold + d + AIRTIME_PENALTY_WEIGHT * eta + 1.0
                            )
                            if DANGER3_KAPPA != 0.0:
                                hold_score *= max(
                                    MIN_DANGER3_MULT,
                                    1.0 + DANGER3_KAPPA * danger_3nn(
                                        (t.x, t.y), t.id,
                                        list(world.planets_by_id.values()),
                                        world.my_id,
                                    ),
                                )
                            missions.append(Mission(
                                mission_class="snipe",
                                src_id=src.id,
                                target_id=t.id,
                                ships=S_hold,
                                score=hold_score,
                                eta=eta,
                                note="hold",
                            ))
                    else:
                        S_hold = None  # source can't afford the holding tier

            # --------- Tier 3: Operational / foothold (HAV-3) -----------
            if USE_OPERATIONAL_TIER and not is_comet:
                foothold = _best_followon(
                    t, world, model, world.my_id, FOLLOWON_RADIUS,
                    arrival_eta=int(eta),
                )
                if foothold is not None:
                    f_target, f_cost, f_eta_from_t, f_hold = foothold
                    base_for_op = S_hold if S_hold is not None else base_ships
                    S_op = base_for_op + f_cost + OP_RESERVE
                    if S_op > base_ships and int(src.ships) - S_op >= SOURCE_DEFENSE_RESERVE:
                        src_threat = model.incoming_enemy_eta(src.id, world.my_id)
                        if src_threat is None or src_threat > eta:
                            # Capture value: full hold (we're holding +
                            # projecting from this base).
                            op_t = max(
                                1.0,
                                pv_horizon(step_now, eta, PV_GAMMA, EPISODE_STEPS),
                            )
                            op_value = t.production * op_t
                            # Foothold value: discounted follow-on PV.
                            f_pv = pv_horizon(0, 0, PV_GAMMA, f_hold)
                            op_value += FOOTHOLD_DISCOUNT * f_target.production * f_pv
                            op_score = priority * op_value / (
                                S_op + d + AIRTIME_PENALTY_WEIGHT * eta + 1.0
                            )
                            if DANGER3_KAPPA != 0.0:
                                op_score *= max(
                                    MIN_DANGER3_MULT,
                                    1.0 + DANGER3_KAPPA * danger_3nn(
                                        (t.x, t.y), t.id,
                                        list(world.planets_by_id.values()),
                                        world.my_id,
                                    ),
                                )
                            missions.append(Mission(
                                mission_class="snipe",
                                src_id=src.id,
                                target_id=t.id,
                                ships=S_op,
                                score=op_score,
                                eta=eta,
                                note=f"op→{f_target.id}",
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
            time_to_hold = max(
                1.0, pv_horizon(step_now, eta, PV_GAMMA, EPISODE_STEPS)
            )
            value = d.production * time_to_hold
            score = value / (cost + d_dist + 1.0)
            # 3-NN allegiance danger map (H17). κ=0 (default) is identity.
            # Same sign as snipe: a planet in our cluster is worth more
            # to defend (we'll hold it and gain a contiguous block);
            # a planet surrounded by enemies is harder to keep — even
            # if defended now, it'll cycle back to them in a few turns,
            # so de-prioritise reinforcement of isolated salient planets.
            if DANGER3_KAPPA != 0.0:
                d3 = danger_3nn(
                    (d.x, d.y), d.id,
                    list(world.planets_by_id.values()),
                    world.my_id,
                )
                score *= max(MIN_DANGER3_MULT, 1.0 + DANGER3_KAPPA * d3)
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

# === inlined: lib/missions/opening.py ===


import math

fleet_speed = speed

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


def propose_opening_missions(world: World, model: WorldModel) -> list[Mission]:
    """One Mission per (our source with ships>8, neutral target) pair,
    fired only during the opening window. Score = production ×
    (remaining_steps)^1.5 / (distance + 1)."""
    if not USE_OPENING_MISSION:
        return []
    if int(world.step) > OPENING_WINDOW:
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

# === inlined: lib/missions/drain.py ===


import math

fleet_speed = speed

EPISODE_STEPS = 500
MIN_DRAIN_SHIPS = 30           # only drain when there's genuine surplus
RESERVE_KEEP = 8               # always leave a defender behind
SAFE_ETA_BUFFER = 5            # require enemy ETA > our ETA + this
DRAIN_BONUS = 1.10             # mild bonus for using SAFE surplus

# Mission Renaissance gate. Default 0 = disabled. A/B candidate: 1.
USE_DRAIN_MISSION = 0


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
            # Quick sanity: target must be capturable by our drain force.
            base_capture = max(1, int(t.ships) + 1)
            if drain_ships < base_capture:
                continue
            # Predicted ownership at arrival — skip if already ours.
            pred_owner = model.owner_at(t.id, our_eta)
            if pred_owner == world.my_id:
                continue
            # Score uses the standard cost-aware ROI shape (rebalanced
            # denominator from wave 1b), bumped by DRAIN_BONUS because
            # this is verified-safe surplus.
            is_comet = t.id in world.comet_ids
            if is_comet:
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

# === inlined: lib/missions/gang_up.py ===


import math

fleet_speed = speed

EPISODE_STEPS = 500
MAX_DELAY = 3                  # cap on slower-source delay
PAIR_SHARE = 0.7               # each paired source sends 70% of its garrison
SINGLE_SOURCE_AFFORDABLE_RATIO = 0.85  # below this, target is "out of reach for one"
GANG_UP_BONUS = 1.30           # timed arrivals > staggered

# Mission Renaissance gate. Distinct from mechanism.GANG_UP_ENABLED
# (which gates the post-Mission gang_up_size mechanism). This flag
# gates the *proposer*. Default 0 = disabled. A/B candidate: 1.
USE_GANG_UP_MISSION = 0


def propose_gang_up_missions(world: World, model: WorldModel) -> list[Mission]:
    """Pair our top two reachable sources at any target where a single
    source's affordable fleet falls short."""
    if not USE_GANG_UP_MISSION:
        return []
    my_planets = [
        p for p in world.planets_by_id.values() if p.owner == world.my_id
    ]
    if len(my_planets) < 2:
        return []
    targets = [
        p for p in world.planets_by_id.values() if p.owner != world.my_id
    ]
    if not targets:
        return []

    step_now = int(world.step)
    missions: list[Mission] = []
    for t in targets:
        # Skip comet targets — gang_up timing on a moving target is too
        # noisy; comets are best handled by the dedicated snipe Mission.
        if t.id in world.comet_ids:
            continue

        # Rank sources by ETA at PAIR_SHARE fleet size — that's what they'd
        # actually launch in a pair.
        source_eta = []
        for src in my_planets:
            send = max(1, int(src.ships * PAIR_SHARE))
            if send < 2:
                continue
            d = math.hypot(t.x - src.x, t.y - src.y)
            v = fleet_speed(send)
            eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
            source_eta.append((eta, src, d, send))
        if len(source_eta) < 2:
            continue
        source_eta.sort(key=lambda x: x[0])
        eta1, s1, d1, ships1 = source_eta[0]
        eta2, s2, d2, ships2 = source_eta[1]
        # Pair must be timeable within MAX_DELAY: slower source must
        # arrive within MAX_DELAY turns of fastest, OR fastest can
        # delay to align.
        eta_gap = eta2 - eta1
        if eta_gap > MAX_DELAY:
            continue
        # Joint ship count must exceed any single-source affordable;
        # otherwise gang_up isn't needed.
        # Predicted garrison at the JOINT arrival step (eta2 — the
        # later arrival; both fleets land same step after timing).
        pred_owner = model.owner_at(t.id, eta2)
        if pred_owner == world.my_id:
            continue
        pred_garrison = model.ships_at(t.id, eta2) or 0.0
        single_source_affordable = max(ships1, ships2)
        if single_source_affordable >= SINGLE_SOURCE_AFFORDABLE_RATIO * pred_garrison:
            # One source can handle this target; skip pair.
            continue
        combined = ships1 + ships2
        # Must actually be able to capture jointly (defensive — even
        # gang_up has limits).
        if combined < pred_garrison + 1:
            continue
        # Joint score: production × time-to-hold, denominator over
        # mean distance + combined-ship cost (halved as in wave 1b).
        time_to_hold = max(1, EPISODE_STEPS - step_now - eta2)
        value = float(t.production) * time_to_hold
        mean_d = (d1 + d2) / 2.0
        score = GANG_UP_BONUS * value / (0.5 * combined + mean_d + 1.0)

        # Emit two Missions: the "lead" (fastest) at its natural eta,
        # the "follow" (slower) at its natural eta. Both share the
        # same score so settle_plan ranks them together.
        # The natural ETAs already align within MAX_DELAY turns of each
        # other; the env combat resolver groups same-step arrivals at
        # each integer step, so a 0-3 turn gap is acceptable.
        missions.append(Mission(
            mission_class="gang_up_lead",
            src_id=s1.id,
            target_id=t.id,
            ships=ships1,
            score=score,
            eta=eta1,
            note=f"pair_with={s2.id}",
        ))
        missions.append(Mission(
            mission_class="gang_up_follow",
            src_id=s2.id,
            target_id=t.id,
            ships=ships2,
            score=score,
            eta=eta2,
            note=f"pair_with={s1.id}",
        ))
    return missions

# === inlined: lib/missions/opp_archetypes.py ===


import math
from typing import Any



# ---------------------------------------------------------------------------
# POV helpers
# ---------------------------------------------------------------------------


def opp_pov_obs(obs: Any, opp_id: int) -> dict:
    """Return a copy of `obs` with `player = opp_id`, suitable for
    `World.from_obs` to produce an opp-POV `World`.

    Same technique as `lib.v7_search._opp_incumbent_action`; factored
    out so archetype builders can be tested in isolation.
    """
    if isinstance(obs, dict):
        obs2 = dict(obs)
        obs2["player"] = opp_id
        return obs2
    keys = (
        "player", "planets", "fleets", "angular_velocity",
        "initial_planets", "comet_planet_ids", "comets",
        "step", "next_fleet_id",
    )
    obs2: dict = {}
    for k in keys:
        v = getattr(obs, k, None)
        if v is not None:
            obs2[k] = v
    obs2["player"] = opp_id
    return obs2


def _largest_source(world: World) -> object | None:
    """Opp's planet with the most ships (excluding comets)."""
    owned = [
        p for p in world.planets_by_id.values()
        if p.owner == world.my_id and p.id not in world.comet_ids
    ]
    if not owned:
        return None
    return max(owned, key=lambda p: p.ships)


def _our_largest_by_ships(world: World) -> object | None:
    """The target — our (= non-opp non-neutral) planet with most ships."""
    candidates = [
        p for p in world.planets_by_id.values()
        if p.owner != world.my_id and p.owner != -1
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.ships)


def _our_largest_by_production(world: World) -> object | None:
    """The target — our planet with highest production."""
    candidates = [
        p for p in world.planets_by_id.values()
        if p.owner != world.my_id and p.owner != -1
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.production, p.ships))


# ---------------------------------------------------------------------------
# Archetypes (opp's POV)
# ---------------------------------------------------------------------------


def archetype_no_launch() -> list[list]:
    """Opp passes the turn — pure defensive baseline."""
    return []


def archetype_v351(opp_world: World, opp_model: WorldModel, opp_obs: dict) -> list[list]:
    """Opp plays the v3.5.1 aggressive pipeline (snipe + reinforce)."""
    if not opp_world.planets_by_id:
        return []
    missions = (
        propose_snipe_missions(opp_world, opp_model, aggressive=True)
        + propose_reinforce_missions(opp_world, opp_model)
    )
    intents = settle_plan(missions, opp_world, opp_model)
    return realize(intents, opp_obs, mechanisms=DEFAULT_MECHANISMS, model=opp_model)


def archetype_counter_reinforce(
    opp_world: World, opp_obs: dict, our_intents: list[Intent],
) -> list[list]:
    """For each of our intents, opp sends `our_ships + 1` to that target
    from opp's nearest planet that can fund it. If multiple of our
    launches go to the same target, only the first (strongest) one is
    countered. Empty if we have no intents or opp has no fundable
    source within reach.
    """
    if not our_intents or not opp_world.planets_by_id:
        return []
    seen_targets: set[int] = set()
    counter: list[Intent] = []
    used_sources: set[int] = set()
    for our_intent in our_intents:
        tgt_id = int(our_intent.target_id)
        if tgt_id in seen_targets:
            continue
        target = opp_world.planets_by_id.get(tgt_id)
        if target is None:
            continue
        needed = int(our_intent.ships) + 1
        candidates = [
            (math.hypot(p.x - target.x, p.y - target.y), p)
            for p in opp_world.planets_by_id.values()
            if (
                p.owner == opp_world.my_id
                and p.id not in opp_world.comet_ids
                and p.id not in used_sources
                and int(p.ships) >= needed
            )
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        _, src = candidates[0]
        seen_targets.add(tgt_id)
        used_sources.add(src.id)
        counter.append(Intent(src_id=src.id, target_id=tgt_id, ships=needed))
    return realize(counter, opp_obs, mechanisms=DEFAULT_MECHANISMS)


def archetype_counter_snipe(opp_world: World, opp_obs: dict) -> list[list]:
    """Opp's biggest source fires 70 % of its garrison at our top-ships
    planet. Single-launch — captures the "concentrated attack" archetype
    yijue1 / bowwowforeach use in top-10 replays.
    """
    src = _largest_source(opp_world)
    tgt = _our_largest_by_ships(opp_world)
    if src is None or tgt is None:
        return []
    if int(src.ships) < 5:
        return []
    ships = max(1, int(src.ships * 0.7))
    return realize(
        [Intent(src_id=src.id, target_id=tgt.id, ships=ships)],
        opp_obs, mechanisms=DEFAULT_MECHANISMS,
    )


def archetype_cross_attack(opp_world: World, opp_obs: dict) -> list[list]:
    """Opp's biggest source fires 70 % of its garrison at our highest-
    production planet. Different from `archetype_counter_snipe` on
    most boards — top-ships ≠ top-prod when an enemy is loading up a
    cheap planet vs holding a high-prod home.
    """
    src = _largest_source(opp_world)
    tgt = _our_largest_by_production(opp_world)
    if src is None or tgt is None:
        return []
    if int(src.ships) < 5:
        return []
    ships = max(1, int(src.ships * 0.7))
    return realize(
        [Intent(src_id=src.id, target_id=tgt.id, ships=ships)],
        opp_obs, mechanisms=DEFAULT_MECHANISMS,
    )


# ---------------------------------------------------------------------------
# Top-level: build the archetype response set
# ---------------------------------------------------------------------------


def build_opp_archetypes(
    opp_obs: dict, our_intents: list[Intent],
) -> list[list[list]]:
    """Return a list of distinct env-format opp actions covering the
    archetype set. Deduplicates exact matches so the payoff matrix
    isn't padded with identical rows.

    `opp_obs` must already have `player = opp_id` (see `opp_pov_obs`).
    `our_intents` is the launch list we plan to emit this turn — used
    by the counter-reinforce archetype.
    """
    opp_world = World.from_obs(opp_obs)
    if not opp_world.planets_by_id:
        return [archetype_no_launch()]
    opp_model = WorldModel.from_world(opp_world)

    archetypes: list[list[list]] = [
        archetype_no_launch(),
        archetype_v351(opp_world, opp_model, opp_obs),
        archetype_counter_reinforce(opp_world, opp_obs, our_intents),
        archetype_counter_snipe(opp_world, opp_obs),
        archetype_cross_attack(opp_world, opp_obs),
    ]

    # Deduplicate by exact equality; preserve order so row 0 (no-launch)
    # stays first — useful for tie-break debugging.
    seen: list[list[list]] = []
    for a in archetypes:
        if a not in seen:
            seen.append(a)
    return seen

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

# === inlined: lib/lookahead.py ===


import copy
from typing import Callable, Sequence

from kaggle_environments import make


def env_from_obs(obs, configuration: dict | None = None):
    """Build a fresh steppable env mirroring the current obs.

    Both player-states get the same public observation; only the
    `player` field is per-seat. status/reward are reset; action is None
    (will be filled in by step()).
    """
    cfg = dict(configuration or {})
    env = make("orbit_wars", configuration=cfg, debug=False)
    env.reset(num_agents=2)
    snapshot_keys = (
        "planets", "fleets", "comets", "comet_planet_ids",
        "initial_planets", "angular_velocity", "step", "next_fleet_id",
    )
    public = {k: copy.deepcopy(obs[k]) for k in snapshot_keys if k in obs}
    for i in range(2):
        env.state[i].observation.update(public)
        env.state[i].observation["player"] = i
        env.state[i].observation["remainingOverageTime"] = obs.get(
            "remainingOverageTime", 60.0
        )
        env.state[i].status = "ACTIVE"
        env.state[i].reward = 0
        env.state[i].action = None
    return env


def _ship_total_by_owner(observation) -> dict[int, float]:
    """Sum ships on owned planets + in fleets per owner."""
    totals: dict[int, float] = {}
    for p in observation.get("planets", []):
        owner = int(p[1])
        if owner >= 0:
            totals[owner] = totals.get(owner, 0.0) + float(p[5])
    for f in observation.get("fleets", []):
        owner = int(f[1])
        if owner >= 0:
            totals[owner] = totals.get(owner, 0.0) + float(f[6])
    return totals


def score_action(
    env,
    action: list,
    K: int,
    my_id: int,
    policy: Callable,
) -> float:
    """Sim<K> score of taking `action` this turn, then K-1 turns of
    `policy` as both players. Returns (our - opp) total ships.

    Caller is responsible for passing `env` already-reconstructed; this
    function clones it so the caller can reuse `env` across candidates.
    """
    clone = env.clone()
    opp_id = 1 - my_id
    # First step: our forced action; opp plays policy on their obs.
    a_opp = policy(clone.state[opp_id].observation)
    actions = [None, None]
    actions[my_id] = action
    actions[opp_id] = a_opp
    if not clone.done:
        clone.step(actions)
    # Remaining K-1 steps: both players use policy.
    for _ in range(max(0, K - 1)):
        if clone.done:
            break
        a0 = policy(clone.state[0].observation)
        a1 = policy(clone.state[1].observation)
        clone.step([a0, a1])
    totals = _ship_total_by_owner(clone.state[my_id].observation)
    return totals.get(my_id, 0.0) - totals.get(opp_id, 0.0)


def score_joint_action(
    env,
    our_action: list,
    opp_action: list,
    K: int,
    my_id: int,
    policy: Callable,
) -> float:
    """Sim<K> score with BOTH first-turn actions injected.

    Unlike `score_action` (which lets `policy` choose opp's first move),
    `score_joint_action` forces both `our_action` and `opp_action` on
    turn 0, then rolls forward K-1 turns under `policy` as both players.

    Used by maximin agents (e.g. v7.1+) to score a full N×M payoff
    matrix where both players' first moves are explicit candidates.
    Returns `(our_ships - opp_ships)` at the rollout's final state.

    Ported from `origin/claude/game-theory-strategy-analysis-0oH4N`.
    """
    clone = env.clone()
    opp_id = 1 - my_id
    actions = [None, None]
    actions[my_id] = our_action
    actions[opp_id] = opp_action
    if not clone.done:
        clone.step(actions)
    for _ in range(max(0, K - 1)):
        if clone.done:
            break
        a0 = policy(clone.state[0].observation)
        a1 = policy(clone.state[1].observation)
        clone.step([a0, a1])
    totals = _ship_total_by_owner(clone.state[my_id].observation)
    return totals.get(my_id, 0.0) - totals.get(opp_id, 0.0)


def score_joint_action_symmetric(
    env,
    our_action: list,
    opp_action: list,
    K: int,
    policy: Callable,
) -> float:
    """Seat-symmetric variant of `score_joint_action` — averages over
    the two seat assignments of `(our_action, opp_action)`.

    Motivation: the Orbit Wars env has a documented seat asymmetry
    (P1-favoring tie-breaks in identical self-play). Without
    symmetrization, that asymmetry leaks through Sim<K> into the
    maximin payoff matrix; both seats' matrices stop being seat-flips
    of each other, and the maximin picks diverge from the σ-mirror.
    Averaging the two seat assignments cancels the env-internal bias.

    Cost: 2× `score_joint_action`. Calling code budgets accordingly.

    Ported from `origin/claude/game-theory-strategy-analysis-0oH4N`.
    """
    a = score_joint_action(env, our_action, opp_action, K, my_id=0, policy=policy)
    b = score_joint_action(env, our_action, opp_action, K, my_id=1, policy=policy)
    return (a + b) / 2.0


def enumerate_drop_one_candidates(action: list) -> list[list]:
    """Generate the smallest non-trivial candidate set.

    Returns [action] + [action with launch i removed for each i].
    A pure subset of the incumbent — we never propose new launches the
    incumbent didn't already consider. With N launches we evaluate
    N + 1 candidates, which controls budget at ~(N+1) × K × 5.6 ms.

    For N=0 (incumbent does nothing) returns [[]] — only the empty
    action; no rollouts needed.
    """
    if not action:
        return [[]]
    cands: list[list] = [list(action)]
    for i in range(len(action)):
        cands.append([m for j, m in enumerate(action) if j != i])
    return cands

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


import math
from typing import Any, Callable

_fleet_speed = speed


Policy = Callable[[Any], list]


# ---------------------------------------------------------------------------
# Tier 0 — mirror self (= v3_snipe pipeline, aggressive sizing OFF)
# ---------------------------------------------------------------------------


def mirror_self_policy(obs: Any) -> list:
    """Run the v3_snipe pipeline against `obs`. Bit-exact equivalent of
    `agents/v3_snipe/main.py`'s `agent(obs)` body.

    Drop-in replacement for `lib.lookahead.score_action`'s `policy`
    argument when you want the Phase 2 default ("opponent plays v2").

    Phase 3c: reuses `_shared_world_model` from the obs if present
    (saves ~3.8 ms WorldModel rebuild when score_candidate has already
    computed it for the OTHER seat at the same step).
    """
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    cached = getattr(obs, "_shared_world_model", None)
    model = cached if cached is not None else WorldModel.from_world(world)
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

    Phase 3c: reuses `_shared_world_model` from the obs if present
    (saves ~3.8 ms WorldModel rebuild when score_candidate has already
    computed it for the OTHER seat at the same step).
    """
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    cached = getattr(obs, "_shared_world_model", None)
    model = cached if cached is not None else WorldModel.from_world(world)
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


def lite_greedy_policy(obs: Any) -> list:
    """Cheap opp policy: ROI-greedy launch picker, no WorldModel.

    Per-call cost is ~1-2 ms (raw obs only; no World object,
    no WorldModel.from_world, no mission framework, no mechanism stack).
    The mirror policies (tier 0, 1) take ~10 ms because they rebuild
    the WorldModel timeline every step. Use this when wallclock budget
    matters more than bit-identical top-tier behaviour (e.g. as the
    per-step opp policy in lookahead rollouts).

    Behaviour: for each owned planet with ships >= 5, find the
    enemy/neutral target with the best production/distance ratio and
    launch enough to win the capture, sized by max(aggressive=0.7×src,
    capture_size). Skips if the source can't afford the capture
    (defenders+production_during_flight+1 > src.ships), avoiding the
    bouncing-fleet failure mode where 0.7×src.ships < defenders.
    """
    player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
    planets = obs.get("planets") if isinstance(obs, dict) else getattr(obs, "planets", None)
    if not planets:
        return []
    targets = [p for p in planets if p[1] != player]
    moves: list = []
    for src in planets:
        if src[1] != player or src[5] < 10:
            continue
        best = None
        best_score = -1.0
        sx = src[2]; sy = src[3]
        for t in targets:
            if t[0] == src[0]:
                continue
            dx = t[2] - sx; dy = t[3] - sy
            d = math.sqrt(dx * dx + dy * dy)
            if d < 1e-6:
                continue
            score = float(t[6]) / (d + 1.0)
            if score > best_score:
                best_score = score
                best = t
        if best is None:
            continue
        # Capture-size estimate: predict defenders at straight-line ETA
        # for an aggressive-sized fleet, only launch if affordable.
        # Straight-line aim/eta — adequate for static targets; orbital
        # targets misaim but the rollout simulator catches the miss.
        budget = int(src[5])
        agg_ships = max(5, int(budget * 0.7))
        if agg_ships > budget:
            agg_ships = budget
        spd = _fleet_speed(agg_ships)
        if spd <= 0:
            continue
        dx = best[2] - sx; dy = best[3] - sy
        d = math.sqrt(dx * dx + dy * dy)
        flight = max(0.0, d - float(src[4]) - float(best[4]) - 0.1)
        eta = max(1, int(math.ceil(flight / spd)))
        # Production accrues only for OWNED planets (env rule:
        # orbit_wars.py:511-514 — neutrals stay at their current count).
        # Treating neutrals as accreting was the bug that made lite_greedy
        # skip capturable openings (e.g. 13-defender prod=1 neutral at
        # d=12 looked like 19 defenders at eta=6, so the policy idled
        # in opp_traj rollouts). Real opps grab near targets at step 4
        # and snowball.
        if int(best[1]) == -1:
            defenders_at_eta = float(best[5])
        else:
            defenders_at_eta = float(best[5]) + float(best[6]) * eta
        needed = int(math.ceil(defenders_at_eta)) + 1
        if needed > budget:
            continue  # can't afford the capture — skip, don't bounce
        ships = max(agg_ships, needed)
        if ships > budget:
            ships = budget
        if ships < 5:
            continue
        angle = math.atan2(best[3] - sy, best[2] - sx)
        moves.append([src[0], angle, ships])
    return moves


# ---------------------------------------------------------------------------
# ME-side defensive policy for the chooser's rollout (bug #14 fix, 2026-05-18)
# ---------------------------------------------------------------------------
#
# The chooser's rollout in agents/baseline/chooser_trajectory.py drives
# opp seats with `lite_greedy_policy` every tick but leaves ME idle
# (except for the candidate injection at `wait_N`). This asymmetry
# under-rates candidates that look attractive but expose our sources to
# counter-attack (rollout shows opp exploiting, we don't defend) and
# over-rates captures whose leaf-state ownership wouldn't survive opp's
# counter past the rollout horizon. Catalog: audit/2026-05-18-bug-
# catalog.md#14.
#
# Option 1 — cheap mirror with `lite_greedy_policy` for ME — failed
# (commit 5f22ea8): lite_greedy is too attack-biased, so the rollout's
# defense-baseline path emitted ATTACK launches from the would-be
# reinforcer planet and the threatened planet fell anyway.
#
# Option 5 (this function): PURELY DEFENSIVE policy for ME. Scan
# inbound enemy fleets, find under-defended owned planets, emit a
# reinforce launch from the nearest viable sister planet. Never emit
# attacks. The chooser's actual attack moves are made on its own turn;
# the rollout's job is to model opp's reaction to OUR move, which
# implies us defending what opp threatens — not us attacking again.
def me_defensive_action(obs: Any, me: int) -> list:
    """Purely-defensive obs-only policy for ME in the rollout.

    Returns env-format launches [[src_id, angle, ships], ...]. Same
    call shape as `lite_greedy_policy`. Stateless. No `WorldModel`
    build (would add 3-5 ms per tick × per candidate × baseline and
    blow the wallclock budget); uses only `fleet_target_planet` +
    arithmetic.

    Algorithm:
    1. Walk obs.fleets; attribute each enemy fleet to a MY planet via
       `lib.world_model.fleet_target_planet` (bug-#11-aware ray-cast).
       Bucket into `{my_pid: [(eta, ships), ...]}`.
    2. For each threatened MY planet P, sum threat force inside the
       bug-#12 window: `threat_force = sum(s for (e, s) in inbound[P]
       if e <= earliest_eta + WAVE_LOOKAHEAD)`. Skip if natural
       production covers it (`P.ships + P.prod × earliest_eta >=
       threat_force + 1`).
    3. Find nearest viable reinforcer Q: own, not P, not in
       `used_srcs`, reinforce-eta < earliest_eta with the sized fleet.
    4. Size: `ships = max(MIN_FLEET_SIZE, ceil(shortfall) +
       SAFETY_MARGIN)`, clamped at `Q.ships`.
    5. Aim: `lib.aim.aim_orbiting` for orbiting P, else `atan2`.
    6. Emit `[int(Q.id), float(angle), int(ships)]`; mark Q used.
    """
    # Local imports to keep the module's top-level fast (this function
    # is on the rollout hot path and must not pay an import cost on
    # first call inside the rollout).
    from kaggle_environments.envs.orbit_wars.orbit_wars import (
        Fleet, Planet,
    )

    MIN_FLEET_SIZE = 2
    SAFETY_MARGIN = 1

    planets_raw = (
        obs.get("planets") if isinstance(obs, dict)
        else getattr(obs, "planets", None)
    )
    if not planets_raw:
        return []
    fleets_raw = (
        obs.get("fleets", []) if isinstance(obs, dict)
        else getattr(obs, "fleets", [])
    )
    if not fleets_raw:
        return []
    omega = float(
        obs.get("angular_velocity", 0.0) if isinstance(obs, dict)
        else getattr(obs, "angular_velocity", 0.0) or 0.0
    )

    planets = [Planet(*p) for p in planets_raw]
    fleets = [Fleet(*f) for f in fleets_raw]
    my_planets_by_id = {int(p.id): p for p in planets if int(p.owner) == me}
    if not my_planets_by_id:
        return []

    # 1. Attribute fleets to MY planets. Enemy fleets become threats;
    # friendly fleets become inbound reinforcements that count toward
    # `garrison_at_eta`. The friendly-counting is the critical
    # idempotency property: without it the stateless policy emits a
    # NEW reinforce every rollout tick because each tick re-evaluates
    # the SAME threat against the SAME garrison without crediting the
    # already-launched reinforce. By tick N we've stacked N redundant
    # reinforces, draining the sister and bloating the fleet count
    # (which slows fs_step). Counting friendlies makes the policy
    # converge after one emit per real threat.
    inbound_enemy: dict[int, list[tuple[int, int]]] = {}
    inbound_friendly_ships: dict[int, int] = {}
    for f in fleets:
        target, eta = fleet_target_planet(f, planets, omega)
        if target is None or int(target.owner) != me:
            continue
        if int(f.owner) == me:
            inbound_friendly_ships[int(target.id)] = (
                inbound_friendly_ships.get(int(target.id), 0)
                + int(f.ships)
            )
        else:
            inbound_enemy.setdefault(int(target.id), []).append(
                (int(eta), int(f.ships))
            )
    if not inbound_enemy:
        return []

    moves: list = []
    used_srcs: set[int] = set()

    # Process threats in eta-order so the most-urgent gets dibs on the
    # nearest reinforcer.
    threat_list = sorted(
        inbound_enemy.items(),
        key=lambda kv: min(e for (e, _s) in kv[1]),
    )
    for pid, waves in threat_list:
        p_target = my_planets_by_id[pid]
        earliest_eta = min(e for (e, _s) in waves)
        threat_force = sum(
            s for (e, s) in waves if e <= earliest_eta + WAVE_LOOKAHEAD
        )
        if threat_force <= 0:
            continue
        garrison_at_eta = (
            float(p_target.ships)
            + float(p_target.production) * float(earliest_eta)
            + float(inbound_friendly_ships.get(pid, 0))
        )
        if garrison_at_eta >= float(threat_force) + 1.0:
            continue  # natural production + in-flight reinforces cover it

        shortfall = float(threat_force) + 1.0 - garrison_at_eta
        # Find nearest viable reinforcer.
        best: tuple | None = None
        best_dist_sq = float("inf")
        for q in planets:
            if int(q.owner) != me or int(q.id) == pid:
                continue
            if int(q.id) in used_srcs:
                continue
            dx = float(p_target.x) - float(q.x)
            dy = float(p_target.y) - float(q.y)
            dist_sq = dx * dx + dy * dy
            if dist_sq >= best_dist_sq:
                continue
            # Single-iteration chicken-and-egg fix: assume worst-case
            # reinforce-eta = earliest_eta - 1, compute ships, verify
            # the resulting eta is still < earliest_eta.
            worst_eta = max(1, int(earliest_eta) - 1)
            ships_guess = max(
                MIN_FLEET_SIZE,
                int(math.ceil(shortfall)) + SAFETY_MARGIN,
            )
            if ships_guess > int(q.ships):
                ships_guess = int(q.ships)
            if ships_guess < MIN_FLEET_SIZE:
                continue
            spd = _fleet_speed(ships_guess)
            if spd <= 0:
                continue
            d = math.sqrt(dist_sq)
            flight = max(
                0.0, d - float(q.radius) - float(p_target.radius) - 0.1
            )
            eta_q = max(1, int(math.ceil(flight / spd)))
            if eta_q >= int(earliest_eta):
                continue  # too slow
            best = (q, ships_guess, eta_q)
            best_dist_sq = dist_sq
        if best is None:
            continue

        q, ships, eta_q = best
        # Aim: orbital lead for orbiting targets, static atan2 otherwise.
        if is_orbiting(p_target):
            target_tuple = (
                int(p_target.id), int(p_target.owner),
                float(p_target.x), float(p_target.y),
                float(p_target.radius),
                int(p_target.ships), int(p_target.production),
            )
            aim_res = aim_orbiting(
                (float(q.x), float(q.y)),
                float(q.radius),
                target_tuple,
                float(p_target.radius),
                int(ships),
                float(omega),
            )
            if aim_res is None:
                continue
            angle = float(aim_res[0])
        else:
            angle = math.atan2(
                float(p_target.y) - float(q.y),
                float(p_target.x) - float(q.x),
            )

        moves.append([int(q.id), float(angle), int(ships)])
        used_srcs.add(int(q.id))

    return moves


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
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterable

# Env-var override for the wallclock budget used by every `choose_*` entry
# point. **Only consulted at the top of each chooser**, never inside the
# search loop, so the production path's `time.perf_counter()` watchdog
# is unchanged. Set by `scripts/bundle_agent.py::_parity_gate` to make
# source-vs-bundle parity tests deterministic: with the default 700 ms
# budget, a chooser may bail mid-candidate-list on system jitter, leaving
# argmax to pick over a different subset of candidates each run. Setting
# the budget effectively unbounded lets every candidate be scored, so
# the agent becomes a pure function of its inputs.
_WALLCLOCK_ENV_VAR = "ORBIT_WARS_PARITY_WALLCLOCK_MS"


def _effective_wallclock_ms(wallclock_ms: float) -> float:
    """Return `wallclock_ms` unless the parity-test env var is set, in
    which case use the env-var value. Invalid values fall back to the
    caller's number rather than crashing the agent."""
    override = os.environ.get(_WALLCLOCK_ENV_VAR)
    if not override:
        return wallclock_ms
    try:
        return float(override)
    except ValueError:
        return wallclock_ms

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
        propose_opening_missions(world, model)
        + propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )
    if include_recapture:
        missions = missions + propose_recapture_missions(world, model)
    # Mission Renaissance: opening + drain + gang_up proposers each
    # return [] when their USE_*_MISSION flag is 0 (default), so v7
    # parity is preserved until the A/B flips a flag.
    missions = (
        missions
        + propose_opening_missions(world, model)
        + propose_drain_missions(world, model)
        + propose_gang_up_missions(world, model)
    )
    chosen = settle_plan(missions, world, model)
    return chosen


# ---------------------------------------------------------------------------
# Enumerators (one per mode)
# ---------------------------------------------------------------------------


@contextmanager
def _bind_shared_world_model(obs_list, model):
    """Temporarily attach ``model`` to each observation in ``obs_list`` as
    the ``_shared_world_model`` attribute so mirror-style policies can
    skip the expensive ``WorldModel.from_world`` rebuild
    (`lib/opp_model.py:76, 112`). Exception-safe — the attribute is
    always removed on exit, even if a policy raises.

    Why this matters: the previous bare ``del obs._shared_world_model``
    cleanup ran only on the happy path. A raise inside the followup
    policy left the attribute on the cloned observation Struct, which
    is then garbage-collected — but in the parity-gate setting where
    the source and bundle agents are called back-to-back in the same
    process on the same input ``obs``, any leaked side-channel state on
    a Struct that participates in both calls is a parity risk. Encoding
    the lifetime as a context manager makes the invariant impossible
    to violate accidentally.
    """
    if model is None or not obs_list:
        yield
        return
    for obs in obs_list:
        obs._shared_world_model = model
    try:
        yield
    finally:
        for obs in obs_list:
            # Use try/except (rather than __dict__.pop) to mirror the
            # same access path that __setattr__ took; Struct's attribute
            # storage isn't guaranteed to be __dict__.
            try:
                del obs._shared_world_model
            except AttributeError:
                pass


def _enumerate_drop_one(incumbent_action: list[list]) -> list[list[list]]:
    """Incumbent + each drop-one variant. Floor for v7 framework lift."""
    if not incumbent_action:
        return [[]]
    cands: list[list[list]] = [list(incumbent_action)]
    for i in range(len(incumbent_action)):
        cands.append([m for j, m in enumerate(incumbent_action) if j != i])
    return cands


def _enumerate_add_one(
    world: World, model: WorldModel,
    incumbent_intents: list[Intent], incumbent_action: list[list],
    obs: Any,
) -> list[list[list]]:
    """Extend the incumbent by one launch from a source the proposer
    didn't pick. Generates at most one variant per *idle* owned source.

    Why this complements drop-one: `_enumerate_drop_one` is monotonically
    narrower than the incumbent — it can only suppress launches. The
    chooser cannot reach moves the proposer skipped. Per the diagnostic
    audit (`audit/2026-05-13-v7-0-loss-modes.md`) and the H11 / depth-2
    failures, the action-space width is plausibly the binding constraint.

    For each owned source NOT already in the incumbent, find that
    source's top-scored snipe-or-reinforce mission and append it to the
    incumbent action. The result is `incumbent + one_more_launch`. The
    caller composes this with drop-one for a union enumerator.
    """
    if not world.planets_by_id:
        return [list(incumbent_action)]
    incumbent_src_ids = {int(intent.src_id) for intent in incumbent_intents}

    # Score per-source missions once; pick each idle source's top.
    candidate_missions = (
        propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )
    by_src: dict[int, list[Mission]] = {}
    for m in candidate_missions:
        src_id = int(m.src_id)
        if src_id in incumbent_src_ids:
            continue
        by_src.setdefault(src_id, []).append(m)

    cands: list[list[list]] = [list(incumbent_action)]
    seen_keys = {_action_key(incumbent_action)}
    for src_id, ranked in by_src.items():
        if not ranked:
            continue
        ranked.sort(key=lambda m: -m.score)
        top = ranked[0]
        new_intents = list(incumbent_intents) + [top.to_intent()]
        new_action = _action_from_intents(new_intents, obs, model)
        if not new_action:
            continue
        k = _action_key(new_action)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        cands.append(new_action)
    return cands


def _enumerate_split_source(
    world: World, model: WorldModel,
    incumbent_intents: list[Intent], incumbent_action: list[list],
    obs: Any,
    *,
    min_leftover_ships: int = 5,
) -> list[list[list]]:
    """Multi-launch from one source — emits incumbent + extra launch from
    each source that has leftover garrison after its incumbent launch.

    Different from `_enumerate_add_one`: that one extends from IDLE
    sources (planets with no incumbent launch). This one extends from
    ACTIVE sources (planets with an incumbent launch that didn't drain
    them). The env supports the same `src_id` appearing multiple times
    in an action — `process_moves` in the interpreter decrements the
    source's garrison per launch, so two launches from one source share
    a budget.

    Motivation: the loss-mode diagnostic
    (`audit/2026-05-13-v7-0-loss-modes.md`) showed top-10 has mean
    garrison-at-launch ~11, midpack ~22. v7_0_drop_one with
    `aggressive=True` launches 0.7 × ships and leaves 0.3 × ships idle.
    For a source with 30 ships that's ~9 leftover — enough for a small
    second launch toward a runner-up target.

    Algorithm per source `src` in the incumbent:
      1. Compute `leftover = src.ships − incumbent_launch.ships`.
      2. If `leftover < min_leftover_ships`, skip (not worth a second
         launch).
      3. Find this source's TOP runner-up snipe target whose required
         `ship_count <= leftover`, excluding the incumbent's target.
      4. Append that intent to the incumbent intents and emit the
         realised action.

    At most one split variant per source is emitted. Combined with
    drop-one via `_enumerate_drop_or_split` for argmax over the union.
    """
    if not world.planets_by_id or not incumbent_intents:
        return [list(incumbent_action)]
    incumbent_by_src = {int(i.src_id): i for i in incumbent_intents}

    # Pre-compute all snipe missions once and bucket by source.
    snipe_missions = propose_snipe_missions(
        world, model, aggressive=True,
    )
    by_src: dict[int, list[Mission]] = {}
    for m in snipe_missions:
        by_src.setdefault(int(m.src_id), []).append(m)

    cands: list[list[list]] = [list(incumbent_action)]
    seen_keys = {_action_key(incumbent_action)}
    for src_id, intent in incumbent_by_src.items():
        src = world.planets_by_id.get(src_id)
        if src is None:
            continue
        leftover = int(src.ships) - int(intent.ships)
        if leftover < min_leftover_ships:
            continue
        # Pick the top-scored runner-up target this source could afford.
        ranked = sorted(by_src.get(src_id, []), key=lambda m: -m.score)
        runner_up: Mission | None = None
        cur_target = int(intent.target_id)
        for m in ranked:
            if int(m.target_id) == cur_target:
                continue
            if int(m.ships) > leftover:
                continue
            runner_up = m
            break
        if runner_up is None:
            continue
        new_intents = list(incumbent_intents) + [runner_up.to_intent()]
        new_action = _action_from_intents(new_intents, obs, model)
        if not new_action:
            continue
        k = _action_key(new_action)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        cands.append(new_action)
    return cands


def _enumerate_drop_or_split(
    world: World, model: WorldModel,
    incumbent_intents: list[Intent], incumbent_action: list[list],
    obs: Any,
) -> list[list[list]]:
    """Union of drop-one and split-source. Strictly wider than either."""
    cands: list[list[list]] = [list(incumbent_action)]
    seen_keys = {_action_key(incumbent_action)}
    for cand in _enumerate_drop_one(incumbent_action):
        k = _action_key(cand)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        cands.append(cand)
    for cand in _enumerate_split_source(
        world, model, incumbent_intents, incumbent_action, obs,
    ):
        k = _action_key(cand)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        cands.append(cand)
    return cands


def _enumerate_drop_or_add_one(
    world: World, model: WorldModel,
    incumbent_intents: list[Intent], incumbent_action: list[list],
    obs: Any,
) -> list[list[list]]:
    """Union of `_enumerate_drop_one` and `_enumerate_add_one`. Strictly
    wider than either alone: incumbent + (drop each launch) + (add one
    launch from each idle source). Deduplicates exact matches.
    """
    cands: list[list[list]] = [list(incumbent_action)]
    seen_keys = {_action_key(incumbent_action)}
    for cand in _enumerate_drop_one(incumbent_action):
        k = _action_key(cand)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        cands.append(cand)
    for cand in _enumerate_add_one(
        world, model, incumbent_intents, incumbent_action, obs,
    ):
        k = _action_key(cand)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        cands.append(cand)
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
    if enumerator_mode == "add_one":
        return _enumerate_add_one(
            world, model, incumbent_intents, incumbent_action, obs,
        )
    if enumerator_mode == "drop_or_add_one":
        return _enumerate_drop_or_add_one(
            world, model, incumbent_intents, incumbent_action, obs,
        )
    if enumerator_mode == "split_source":
        return _enumerate_split_source(
            world, model, incumbent_intents, incumbent_action, obs,
        )
    if enumerator_mode == "drop_or_split":
        return _enumerate_drop_or_split(
            world, model, incumbent_intents, incumbent_action, obs,
        )
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
    followup_policy: Callable | None = None,
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

    `followup_policy(observation) -> list` is the policy applied to
    BOTH seats for the K-1 follow-up steps after our forced action.
    Defaults to `top_tier_mirror_policy` (v3.5.1 pipeline, ~10 ms /
    call). Pass `lite_greedy_policy` (~1 ms / call) when the rollout
    only needs to estimate trajectory direction and bit-fidelity to
    v3.5.1 isn't required — typically the case for wider/deeper
    multi-candidate search.
    """
    if snap.num_seats != 2:
        raise ValueError(f"v7 score_candidate is 2P only (got {snap.num_seats})")
    clone = fs_clone(snap)
    opp_id = 1 - my_id

    opp_policy = make_opp_policy(opp_tier)
    if followup_policy is None:
        followup_policy = top_tier_mirror_policy

    # First step: forced action for us; opp plays its policy.
    a_opp = opp_policy(clone.state[opp_id].observation)
    actions = [None, None]
    actions[my_id] = action
    actions[opp_id] = a_opp
    if not clone.done:
        clone = fs_step(clone, actions, in_place=True)

    # Remaining K-1 steps: both seats play follow-up policy.
    # OPTIMIZATION (Phase 3c): both seats see the same planets/fleets/
    # comets/angular_velocity, so `WorldModel.from_world` produces the
    # SAME object regardless of which seat's obs we built it from. We
    # build it once per step and stash it on both seats' observations
    # via the `_shared_world_model` attribute; mirror-style policies
    # (lib.opp_model.top_tier_mirror_policy / mirror_self_policy) check
    # for this attribute and skip the expensive rebuild (~3.8 ms each).
    # Net savings: ~3.8 ms per rollout step. Bit-exact parity preserved
    # because the same model object is used either way.
    for _ in range(max(0, K - 1)):
        if clone.done:
            break
        obs0 = clone.state[0].observation
        obs1 = clone.state[1].observation
        # Build shared World/Model once. Cheap to construct World per
        # seat (it's a tiny dataclass); the expensive part is WorldModel.
        shared_world = World.from_obs(obs0)
        shared_model = (
            WorldModel.from_world(shared_world)
            if shared_world.planets_by_id else None
        )
        with _bind_shared_world_model((obs0, obs1), shared_model):
            a0 = followup_policy(obs0)
            a1 = followup_policy(obs1)
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
    wallclock_ms = _effective_wallclock_ms(wallclock_ms)
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
    wallclock_ms = _effective_wallclock_ms(wallclock_ms)
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
    wallclock_ms = _effective_wallclock_ms(wallclock_ms)
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


def _score_after_opp_response(
    snap_i: Snapshot,
    opp_act: list[list],
    *,
    my_id: int,
    opp_id: int,
    K_tail: int,
    value_fn: Callable | None = None,
) -> float:
    """Score after a forced opp response on turn 2.

    From `snap_i` (a snapshot that has already been advanced one turn by
    our forced action paired with the opp's incumbent), force the opp's
    response action `opp_act` on this turn (we pass — we've committed),
    then run `K_tail` mirror-mirror follow-up steps. Score from `my_id`'s
    POV via `value_fn` (default `delta_us_minus_them`).

    Used by `choose_depth2` to fill the maximin payoff matrix.
    """
    clone = fs_clone(snap_i)
    if not clone.done:
        actions: list[Any] = [None, None]
        actions[my_id] = []  # we pass
        actions[opp_id] = opp_act
        clone = fs_step(clone, actions, in_place=True)

    for _ in range(max(0, K_tail)):
        if clone.done:
            break
        a0 = top_tier_mirror_policy(clone.state[0].observation)
        a1 = top_tier_mirror_policy(clone.state[1].observation)
        clone = fs_step(clone, [a0, a1], in_place=True)

    if value_fn is None:
        return delta_us_minus_them(clone, my_id)
    return value_fn(clone.state[my_id].observation, my_id)


def choose_depth2(
    obs: Any,
    configuration: Any = None,
    *,
    K: int = 6,
    wallclock_ms: float = 700.0,
    my_id: int | None = None,
    include_recapture: bool = True,
    value_fn: Callable | None = None,
    max_our_candidates: int = 8,
    max_opp_candidates: int = 4,
) -> list[list]:
    """v7 depth-2 maximin (action-SEQUENCE depth-2, not joint-1-ply).

    Algorithm:
    1. Enumerate our drop-one candidate set (≤ `max_our_candidates`).
    2. For each our candidate i:
       a. Step the snapshot one turn with [our_i, opp_initial_incumbent].
       b. From the post-step state, recompute the opp's incumbent and
          enumerate the opp's drop-one set (≤ `max_opp_candidates`).
       c. For each opp candidate j, force it on turn 2 (we pass), then
          rollout `K-2` mirror-mirror steps. Record payoff[i][j].
    3. Maximin: argmax_i min_j payoff[i][j]. Tie → row 0 (incumbent).

    Budget shape (defaults): 8 × 4 × ~15 ms = ~500 ms wall; under 700 ms
    actTimeout. Watchdog: bail outer rows past 0.5 × wallclock_ms (row 0
    always evaluated in full first), bail inner cells past wallclock_ms.

    4P fallback: return the incumbent (depth-2 minimax is 2P-only —
    Nash maximin doesn't generalise cleanly to n > 2).
    """
    wallclock_ms = _effective_wallclock_ms(wallclock_ms)
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

    # 4P fallback — depth-2 maximin is 2P only.
    if _infer_num_seats(world) != 2:
        return incumbent_action

    our_C = _enumerate_drop_one(incumbent_action)
    if max_our_candidates and len(our_C) > max_our_candidates:
        our_C = our_C[:max_our_candidates]
    if len(our_C) <= 1:
        return incumbent_action

    snap = fs_from_obs(obs, configuration, episode_seed=0, num_seats=2)
    opp_id = 1 - my_id

    # Opp plays its v3.5.1 incumbent on turn 1 against every one of our
    # candidates (same opp action across all rows — keeps the matrix
    # comparable to v7_0_drop_one's evaluation).
    opp_initial_action = _opp_incumbent_action(world, obs, opp_id)

    K_tail = max(0, K - 2)
    N = len(our_C)
    P: list[list[float]] = [[] for _ in range(N)]

    for i in range(N):
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if i > 0 and elapsed_ms > 0.5 * wallclock_ms:
            # Row 0 (incumbent) is always evaluated in full first; later
            # rows bail if half the budget is gone.
            P[i] = []
            continue

        try:
            snap_i = fs_clone(snap)
            if not snap_i.done:
                actions: list[Any] = [None, None]
                actions[my_id] = our_C[i]
                actions[opp_id] = opp_initial_action
                snap_i = fs_step(snap_i, actions, in_place=True)
        except Exception:
            P[i] = []
            continue

        if snap_i.done:
            # Game ended in turn 1 — score the leaf directly. Same
            # payoff for any opp_C[j] since there's no turn 2.
            try:
                terminal = (
                    delta_us_minus_them(snap_i, my_id)
                    if value_fn is None
                    else value_fn(snap_i.state[my_id].observation, my_id)
                )
            except Exception:
                terminal = float("-inf")
            P[i] = [terminal]
            continue

        # Recompute opp's incumbent from the post-turn-1 state.
        opp_obs_after = snap_i.state[opp_id].observation
        try:
            opp_world = World.from_obs(opp_obs_after)
            opp_model = WorldModel.from_world(opp_world)
            opp_inc_intents = _build_incumbent_intents(
                opp_world, opp_model, include_recapture=include_recapture,
            )
            opp_inc_action = _action_from_intents(
                opp_inc_intents, opp_obs_after, opp_model,
            )
            opp_C = _enumerate_drop_one(opp_inc_action)
            if max_opp_candidates and len(opp_C) > max_opp_candidates:
                opp_C = opp_C[:max_opp_candidates]
        except Exception:
            opp_C = [[]]
        if not opp_C:
            opp_C = [[]]

        row_scores: list[float] = []
        for j, opp_act in enumerate(opp_C):
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            if elapsed_ms > wallclock_ms:
                break
            try:
                payoff = _score_after_opp_response(
                    snap_i, opp_act,
                    my_id=my_id, opp_id=opp_id, K_tail=K_tail,
                    value_fn=value_fn,
                )
            except Exception:
                payoff = float("-inf")
            row_scores.append(payoff)

        P[i] = row_scores

    # Maximin over the evaluated rows.
    NEG_INF = float("-inf")
    best_i = 0
    best_worst = NEG_INF
    for i in range(N):
        if not P[i]:
            continue
        worst = min(P[i])
        if worst > best_worst:
            best_worst = worst
            best_i = i

    return our_C[best_i] if best_worst > NEG_INF else incumbent_action


def choose_archetype_minregret(
    obs: Any,
    configuration: Any = None,
    *,
    K: int = 6,
    wallclock_ms: float = 700.0,
    my_id: int | None = None,
    use_min_regret: bool = True,
    include_recapture: bool = True,
    value_fn: Callable | None = None,
    max_our_candidates: int = 8,
) -> list[list]:
    """Depth-2 chooser using hand-crafted opp archetypes (not v3.5.1
    drop-ones) as the opp candidate set, with either min-regret or
    maximin row aggregation.

    Why this exists: the prior `choose_depth2` derives `opp_C` from
    v3.5.1's incumbent via drop-one. Both v7_1 and v7_2 failed in
    scalar A/B against v7_0_drop_one — strong evidence the v3.5.1
    opp assumption is biased (the live ladder is heterogeneous). The
    archetype set (`lib.missions.opp_archetypes`) gives 5 distinct
    opp threat patterns: no-launch / v3.5.1 / counter-reinforce /
    counter-snipe / cross-attack. Min-regret aggregation picks our
    action with the smallest worst-case gap from its best response
    over any of those archetypes — robust under opp uncertainty.

    `use_min_regret=True` (default) uses min-regret aggregation:
      regret[i] = max_j (max_k P[k][j] - P[i][j])
      return argmin_i regret[i]
    `use_min_regret=False` falls back to maximin:
      return argmax_i min_j P[i][j]

    4P fallback: depth-2 game-theory is 2P only; 4P returns incumbent.
    """
    wallclock_ms = _effective_wallclock_ms(wallclock_ms)
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

    if _infer_num_seats(world) != 2:
        return incumbent_action

    our_C = _enumerate_drop_one(incumbent_action)
    if max_our_candidates and len(our_C) > max_our_candidates:
        our_C = our_C[:max_our_candidates]
    if len(our_C) <= 1:
        return incumbent_action

    snap = fs_from_obs(obs, configuration, episode_seed=0, num_seats=2)
    opp_id = 1 - my_id

    # Opp plays their natural incumbent on turn 1. Same as choose_depth2.
    opp_initial_action = _opp_incumbent_action(world, obs, opp_id)

    K_tail = max(0, K - 2)
    N = len(our_C)
    P: list[list[float]] = [[] for _ in range(N)]

    # Lazy import — keeps the existing lib graph clean.

    for i in range(N):
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if i > 0 and elapsed_ms > 0.5 * wallclock_ms:
            P[i] = []
            continue

        # Forced turn 1.
        try:
            snap_i = fs_clone(snap)
            if not snap_i.done:
                actions: list[Any] = [None, None]
                actions[my_id] = our_C[i]
                actions[opp_id] = opp_initial_action
                snap_i = fs_step(snap_i, actions, in_place=True)
        except Exception:
            P[i] = []
            continue

        if snap_i.done:
            try:
                terminal = (
                    delta_us_minus_them(snap_i, my_id)
                    if value_fn is None
                    else value_fn(snap_i.state[my_id].observation, my_id)
                )
            except Exception:
                terminal = float("-inf")
            P[i] = [terminal]
            continue

        # Build opp archetypes from the POST-turn-1 state. Counter-
        # reinforce uses the OPP's intents that would best counter our
        # turn-1 launches, so we pass the incumbent intents (we already
        # committed to a subset of these on this candidate row).
        opp_obs_after = opp_pov_obs(snap_i.state[opp_id].observation, opp_id)
        try:
            opp_archetypes = build_opp_archetypes(
                opp_obs_after, our_intents=incumbent_intents,
            )
        except Exception:
            opp_archetypes = [[]]
        if not opp_archetypes:
            opp_archetypes = [[]]

        row_scores: list[float] = []
        for j, opp_act in enumerate(opp_archetypes):
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            if elapsed_ms > wallclock_ms:
                break
            try:
                payoff = _score_after_opp_response(
                    snap_i, opp_act,
                    my_id=my_id, opp_id=opp_id, K_tail=K_tail,
                    value_fn=value_fn,
                )
            except Exception:
                payoff = float("-inf")
            row_scores.append(payoff)

        P[i] = row_scores

    # Aggregate the payoff matrix.
    NEG_INF = float("-inf")

    if not use_min_regret:
        # Maximin: argmax_i min_j P[i][j].
        best_i = 0
        best_worst = NEG_INF
        for i in range(N):
            if not P[i]:
                continue
            worst = min(P[i])
            if worst > best_worst:
                best_worst = worst
                best_i = i
        return our_C[best_i] if best_worst > NEG_INF else incumbent_action

    # Min-regret: column-wise best response (only over fully-scored rows
    # so a budget-bailed row doesn't poison the column best), then pick
    # the row whose worst regret is smallest.
    M = max((len(row) for row in P), default=0)
    if M == 0:
        return incumbent_action
    col_best: list[float] = []
    for j in range(M):
        vals = [P[i][j] for i in range(N) if j < len(P[i])]
        col_best.append(max(vals) if vals else NEG_INF)

    best_i = 0
    best_regret = float("inf")
    for i in range(N):
        if not P[i] or len(P[i]) < M:
            # Skip rows that didn't complete every column — would give
            # pessimistic regret. The incumbent (i=0) is fully evaluated
            # by the budget-first guarantee above, so the safe default
            # of `best_i = 0` survives.
            continue
        regret_i = max(col_best[j] - P[i][j] for j in range(M))
        # Tie-break: lower index wins (=row 0 incumbent).
        if regret_i < best_regret:
            best_regret = regret_i
            best_i = i

    return our_C[best_i] if best_regret < float("inf") else incumbent_action


def choose_archetype_minregret_with_4p(
    obs: Any,
    configuration: Any = None,
    *,
    K: int = 6,
    K_4p: int = 8,
    wallclock_ms: float = 700.0,
    use_min_regret: bool = True,
    include_recapture: bool = True,
    value_fn: Callable | None = None,
) -> list[list]:
    """v7.3 entry that auto-routes 2P → `choose_archetype_minregret`,
    4P → `choose_4p` (no maximin / regret in 4P)."""
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    n_seats = _infer_num_seats(world)
    if n_seats == 2:
        return choose_archetype_minregret(
            obs, configuration,
            K=K, wallclock_ms=wallclock_ms,
            use_min_regret=use_min_regret,
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
    model = WorldModel.from_world(world)
    intents = _build_incumbent_intents(
        world, model, include_recapture=include_recapture,
    )
    return _action_from_intents(intents, obs, model)


def choose_depth2_with_4p(
    obs: Any,
    configuration: Any = None,
    *,
    K_2p: int = 6,
    K_4p: int = 8,
    wallclock_ms: float = 700.0,
    include_recapture: bool = True,
    value_fn: Callable | None = None,
) -> list[list]:
    """v7 depth-2 entry that auto-routes 2P → choose_depth2, 4P → choose_4p.

    Use this as the agent's `agent(obs, configuration)` entry point when
    bundling a depth-2 variant. 4P games fall through to the drop-one
    chooser (no maximin guarantee at n > 2).
    """
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    n_seats = _infer_num_seats(world)
    if n_seats == 2:
        return choose_depth2(
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
    intents = _build_incumbent_intents(
        world, model, include_recapture=include_recapture,
    )
    return _action_from_intents(intents, obs, model)


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
    opp_tiers: list[int] | None = None,
    value_fn: Callable | None = None,
    followup_policy: Callable | None = None,
) -> list[list]:
    """End-to-end: build incumbent, enumerate, score with watchdog,
    return argmax. Always returns the incumbent if no candidate
    scores strictly higher (parity floor).

    `my_id` defaults to `obs.player`. `configuration` is forwarded to
    `fs_from_obs`; if `None`, defaults are used (live ladder
    scrubs the episode seed anyway).

    `opp_tiers` is the opponent-policy pool used to score each
    candidate. Defaults to `[1]` (Tier-1 v3.5.1 mirror, v7_0 default).
    With multiple tiers, the chooser uses MAXIMIN — pick the candidate
    whose MIN-over-tiers score is highest. Robust to opp-policy
    uncertainty across the live ladder.

    `value_fn(observation, my_id) -> float` is the leaf-state scoring
    head. Defaults to `delta_us_minus_them` (Phase-2-validated baseline).
    Phase 3c uses a composite (ship_delta + denial + survivor) blend.
    """
    wallclock_ms = _effective_wallclock_ms(wallclock_ms)
    t_start = time.perf_counter()
    if opp_tiers is None:
        opp_tiers = [1]

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
        # Maximin: score this candidate against every opp tier in the
        # pool, take the WORST (min) score. Picking the candidate that
        # maximises this worst-case is the maximin / game-theoretic
        # robust choice against opp-policy uncertainty. For a single
        # tier (the v7_0 default) min-of-one is just the score itself.
        per_tier = []
        for tier in opp_tiers:
            s = score_candidate(
                snap, cand, my_id=my_id, K=K,
                opp_tier=tier, value_fn=value_fn,
                followup_policy=followup_policy,
            )
            per_tier.append(s)
        score = min(per_tier)
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

# Single-line imports below: the submission bundler's per-line
# import-stripping regex would leak continuation lines from a parenthesised
# multi-line import as indented orphans (IndentationError at runtime).
# Friction tag: `bundler-modular-agent-namespace-access-breaks-bundle`
# documented in agents/baseline/main.py.


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


# ---------------------------------------------------------------------------
# composite_capture_value — anti-waste + capture-aware (v7.4)
# ---------------------------------------------------------------------------


# Coefficients tuned so the three terms are comparable in scale on a
# typical mid-game board (ship-delta in the ~10-50 range, capture bonus
# ~0.05 × 3 × 300 = 45 per high-value capture, waste penalty ~0.5 × ships).
CAPTURE_REWARD_WEIGHT: float = 0.05
WASTE_PENALTY_WEIGHT: float = 0.5
EPISODE_STEPS_TOTAL: int = 500

# Discount factor for the per-planet production-PV term. Matches
# `agents/baseline/value.favor`'s default gamma so composite's
# ownership-credit scales consistently with favor across the 2P
# composite / 4P A2-favor split in `favor_hybrid`.
PRODUCTION_PV_GAMMA: float = 0.99

# Diagnostic toggle for the production-PV term in `composite_capture_value`.
# Bug #15 fix v2 (2026-05-18 PM) shipped this term default-ON; subsequent
# A/B vs the pre-fix bundle settled at 39.6% (n=96, Wlo=0.304, FAIL).
# Bug #14 option 5 (smart reactive defense in candidate rollouts) was the
# hypothesised cure; it ALSO failed at 39.6% — the hypothesis is fully
# falsified. The convergent failure means the PV term itself over-credits
# captures: chooser was calibrated WITHOUT PV; adding ~100 units per
# captured planet at leaf uniformly inflates candidate scores → over-
# emission → drained sources → losses. Disabling PV restores the chooser's
# pre-#15 calibration (~50% vs bundle). Cost: sanity oracle (`test_oracle
# _sanity_trivial_capture`) reverts to xfail — that property is real but
# the cost-benefit tilts to "disable and revisit with chooser-gate
# recalibration in a future session". See
# audit/2026-05-18-postmortem-bug-15-v2-and-bug-14-option-5.md and
# knowledge-base/thoughts/2026-05-18-PV-term-recalibration-debt.md.
# Default OFF as of 2026-05-18 PM session wrap. Set
# `COMPOSITE_PRODUCTION_PV=1` to re-enable for A/Bs.
import os as _os
_COMPOSITE_PV_ENABLED = _os.environ.get("COMPOSITE_PRODUCTION_PV", "0") != "0"


def composite_capture_value(
    obs: Any, my_id: int,
    *,
    horizon: int = DEFAULT_HORIZON,
    capture_weight: float = CAPTURE_REWARD_WEIGHT,
    waste_weight: float = WASTE_PENALTY_WEIGHT,
) -> float:
    """Ship-delta + production-PV + per-fleet waste penalty.

    Base = `(my_ships − opp_ships) + (my_prod − opp_prod) × pv`. The PV
    term values planet ownership beyond the leaf horizon so captures
    register at the leaf even after the capturing fleet has arrived
    (without it, ship counts net out symmetrically and equal-production
    captures score Δ = 0 vs idle). This is bug #15's fix: a leaf state
    where we just captured opp's last planet now scores higher than
    the do-nothing baseline.

    For each of OUR in-flight fleets, the per-fleet WASTE PENALTY fires
    when the launch is structurally lost:
    - no planet on the trajectory (OOB);
    - trajectory crosses the sun (engine kills the fleet mid-flight);
    - target is a comet that expires before arrival;
    - predicted owner at ETA is NOT us (we bounce off a stronger
      defender, or multi-arrival combat goes the other way).

    There is NO per-fleet capture-credit term. The bug #15 fix v1
    (2026-05-18 AM) added a counterfactual per-fleet capture credit
    on top of the PV term, but the A/B ablation (n=64) showed the two
    terms double-credit the same capture and the chooser systematically
    over-emits (winrate 40.6% with both halves on, 46.9% with PV only,
    vs 50% baseline). v2 (this version) keeps PV-only — the PV term
    already credits the capture at the leaf via planet ownership.

    Set `COMPOSITE_PRODUCTION_PV=0` to disable the PV term for a clean
    A/B revert to pre-2026-05-18 behaviour (sanity oracle fails when
    PV is off, but the chooser's calibration matches the pre-bug-#15
    state used by submission `52754310`).
    """
    base = delta_us_minus_them_obs(obs, my_id)
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return base
    step_now = int(world.step)

    # Per-planet production-PV term. Mirrors `favor()`'s
    # `(my_prod - opp_prod) * pv_horizon`. Without this term the base
    # ship-delta is invariant to a capture of an equal-production
    # planet (both owners produce at the same rate over the rollout,
    # so net ships cancel out), which means a candidate that captures
    # opp's planet during the rollout scores Δ ≈ 0 vs idle even though
    # we won the planet's future production. Bug #15 root cause is two
    # things together: (a) the per-fleet credit was broken by a
    # chicken-and-egg in WorldModel prediction (see below), AND (b)
    # base lacked any term that values ownership beyond the leaf
    # horizon — so even with the per-fleet fix, post-arrival captures
    # (eta < rollout horizon) would still not register. Sanity oracle
    # `tests/test_planner_oracles.py::test_oracle_sanity_trivial_capture`
    # surfaced (b); the bug catalog at audit/2026-05-18-bug-catalog.md
    # documents (a). 2026-05-18 fix.
    if _COMPOSITE_PV_ENABLED:
        pv = pv_horizon(
            step_now, 0,
            gamma=PRODUCTION_PV_GAMMA,
            t_total=EPISODE_STEPS_TOTAL,
        )
        my_prod = 0.0
        opp_prod = 0.0
        for p in world.planets_by_id.values():
            owner = int(p.owner)
            if owner == my_id:
                my_prod += float(p.production)
            elif owner >= 0:
                opp_prod += float(p.production)
        base += (my_prod - opp_prod) * pv

    raw = world.obs_raw
    fleets_raw = (
        raw.get("fleets", []) if isinstance(raw, dict)
        else getattr(raw, "fleets", [])
    )
    if not fleets_raw:
        return base

    # Reuse the kaggle namedtuple so `fleet_target_planet` gets the same
    # type it expects.
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet  # noqa: E402
    fleets = [Fleet(*f) for f in fleets_raw]
    planets_list = list(world.planets_by_id.values())
    # Thread omega through to fleet_target_planet so orbiting-target
    # attribution works (bug #11 fix, 2026-05-18).
    omega = float(
        raw.get("angular_velocity", 0.0) if isinstance(raw, dict)
        else getattr(raw, "angular_velocity", 0.0) or 0.0
    )

    # Pre-pass: compute each of OUR fleets' target/eta so we can scope
    # the WorldModel build to the actual look-ahead needed. The full
    # DEFAULT_HORIZON (=30) is overkill when our longest fleet eta is
    # 10 — WorldModel.from_world is O(horizon × planets), so scaling
    # horizon to max_eta cuts the dominant cost roughly in half on
    # short-range turns. 2026-05-17 timing-fix item #2.
    fleet_targets: list[tuple[Fleet, float, object | None, int]] = []
    max_eta = 0
    for f in fleets:
        if int(f.owner) != my_id:
            continue
        ships = float(f.ships)
        target, eta = fleet_target_planet(f, planets_list, omega)
        eta_int = int(eta) if eta is not None else 0
        fleet_targets.append((f, ships, target, eta_int))
        if target is not None and eta_int > max_eta:
            max_eta = eta_int

    if not fleet_targets:
        return base

    effective_horizon = max(1, min(horizon, max_eta + 1))
    model = WorldModel.from_world(world, horizon=effective_horizon)

    delta = 0.0
    for f, ships, target, eta in fleet_targets:
        if target is None:
            # No planet on our trajectory — destined for OOB or sun.
            delta -= waste_weight * ships
            continue
        # Sun-crossing gate: `fleet_target_planet` ray-casts to the first
        # planet on the angle, ignoring the sun. If the fleet's chord
        # (current pos → target pos) passes within SUN_RADIUS of the
        # sun, the engine kills the fleet at the crossing tick
        # (orbit_wars.py:607: `point_to_segment_distance((CENTER, CENTER),
        # old_pos, new_pos) < SUN_RADIUS`). Without this gate, composite
        # silently credits captures the fleet never gets to make. Origin:
        # PI live observation 2026-05-17 PM ("large fleet into the sun").
        fleet_pos = (float(f.x), float(f.y))
        target_pos = (float(target.x), float(target.y))
        if point_to_segment_distance(
            (CENTER, CENTER), fleet_pos, target_pos,
        ) < SUN_RADIUS:
            delta -= waste_weight * ships
            continue
        # Comet-lifetime gate: WorldModel's simulate_planet_timeline
        # assumes planets persist for the full horizon and is unaware
        # that comets exit the board after `path_index` reaches the
        # end of the path (engine: orbit_wars.py:528-561). A fleet
        # aimed at a comet that expires before arrival hits empty space
        # — it never enters combat, never captures, never bounces. The
        # pred_owner check below would say "we'll own it at eta" for a
        # comet that's actually GONE by then. Pre-check matches the
        # engine's truth. Mirrors lib/missions/snipe.py:404-420 (H15)
        # and PI direction 2026-05-17: "use comets only if really
        # worth the risk and short lifetime".
        if int(target.id) in world.comet_ids:
            comet_life = comet_remaining_lifetime(int(target.id), world)
            if comet_life is None or comet_life <= eta:
                delta -= waste_weight * ships
                continue
        # Predict ownership at ETA. WorldModel includes THIS fleet in
        # its ledger, so the prediction reflects "world if we let this
        # fleet land." If pred_owner is NOT us at eta, the launch is
        # structurally lost (we bounce off a stronger defender, or
        # multi-arrival combat goes the other way) — apply waste
        # penalty. Otherwise the launch is constructive (causes the
        # capture OR over-reinforces a planet we'd hold anyway); either
        # way we do NOT add a per-fleet capture credit — the PV term in
        # the base already values the resulting ownership at the leaf.
        # See bug #15 fix v2 rationale in the docstring above.
        pred_owner = model.owner_at(target.id, eta)
        if pred_owner != my_id:
            delta -= waste_weight * ships

    return base + delta

# === inlined: lib/joint_solver/opening_planner.py ===


import math
from dataclasses import dataclass, field
from typing import Optional

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    _MILP_AVAILABLE = True
except ImportError:
    _MILP_AVAILABLE = False
    milp = None  # type: ignore[assignment]
    LinearConstraint = None  # type: ignore[assignment]
    Bounds = None  # type: ignore[assignment]

fleet_speed = speed


# ---------------------------------------------------------------------------
# Constants (tunable; initial values per the plan)
# ---------------------------------------------------------------------------

OPENING_HORIZON = 30        # planner active for steps 0..(OPENING_HORIZON-1)
# NOTE on renames vs the analogous constants in lp_outcome.py:
# the bundler flattens all inlined modules into one namespace, so
# `T_END`, `HOLD_WINDOW`, `DEFENDER_GUARD` would collide with lp_outcome's
# values (500/—/0) and silently overwrite the opening planner's intent.
# These three constants are file-local — no caller imports them — so
# we rename here to be collision-safe.
OPENING_T_END = 200         # value horizon: prod·(OPENING_T_END - arrival)
OPP_BONUS = 1.10            # multiplier for capturing opp planets (strips their prod)
OPENING_HOLD_WINDOW = 12    # ticks of post-capture defense we require feasibility for
OPENING_DEFENDER_GUARD = 2  # reserve at least this many ships on each source (subtracted ONCE from budget)
MIN_SOURCE_SHIPS = 3        # skip sources with fewer ships (newly captured planets fire sooner)
MAX_CONTESTERS_PER_TARGET = 1  # opening: each target captured at most once (avoid wasteful gang-ups)
TOP_PAIRS_PER_SOURCE = 20   # max candidates per source after pruning
TOP_TARGETS_PER_SOURCE = 8  # K in "top-K targets by prod/(dist+1)"
STRIDE = 1                  # launch-tick stride (must-include t=step_now)
ROI_THRESHOLD = 0.5         # accept launches with value ≥ ROI_THRESHOLD × ships invested
SPREAD_GAP = 6              # min fire_step separation between kept candidates
                            # of the same (src, tgt) pair — guarantees the
                            # MILP sees a budget-feasible late fire, not
                            # only the earliest 3 budget-conflicted ones.
                            # Fix 2 (modeling gap C, seed 384458460).
OPENING_VALUE_GAMMA = 0.95  # per-tick discount applied to candidate value
                            # over (wait + flight) time. Mirrors
                            # `prod_stream_discounted` in lp_outcome.py;
                            # penalises cross-board long-flight captures
                            # whose nominal value ignored the opportunity
                            # cost of ship-tied-up-in-transit time.
                            # Fix 3 (Bug B, seed 384458460).
OPP_RESPONSE_LAG = 4        # ticks of slack added to opp's optimal eta
                            # when checking whether opp can plausibly
                            # contest our arrival. Fix 4 (Modeling gap D).


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScheduleEntry:
    """One scheduled launch. `fire_step` is ABSOLUTE (not relative to step_now)."""
    fire_step: int
    src_id: int
    tgt_id: int
    ships: int
    angle: float
    eta: int
    value: float


@dataclass
class OpeningPlan:
    """Output of `plan()`."""
    schedule: list[ScheduleEntry]
    objective: float
    n_vars: int
    n_constraints: int
    status: str
    pruning_waterfall: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal candidate (one pre-pruned MILP variable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Candidate:
    column_id: int
    src_id: int
    tgt_id: int
    fire_step: int     # absolute step in the env
    eta: int           # ticks of flight from fire_step
    arrival: int       # fire_step + eta (absolute step)
    ships: int         # capture size required at arrival
    angle: float
    value: float       # objective coefficient
    src_idx: int       # row index in source-budget constraints
    tgt_idx: int       # row index in target-cap constraints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dist(a, b) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def _read_obs_fleets(world):
    raw = world.obs_raw
    if isinstance(raw, dict):
        return raw.get("fleets", []) or []
    return getattr(raw, "fleets", []) or []


def _nearest_enemy(world, tgt, me):
    """Return the enemy planet (not me, not neutral) closest to `tgt`, or None."""
    best = None
    best_d = float("inf")
    for p in world.planets_by_id.values():
        owner = int(p.owner)
        if owner == int(me) or owner < 0:
            continue
        d = _dist(p, tgt)
        if d < best_d:
            best_d = d
            best = p
    return best


def _ships_to_capture(tgt, owner_at_arrival: int, garrison_at_arrival: float,
                      my_id: int) -> int:
    """Closed-form ship count needed to capture `tgt` at arrival.
    If `tgt` is already mine at arrival, return 0 (reinforce = no-op for opening).
    """
    if owner_at_arrival == int(my_id):
        return 0
    return max(1, int(math.ceil(garrison_at_arrival)) + 1)


def _predict_opp_ships_at_target(tgt, arrival_step: int, world, my_id: int,
                                 ) -> int:
    """Maximum ships an opp could land at `tgt` within `arrival_step +
    OPP_RESPONSE_LAG` ticks. Considers every enemy planet with
    sufficient garrison; returns the strongest single source's
    available force (opp can't fire from every planet at once during
    the opening, so the strongest contestor is the worst case).

    Returns 0 if no opp source can plausibly contest. Fix 4
    (Modeling gap D, seed 384458460 p0→p2 misfire).
    """
    best = 0
    for p in world.planets_by_id.values():
        owner = int(p.owner)
        if owner == int(my_id) or owner < 0:
            continue
        ships_avail = int(p.ships) - OPENING_DEFENDER_GUARD
        if ships_avail < MIN_SOURCE_SHIPS:
            continue
        d = math.hypot(float(p.x) - float(tgt.x),
                       float(p.y) - float(tgt.y))
        v = fleet_speed(ships_avail)
        if v <= 0:
            continue
        eta = int(math.ceil(d / v))
        if eta <= arrival_step + OPP_RESPONSE_LAG:
            if ships_avail > best:
                best = ships_avail
    return best


def _expected_hold_duration(tgt, arrival: int, capture_residual: int,
                            world, model, my_id: int) -> int:
    """Closed-form expected hold duration in ticks AFTER capture.

    Two-stage check:

      Stage 1 (Fix 4 — ship-count opp race): if any opp source can
      plausibly land more ships at `tgt` near our arrival than our
      capture residual can hold, return 0. This is the "overwhelmed-
      on-arrival" case the eta-only model used to miss (e.g.
      seed-384458460 p0→p2 with opp's 60+ ship source in their
      quadrant).

      Stage 2 (legacy eta-delta): use `time_to_enemy_threat` for the
      eta race:
      - If opp arrives BEFORE us (delta ≤ 0): hold = 0.
      - If opp's earliest arrival ≥ arrival + OPENING_HOLD_WINDOW: full
        game-end credit (opp likely doesn't prioritise this outpost).
      - Tight race: scale the hold by 3 × the delta to reflect that
        opp won't typically spend their first action attacking us.
    """
    # Stage 1: ship-count check.
    opp_force = _predict_opp_ships_at_target(tgt, arrival, world, my_id)
    if opp_force >= int(capture_residual) + 3:
        return 0

    # Stage 2: eta-delta check (legacy).
    try:
        opp_threat_eta = model.time_to_enemy_threat(int(tgt.id), int(my_id), world)
    except Exception:
        opp_threat_eta = None
    if opp_threat_eta is None:
        return max(0, OPENING_T_END - arrival)
    delta = int(opp_threat_eta) - arrival
    if delta <= 0:
        return 0
    if delta >= OPENING_HOLD_WINDOW:
        return max(0, OPENING_T_END - arrival)
    return min(max(0, OPENING_T_END - arrival), 3 * delta)


def _target_already_claimed(tgt, base_arrivals, my_id: int,
                            horizon: int = OPENING_HORIZON + 50) -> bool:
    """True iff an existing in-flight FRIENDLY arrival will capture
    `tgt` within `horizon` ticks. When True, adding a new launch at
    `tgt` creates a redundant attack: either our new fleet arrives
    AFTER the in-flight one (wasted reinforcement) or BEFORE it
    (making the in-flight wasted). Cross-turn dedup — the per-solve
    `MAX_CONTESTERS_PER_TARGET = 1` cap doesn't catch this because the
    in-flight fleet is in `model.ledger`, not in this solve's
    candidate set. Closes the seed-384458460 step-13 redundancy where
    a p16→p8 launch was proposed while p0→p8 was already in flight.
    """
    if not base_arrivals:
        return False
    # Only friendly arrivals matter; opp arrivals can't claim FOR us.
    friendly = [a for a in base_arrivals if int(a[1]) == int(my_id)]
    if not friendly:
        return False
    timeline = simulate_planet_timeline(tgt, base_arrivals, horizon=horizon)
    owner_at = timeline["owner_at"]
    for t in range(1, horizon + 1):
        if int(owner_at.get(t, -1)) == int(my_id):
            return True
    return False


def _is_minimally_holdable(tgt, arrival: int, capture_residual: int,
                           world, model, my_id: int) -> bool:
    """Lower-bound feasibility: did we get to the planet first AND survive
    immediate counter? If opp arrives before us, the capture is futile.

    Anything beyond this is handled by the value-weighting in
    `_expected_hold_duration`, so the MILP picks captures with the best
    production-over-hold-window."""
    try:
        opp_threat_eta = model.time_to_enemy_threat(int(tgt.id), int(my_id), world)
    except Exception:
        opp_threat_eta = None
    if opp_threat_eta is None:
        return True
    return int(opp_threat_eta) > arrival


# ---------------------------------------------------------------------------
# Candidate generation (the prune chain)
# ---------------------------------------------------------------------------


def _build_candidates(world, model, my_id: int, num_seats: int,
                      ) -> tuple[list[_Candidate], dict[str, int]]:
    """Apply the 6-step prune chain. Return (candidates, waterfall_stats)."""
    waterfall = {"naive_upper_bound": 0, "after_source_pool": 0,
                 "after_top_targets": 0, "after_reachability": 0,
                 "after_trajectory": 0, "after_feasibility": 0,
                 "after_top_pairs": 0}

    step_now = int(world.step)
    omega = float(world.omega)

    # 1. Source pool — my planets with at least MIN_SOURCE_SHIPS ships.
    my_planets = [p for p in world.planets_by_id.values()
                  if int(p.owner) == int(my_id)
                  and int(p.ships) >= MIN_SOURCE_SHIPS]
    waterfall["after_source_pool"] = len(my_planets)
    if not my_planets:
        return [], waterfall

    # All non-mine, non-comet planets are potential targets.
    comet_ids = set(world.comet_ids) if world.comet_ids else set()
    all_targets = [p for p in world.planets_by_id.values()
                   if int(p.owner) != int(my_id) and int(p.id) not in comet_ids]

    waterfall["naive_upper_bound"] = (
        len(my_planets) * len(all_targets) * ((OPENING_HORIZON // STRIDE) + 1)
    )

    src_ids_in_use: set[int] = set()
    tgt_ids_in_use: set[int] = set()
    all_candidates: list[_Candidate] = []
    next_id = 0

    fire_offsets = [0] + list(range(STRIDE, OPENING_HORIZON, STRIDE))

    for src in my_planets:
        # 2. Per-source top-K targets by prod / (dist + 1).
        scored_targets = sorted(
            ((float(t.production) / (_dist(src, t) + 1.0), t) for t in all_targets),
            key=lambda x: x[0], reverse=True,
        )
        top_targets = [t for _s, t in scored_targets[:TOP_TARGETS_PER_SOURCE]]
        waterfall["after_top_targets"] += len(top_targets)

        # 3+4+5: reachability × trajectory × stride-2 fire ticks.
        per_src_pruned: list[_Candidate] = []
        for tgt in top_targets:
            # Fix 1 (Bug A): drop targets that an in-flight friendly
            # arrival will capture. Cross-turn dedup — at turn T+1
            # the previous turn's emission is in `model.ledger` and a
            # second launch from any source would create a redundant
            # attack.
            tgt_base_arrivals = list(model.ledger.get(int(tgt.id), []))
            if _target_already_claimed(tgt, tgt_base_arrivals, my_id):
                waterfall.setdefault("dropped_already_claimed", 0)
                waterfall["dropped_already_claimed"] += 1
                continue
            for offset in fire_offsets:
                fire_step = step_now + offset
                # Initial ship estimate for aim_and_eta — refine via fixed point.
                ships_est = max(OPENING_DEFENDER_GUARD, int(tgt.ships) + 1)
                # Two-step refinement is enough (eta converges fast).
                for _ in range(2):
                    res = aim_and_eta(src, tgt, ships_est, omega, wait_N=offset)
                    if res is None:
                        break
                    angle, eta_flight = res
                    if eta_flight is None or eta_flight <= 0 or eta_flight > OPENING_HORIZON + 10:
                        res = None
                        break
                    arrival_total = offset + int(eta_flight)
                    # Predict garrison at arrival via WorldModel (closed-form).
                    # Use the ledger as-is; opp counter-projection lives in (C3).
                    base_arrivals = list(model.ledger.get(int(tgt.id), []))
                    try:
                        owner_at_arr, gar_at_arr = predict_garrison_at(
                            tgt, arrival_total, base_arrivals,
                        )
                    except Exception:
                        res = None
                        break
                    needed = _ships_to_capture(tgt, int(owner_at_arr), float(gar_at_arr), my_id)
                    if needed <= 0:
                        res = None
                        break
                    if needed == ships_est:
                        break
                    ships_est = needed
                if res is None:
                    continue
                angle, eta_flight = res
                arrival_total = offset + int(eta_flight)
                # Recompute garrison with final ships estimate (no opp projection here).
                base_arrivals = list(model.ledger.get(int(tgt.id), []))
                owner_at_arr, gar_at_arr = predict_garrison_at(
                    tgt, arrival_total, base_arrivals,
                )
                needed = _ships_to_capture(tgt, int(owner_at_arr), float(gar_at_arr), my_id)
                if needed <= 0:
                    continue
                # Source budget at fire tick (post-production, pre-launch).
                src_ships_at_fire = int(src.ships) + int(src.production) * offset
                if needed + OPENING_DEFENDER_GUARD > src_ships_at_fire:
                    continue  # can't afford while keeping defender

                # 4. Trajectory feasibility against fire-time geometry.
                # predict_fleet_fate advances planet positions by wait_N=offset
                # orbital ticks so wait-then-fire candidates are checked against
                # their actual fire-time geometry, not the turn-now snapshot.
                try:
                    fate = predict_fleet_fate(
                        src, tgt, angle, needed, world, wait_N=int(offset),
                    )
                except Exception:
                    fate = None
                if fate is not None and getattr(fate, "outcome", "") != "target":
                    waterfall.setdefault("dropped_trajectory", 0)
                    waterfall["dropped_trajectory"] += 1
                    continue

                capture_residual = needed - int(math.ceil(gar_at_arr))
                if capture_residual < 1:
                    continue

                # Value = production × hold_window × opp_bonus, where
                # hold_window is the expected ticks we hold post-capture
                # before opp could plausibly recapture. Indefensible
                # captures get hold_window=0 → value=0 → naturally rejected
                # by the ROI gate below. Contested captures with positive
                # hold get a fair-but-not-inflated value.
                hold_dur = _expected_hold_duration(
                    tgt, arrival_total, capture_residual, world, model, my_id,
                )
                if hold_dur <= 0:
                    waterfall.setdefault("dropped_defense", 0)
                    waterfall["dropped_defense"] += 1
                    continue
                opp_bonus = OPP_BONUS if int(tgt.owner) != -1 else 1.0
                # Fix 3: discount value by time-to-capture so cross-
                # board long-flight candidates lose their nominal
                # advantage relative to close fast captures. Matches
                # `prod_stream_discounted` semantics in lp_outcome.py.
                time_to_capture = int(offset) + int(eta_flight)
                discount = OPENING_VALUE_GAMMA ** float(time_to_capture)
                value = (float(int(tgt.production)) * float(hold_dur)
                         * float(opp_bonus) * float(discount))
                # Per-launch ROI filter — gentler than 1:1 to match the
                # baseline's aggressive opening throughput. Even half-ROI
                # captures contribute to the production base once held.
                if value < ROI_THRESHOLD * float(needed):
                    waterfall.setdefault("dropped_low_roi", 0)
                    waterfall["dropped_low_roi"] += 1
                    continue

                src_ids_in_use.add(int(src.id))
                tgt_ids_in_use.add(int(tgt.id))
                per_src_pruned.append(_Candidate(
                    column_id=next_id, src_id=int(src.id), tgt_id=int(tgt.id),
                    fire_step=fire_step, eta=int(eta_flight), arrival=step_now + arrival_total,
                    ships=int(needed), angle=float(angle), value=float(value),
                    src_idx=-1, tgt_idx=-1,  # filled in below
                ))
                next_id += 1

        # 6. Per-(src, tgt) cap with budget-aware FIRE_STEP SPREAD.
        # Group by (src, tgt); within each group keep up to 3
        # candidates, picking by descending value but requiring each
        # kept candidate's fire_step to be ≥ SPREAD_GAP away from
        # already-kept fires in the same group. Without the spread,
        # top-3-by-value always picks the earliest 3 fire_steps
        # (value monotonically decreases with fire_step) — all of
        # which share the source's cramped early ship budget. The
        # spread guarantees the MILP sees at least one budget-
        # feasible LATE fire per pair so a second wave from a
        # regenerated source becomes pickable. Fix 2.
        by_tgt: dict[int, list[_Candidate]] = {}
        for c in per_src_pruned:
            by_tgt.setdefault(int(c.tgt_id), []).append(c)
        diverse: list[_Candidate] = []
        for tid, group in by_tgt.items():
            kept_in_group: list[_Candidate] = []
            for c in sorted(group, key=lambda c: c.value, reverse=True):
                if len(kept_in_group) >= 3:
                    break
                if all(abs(int(c.fire_step) - int(k.fire_step)) >= SPREAD_GAP
                       for k in kept_in_group):
                    kept_in_group.append(c)
            diverse.extend(kept_in_group)
        # Now take top TOP_PAIRS_PER_SOURCE by raw value across this source's
        # diverse candidates.
        diverse.sort(key=lambda c: c.value, reverse=True)
        keep = diverse[:TOP_PAIRS_PER_SOURCE]
        all_candidates.extend(keep)

    waterfall["after_top_pairs"] = len(all_candidates)
    waterfall["after_reachability"] = len(all_candidates)  # rolled together
    waterfall["after_trajectory"] = len(all_candidates)
    waterfall["after_feasibility"] = len(all_candidates)

    # Renumber column_ids contiguously and fill in src/tgt indexes.
    src_idx_map = {sid: i for i, sid in enumerate(sorted(src_ids_in_use))}
    tgt_idx_map = {tid: i for i, tid in enumerate(sorted(tgt_ids_in_use))}
    out: list[_Candidate] = []
    for new_id, c in enumerate(all_candidates):
        out.append(_Candidate(
            column_id=new_id, src_id=c.src_id, tgt_id=c.tgt_id,
            fire_step=c.fire_step, eta=c.eta, arrival=c.arrival,
            ships=c.ships, angle=c.angle, value=c.value,
            src_idx=src_idx_map[c.src_id], tgt_idx=tgt_idx_map[c.tgt_id],
        ))
    return out, waterfall


# ---------------------------------------------------------------------------
# Greedy fallback
# ---------------------------------------------------------------------------


def _greedy_fallback(candidates: list[_Candidate], world, my_id: int,
                     ) -> tuple[list[_Candidate], float]:
    """Pure-Python descending-value greedy with budget + gang-up tracking."""
    step_now = int(world.step)
    # Per-source remaining ship pool, indexed by (src_id, fire_offset).
    src_inv: dict[int, tuple[int, int]] = {}  # src_id -> (initial_ships, production)
    for c in candidates:
        if c.src_id in src_inv:
            continue
        src_p = world.planets_by_id.get(c.src_id)
        if src_p is not None:
            src_inv[c.src_id] = (int(src_p.ships), int(src_p.production))

    emitted_by_src_fire: dict[tuple[int, int], int] = {}  # (src, fire_step) -> ships used
    tgt_count: dict[int, int] = {}
    chosen: list[_Candidate] = []
    obj = 0.0

    for c in sorted(candidates, key=lambda x: x.value, reverse=True):
        if tgt_count.get(c.tgt_id, 0) >= MAX_CONTESTERS_PER_TARGET:
            continue
        # Source budget check: cumulative emissions up to c.fire_step ≤ available.
        initial, prod = src_inv.get(c.src_id, (0, 0))
        offset = c.fire_step - step_now
        used = sum(v for (s, fs), v in emitted_by_src_fire.items()
                   if s == c.src_id and fs <= c.fire_step)
        if used + c.ships > initial + prod * max(0, offset) - OPENING_DEFENDER_GUARD:
            continue
        chosen.append(c)
        emitted_by_src_fire[(c.src_id, c.fire_step)] = (
            emitted_by_src_fire.get((c.src_id, c.fire_step), 0) + c.ships
        )
        tgt_count[c.tgt_id] = tgt_count.get(c.tgt_id, 0) + 1
        obj += c.value
    return chosen, obj


# ---------------------------------------------------------------------------
# MILP solver
# ---------------------------------------------------------------------------


def _solve_milp(candidates: list[_Candidate], world, my_id: int,
                time_limit_seconds: float):
    """Run the MILP. Return (chosen_candidates, objective, status, n_constraints)."""
    if not candidates:
        return [], 0.0, "empty", 0
    if not _MILP_AVAILABLE:
        chosen, obj = _greedy_fallback(candidates, world, my_id)
        return chosen, obj, "greedy_fallback", 0

    import numpy as np

    n = len(candidates)
    step_now = int(world.step)

    # Inventories.
    src_ids = sorted({c.src_id for c in candidates})
    src_inv: dict[int, tuple[int, int]] = {}
    for sid in src_ids:
        p = world.planets_by_id.get(sid)
        if p is None:
            src_inv[sid] = (0, 0)
        else:
            src_inv[sid] = (int(p.ships), int(p.production))

    tgt_ids = sorted({c.tgt_id for c in candidates})

    # Objective: minimize -value (with lex tie-breaker for stability).
    c_vec = np.array(
        [-(c.value - 1e-6 * c.column_id) for c in candidates], dtype=float,
    )

    A_rows: list[list[float]] = []
    b_ub: list[float] = []

    # (C1) Per-source budget over time. OPENING_DEFENDER_GUARD is subtracted
    # ONCE from the right-hand side per (src, u) row — not per launch — so a
    # source can do many launches as long as its CUMULATIVE outflow leaves
    # OPENING_DEFENDER_GUARD ships at the home.
    #   Σ_{c: src(c)=src, c.fire_step ≤ u} ships(c) · x_c
    #     ≤ initial(src) + prod(src) · (u - step_now) - OPENING_DEFENDER_GUARD
    fire_ticks_for_budget = sorted({c.fire_step for c in candidates})
    for sid in src_ids:
        initial, prod = src_inv[sid]
        for u in fire_ticks_for_budget:
            row = [0.0] * n
            any_in_row = False
            for j, c in enumerate(candidates):
                if c.src_id == sid and c.fire_step <= u:
                    row[j] = float(c.ships)
                    any_in_row = True
            if not any_in_row:
                continue
            A_rows.append(row)
            b_ub.append(float(initial + prod * max(0, u - step_now) - OPENING_DEFENDER_GUARD))

    # (C2) Per-target gang-up cap.
    for tid in tgt_ids:
        row = [0.0] * n
        any_in_row = False
        for j, c in enumerate(candidates):
            if c.tgt_id == tid:
                row[j] = 1.0
                any_in_row = True
        if not any_in_row:
            continue
        A_rows.append(row)
        b_ub.append(float(MAX_CONTESTERS_PER_TARGET))

    if not A_rows:
        # No constraints — just pick all positive.
        chosen = [c for c in candidates if c.value > 0]
        obj = sum(c.value for c in chosen)
        return chosen, obj, "no_constraints", 0

    A = np.array(A_rows, dtype=float)
    b = np.array(b_ub, dtype=float)
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    integrality = np.ones(n, dtype=int)
    constraints = LinearConstraint(A, ub=b)

    try:
        res = milp(c=c_vec, constraints=constraints, integrality=integrality,
                   bounds=bounds, options={"time_limit": time_limit_seconds})
    except Exception:
        chosen, obj = _greedy_fallback(candidates, world, my_id)
        return chosen, obj, "milp_exception_greedy", len(A_rows)

    if res.x is None:
        chosen, obj = _greedy_fallback(candidates, world, my_id)
        return chosen, obj, "milp_no_solution_greedy", len(A_rows)

    chosen = [c for j, c in enumerate(candidates) if res.x[j] > 0.5]
    obj = sum(c.value for c in chosen)
    return chosen, obj, "milp_ok", len(A_rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def opening_plan(world, model, my_id: int, num_seats: int,
         *, time_limit_seconds: float = 0.15) -> OpeningPlan:
    """Build the opening schedule for the current world."""
    candidates, waterfall = _build_candidates(world, model, my_id, num_seats)
    if not candidates:
        return OpeningPlan(schedule=[], objective=0.0,
                           n_vars=0, n_constraints=0,
                           status="no_candidates",
                           pruning_waterfall=waterfall)

    chosen, obj, status, n_constraints = _solve_milp(
        candidates, world, my_id, time_limit_seconds,
    )

    schedule = [
        ScheduleEntry(
            fire_step=c.fire_step, src_id=c.src_id, tgt_id=c.tgt_id,
            ships=c.ships, angle=c.angle, eta=c.eta, value=c.value,
        )
        for c in sorted(chosen, key=lambda c: (c.fire_step, c.column_id))
    ]

    return OpeningPlan(
        schedule=schedule, objective=float(obj),
        n_vars=len(candidates), n_constraints=int(n_constraints),
        status=str(status), pruning_waterfall=waterfall,
    )

# === inlined: agents/baseline/proposer.py ===
"""Candidate proposer: fire-now + multi-wait-grid, cheap-ranked, banded-deduped.

Pipeline per turn:
  1. for each owned source S with >= MIN_FLEET_SIZE ships:
       for each non-owned-or-threatened target T in nearest-K of S:
         emit fire-now candidates at (capture_size, 2*capture, full-budget)
         emit wait-then-fire candidates at extra_surplus in (0, 5, 12)
  2. cheap-rank each candidate by analytic Δ (capture/bounce/reinforce)
  3. dedup per (src_id, tgt_id, wait_band) keeping the top cheap-Δ
     — wait_band = {0, 1..7, >=8}; lets the validator compare fire-now
     vs short-wait vs long-wait against the same target.
"""


import math
import os

# from lib.aim import aim_comet, aim_orbiting  # inlined by bundle_agent.py
# from lib.fleet import speed as fleet_speed  # inlined by bundle_agent.py
fleet_speed = speed
# from lib.orbit import is_orbiting, predict_relative  # inlined by bundle_agent.py
# from lib.scoring import pv_horizon  # inlined by bundle_agent.py
# from lib.trajectory import predict_fleet_fate  # inlined by bundle_agent.py
# from lib.world_model import _comet_paths_by_id, _position_at, comet_remaining_lifetime  # inlined by bundle_agent.py

NUM_TARGETS_PER_SOURCE = 8
MIN_FLEET_SIZE = 2
SIM_SETTLE_TURNS = 2
MIN_HORIZON = 25
MAX_HORIZON = 40
WAIT_EXTRA_SURPLUS = (0, 5, 12)  # legacy forward grid (kept for rollback)
CHEAP_REJECT_THRESHOLD = -10.0
EPISODE_STEPS = 500
GAMMA = 0.99
# Fix (strategic defense, 2026-05-21): for high-prod own planets,
# floor the reinforce target to a preemptive stockpile so the LP can
# see strategic defense before shortfall materialises. Without this,
# `capture_size` returns 0 whenever current garrison covers current
# threat — blinding the LP to opp's build-up on a planet we won't
# defend until it's too late. Prefixed STRATEGIC_* to avoid bundler
# namespace collision (cf. OPENING_* rename, 2026-05-21 AM).
STRATEGIC_DEFENSE_PROD = 4    # production threshold for "strategic"
STRATEGIC_STOCKPILE_TICKS = 5 # buffer = N ticks × planet's production

# Larger→smaller source-drain hardening (2026-05-27, PI direction).
# Active in `_source_survives_launch` only when
# `src.production > tgt.production` — i.e. we're sending a fleet from
# a higher-prod planet to a lower-prod one. See plan
# `/root/.claude/plans/fix-one-and-two-cuddly-dewdrop.md` Fix 3.
SAFETY_MARGIN_DRAIN = 1.3      # stricter margin under threat
STOCKPILE_PROD_MULT = 5        # `floor = N × source production`

# Predicted-threat garrison reserve (2026-05-27). When ALPHA > 0, the
# source must retain `ceil(ALPHA × predicted_threat_force(src, WINDOW))`
# ships after the launch. `predicted_threat_force` (lib/world_model.py)
# sums both in-flight ledger arrivals AND potential launches from
# stationary opp planets that can reach `src` within WINDOW steps —
# i.e., the rotating-opp wave that the legacy ledger-only force misses.
# Default ALPHA=0.0 (clause skipped, no-op). WINDOW=30 matches the
# 30-tick threat horizon convention used elsewhere in this file.
THREAT_RESERVE_ALPHA = float(
    os.environ.get("BASELINE_THREAT_RESERVE_ALPHA", "0.0"),
)
THREAT_RESERVE_WINDOW = int(
    os.environ.get("BASELINE_THREAT_RESERVE_WINDOW", "30"),
)

# Backward wait grid (2026-05-18): anchored on min_wait_affordable.
# Replaces forward WAIT_EXTRA_SURPLUS = (0, 5, 12) grid that caused
# under-emission. Diagnosis: at Roman game (76941081) step 90 with 454
# ships across 9 planets, proposer emitted 18 candidates, 15 of which
# were wait_N > 0. Chooser picked top-Δ candidate (wait_N=17, fire-now-
# capable src reserved), emitted 0 launches. Repeat every turn → 59pct
# idle. With backward grid, already-affordable (src, tgt) pairs emit
# NO wait variants; chooser only sees fire-now → emits.
WAIT_GRID_MODE = os.environ.get("BASELINE_WAIT_GRID", "backward").strip().lower()
WAIT_BUFFER_OFFSET = 3   # backward grid emits {min_w, min_w + 3}

# Bug #12 window constant — promoted to `lib/world_model.py` so both
# this proposer and the in-rollout defensive policy
# (`lib/opp_model.me_defensive_action`) import it from one location.
# from lib.world_model import WAVE_LOOKAHEAD  # noqa: E402  # inlined by bundle_agent.py

# Reactor-aware launch selection (2026-05-19 PM).
#
# Two-part fix for the "predictable first-mover" trap PI surfaced from
# live-replay observation: we launch a fleet across the map, opp sees
# it in flight and either reinforces the target so we bounce OR lets
# us land and recaptures cheaply. Holdability check (existing) catches
# "we'll keep the planet"; this catches "did we pay more than opp would
# have paid to take it?"
#
# Part A — cost-parity filter (`_target_cost_parity_ok`): drops
# candidate launches where the cheapest opp reactor pays materially
# less ships than our capture cost. A launch can be holdable AND
# wasteful (we keep it but paid more than necessary).
# Part B — reactor candidate generator (`_enumerate_reactor_candidates`):
# for each opp fleet in flight to a non-our target, propose our own
# launch from a nearby source sized to recapture the target after opp
# lands. We become the cheap second-mover.
COST_PARITY_MARGIN_DEFAULT = 0.7        # reject if opp pays < 70 % of our cost
MIN_REACTOR_SHIPS = 8                    # below this an opp planet can't realistically reactor
MAX_REACTOR_CANDIDATES_PER_TURN = 12     # global cap on Part B output
REACTOR_TOP_K_SOURCES_PER_TARGET = 3     # per-target source enumeration cap


def _comet_path_entry(world, tgt_id):
    """Look up (path, path_index) for a comet target, or None if not a comet.

    Honoured by `aim_and_eta` only when `BASELINE_COMET_AIM` is enabled.
    Wrapper around `lib.world_model._comet_paths_by_id` that's local to
    the proposer so test fixtures can monkey-patch it independently if
    needed.
    """
    if world is None:
        return None
    if int(tgt_id) not in getattr(world, "comet_ids", set()):
        return None
    paths = _comet_paths_by_id(world)
    return paths.get(int(tgt_id))


def aim_and_eta(src, tgt, ships: int, omega: float, wait_N: int = 0, world=None):
    """Return (aim_angle_radians, ceil_eta_turns) for one (src, tgt, ships).

    For COMET targets (target_id in world.comet_ids) AND when `world` is
    provided AND env-var `BASELINE_COMET_AIM != "off"`, uses path-indexed
    lead via `lib.aim.aim_comet`. Comets travel polynomial paths at
    cometSpeed=4 board-units/turn, NOT orbital rotation; using the
    orbital lead causes 20-40-board-unit misses (ep 77087563 / sub
    52811320, fleet 32 OOB).

    For orbiting non-comet targets, jointly solves aim + eta via
    lib.aim.aim_orbiting. For wait_N>0 candidates, pre-rotates BOTH src
    and tgt by omega*wait_N so aim is computed at the geometry that
    will hold at fire time (co-rotating planets preserve relative
    geometry).

    The `world` argument is optional (default None) so existing callers
    that don't pass it keep the pre-fix orbital behaviour. The proposer
    `propose()` entry threads `world` through here.
    """
    # Path-indexed lead for comet targets (Part C, 2026-05-19 PM).
    if (
        world is not None
        and os.environ.get("BASELINE_COMET_AIM", "").strip().lower() != "off"
    ):
        comet_entry = _comet_path_entry(world, int(tgt.id))
        if comet_entry is not None:
            path, path_index = comet_entry
            # For wait_N>0, advance the effective path_index by wait_N
            # (the comet will have moved that many positions by the time
            # we launch). The source planet is treated as having waited
            # in place; if src is itself orbiting we pre-rotate it too.
            effective_index = int(path_index) + int(wait_N)
            src_x, src_y = float(src.x), float(src.y)
            if wait_N > 0 and is_orbiting(list(src)):
                src_x, src_y = predict_relative(list(src), omega, wait_N)
            res = aim_comet(
                (src_x, src_y), src.radius, list(tgt), tgt.radius, ships,
                path, effective_index,
            )
            if res is not None:
                return float(res[0]), max(1, int(math.ceil(float(res[2]))))
            # Comet exits before arrival: fall through to the simple
            # atan2-at-current-position path below. The trajectory
            # filter or comet-lifetime gate will catch the resulting
            # candidate as a non-target outcome.

    if is_orbiting(list(tgt)):
        tgt_list = list(tgt)
        src_x, src_y = float(src.x), float(src.y)
        if wait_N > 0:
            fx, fy = predict_relative(tgt_list, omega, wait_N)
            tgt_list[2] = fx
            tgt_list[3] = fy
            src_x, src_y = predict_relative(list(src), omega, wait_N)
        res = aim_orbiting(
            (src_x, src_y), src.radius, tgt_list, tgt.radius, ships, omega,
        )
        if res is not None:
            return float(res[0]), max(1, int(math.ceil(float(res[2]))))
    angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
    flight = max(
        0.0,
        math.hypot(src.x - tgt.x, src.y - tgt.y) - src.radius - tgt.radius - 0.1,
    )
    spd = fleet_speed(ships)
    if spd <= 0:
        return angle, 999
    return angle, int(math.ceil(flight / spd))


def nearest_k(targets, src, k: int):
    return sorted(
        targets,
        key=lambda t: math.hypot(src.x - t.x, src.y - t.y),
    )[:k]


def capture_size(src, tgt, model, omega: float, me: int, world) -> int:
    """WorldModel-aware minimum size to take (or hold) tgt from src."""
    if int(tgt.owner) == me:
        # Reinforce: cover the predicted shortfall vs incoming threat.
        enemy_eta = model.time_to_enemy_threat(int(tgt.id), me, world)
        if enemy_eta is None:
            return 0
        # Bug #12 fix (2026-05-18): widen the inflight window from
        # `enemy_eta + 1` to `enemy_eta + WAVE_LOOKAHEAD` so a staggered
        # multi-wave attack (eta=2 + eta=4) counts together. Pre-fix,
        # asdf-game (76947663) step 37 had two opp fleets inbound to
        # P15 (40 ships at eta=2, 65 ships at eta=4); only the 40-ship
        # earliest wave entered the sum, shortfall was negative, no
        # reinforce candidate emitted, P15 fell.
        enemy_inflight = sum(
            ships
            for (eta_arr, owner, ships) in model.ledger.get(int(tgt.id), [])
            if owner != me and eta_arr <= enemy_eta + WAVE_LOOKAHEAD
        )
        enemy_potential = 0.0
        if enemy_inflight <= 0:
            # Bug #3 fix (2026-05-18): the speculative-launch potential
            # accrues opp production over `enemy_eta` ticks, matching the
            # accrual already applied to our garrison below. Pre-fix
            # enemy_potential was the OPP planet's current ship count
            # (static) while my_garrison accrued — asymmetric prediction
            # made shortfall almost always negative, so no preemptive
            # reinforce candidates were emitted.
            best_enemy_ships = 0.0
            best_enemy_prod = 0.0
            for p in world.planets_by_id.values():
                if int(p.owner) < 0 or int(p.owner) == me:
                    continue
                if int(p.ships) > best_enemy_ships:
                    best_enemy_ships = float(p.ships)
                    best_enemy_prod = float(p.production)
            enemy_potential = (
                best_enemy_ships + best_enemy_prod * float(enemy_eta)
            )
        enemy_strength = max(enemy_inflight, enemy_potential)
        my_garrison = float(tgt.ships) + float(tgt.production) * enemy_eta
        shortfall = enemy_strength - my_garrison + 1
        base = max(0, int(math.ceil(shortfall)))
        # Fix (strategic stockpile): high-prod own planets get a
        # preemptive defensive buffer even when current shortfall ≤ 0.
        # Pre-fix `base == 0` returned 0 → `enumerate_ship_counts`
        # returned [] → no reinforce candidate → drained home stays
        # undefended through mid-game build-up.
        if int(tgt.production) >= STRATEGIC_DEFENSE_PROD:
            base = max(base, STRATEGIC_STOCKPILE_TICKS * int(tgt.production))
        return base

    initial = max(MIN_FLEET_SIZE, int(tgt.ships) + 1)
    _angle, eta = aim_and_eta(src, tgt, initial, omega, world=world)
    pred = float(model.ships_at(int(tgt.id), eta) or 0.0)
    return max(MIN_FLEET_SIZE, int(math.ceil(pred)) + 1)


def enumerate_ship_counts(src, tgt, model, omega: float, me: int, world) -> list[int]:
    """Fire-now ship-count set: capture-size, 2x capture-size, full budget.

    Always emits `budget` as a candidate when `budget >= MIN_FLEET_SIZE`
    (Fix — bundle blind spot, 2026-05-21). Pre-fix the third size was
    gated by `budget > cap`, which dropped candidates from sources
    that couldn't solo-capture — the LP literally never saw multi-
    source bundle options. With the gate removed, every source within
    range emits at least one column; the LP's outcome-table subset
    enumeration (lib/joint_solver/outcome_table.py:73-130) correctly
    scores the joint capture.
    """
    cap = capture_size(src, tgt, model, omega, me, world)
    budget = int(src.ships)
    if cap == 0:
        return []  # reinforce-targets with no threat
    sizes = set()
    if MIN_FLEET_SIZE <= cap <= budget:
        sizes.add(cap)
    if 2 * cap <= budget:
        sizes.add(2 * cap)
    if budget >= MIN_FLEET_SIZE:
        sizes.add(budget)
    return sorted(sizes)


def wait_then_fire_variants_forward(src, tgt, model, omega: float, me: int, world=None):
    """Forward wait-grid (legacy): enumerate fixed WAIT_EXTRA_SURPLUS = (0, 5, 12).

    Kept for rollback via BASELINE_WAIT_GRID=forward. Caused under-emission
    when src is already armed (always emits wait_N=1 variant that
    out-scores fire-now in chooser Δ; chooser picks the wait, reserves
    src+tgt, emits nothing).
    """
    if int(tgt.owner) == me:
        return []
    prod = int(src.production)
    if prod <= 0:
        return []

    initial = max(MIN_FLEET_SIZE, int(tgt.ships) + 1)
    _a0, eta0 = aim_and_eta(src, tgt, initial, omega, world=world)
    pred_now = float(model.ships_at(int(tgt.id), eta0) or 0.0)
    cap_now = max(MIN_FLEET_SIZE, int(math.ceil(pred_now)) + 1)

    variants = []
    seen: set[tuple[int, int]] = set()
    for extra_surplus in WAIT_EXTRA_SURPLUS:
        target_fleet = cap_now + extra_surplus
        shortfall = target_fleet - int(src.ships)
        if shortfall <= 0:
            wait_N = 1  # feasible-now still gets a wait-1 variant
        else:
            wait_N = (shortfall + prod - 1) // prod  # ceil
        if wait_N < 1:
            continue

        angle, eta = aim_and_eta(src, tgt, target_fleet, omega, wait_N=wait_N, world=world)
        pred_at_arr = float(model.ships_at(int(tgt.id), wait_N + eta) or 0.0)
        cap_final = max(MIN_FLEET_SIZE, int(math.ceil(pred_at_arr)) + 1)
        final_fleet = cap_final + extra_surplus

        budget_at_wait = int(src.ships) + prod * wait_N
        if final_fleet > budget_at_wait:
            final_fleet = budget_at_wait

        if wait_N + eta + SIM_SETTLE_TURNS > MAX_HORIZON:
            continue

        key = (wait_N, final_fleet)
        if key in seen:
            continue
        seen.add(key)
        variants.append((final_fleet, wait_N, angle, eta))
    return variants


def min_wait_affordable(src, tgt, model, omega: float, me: int, world=None) -> int | None:
    """Smallest wait_N at which src can affordably capture tgt.

    Returns:
      0   — src can already fire-now (cap_now ≤ src.ships).
      N>0 — src must accumulate N turns before firing.
      None — hopeless within MAX_HORIZON (opp accumulates faster than
             we can; pair never affordable).

    Mirrors the affordability math in `wait_then_fire_variants_forward`
    so callers get a consistent answer. Used to anchor the backward
    wait-grid: when min_wait == 0, NO wait variants are emitted (the
    fire-now path covers it; speculative waits like the old wait_N=1
    block fire-now from being chosen).
    """
    if int(tgt.owner) == me:
        return None  # reinforce path handled separately
    if int(src.production) <= 0:
        return None  # src can't accumulate; wait is pointless
    prod = int(src.production)

    # Fire-now feasibility check
    initial = max(MIN_FLEET_SIZE, int(tgt.ships) + 1)
    _a0, eta0 = aim_and_eta(src, tgt, initial, omega, wait_N=0, world=world)
    pred_now = float(model.ships_at(int(tgt.id), eta0) or 0.0)
    cap_now = max(MIN_FLEET_SIZE, int(math.ceil(pred_now)) + 1)
    if cap_now <= int(src.ships):
        return 0

    # Iterate wait_N until affordable (no closed form due to
    # fleet_speed(ships) nonlinearity)
    for wait_N in range(1, MAX_HORIZON):
        budget = int(src.ships) + prod * wait_N
        # Cheap pre-check: even bare capture of current garrison exceeds budget
        if max(MIN_FLEET_SIZE, int(tgt.ships) + 1) > budget:
            continue
        target_fleet = max(MIN_FLEET_SIZE, int(tgt.ships) + 1)
        _angle, eta = aim_and_eta(src, tgt, target_fleet, omega, wait_N=wait_N, world=world)
        pred_at_arrival = float(model.ships_at(int(tgt.id), wait_N + eta) or 0.0)
        cap_final = max(MIN_FLEET_SIZE, int(math.ceil(pred_at_arrival)) + 1)
        if cap_final <= budget and wait_N + eta + SIM_SETTLE_TURNS <= MAX_HORIZON:
            return wait_N
    return None  # hopeless within MAX_HORIZON


def wait_then_fire_variants(src, tgt, model, omega: float, me: int, world=None):
    """Backward wait grid: anchor on min_wait_affordable.

    Returns list of (ships, wait_N, angle, eta). Behaviour:
    - Already-armed src (min_wait == 0) → return []. Fire-now path
      handles this; we don't emit speculative waits that compete with
      fire-now in chooser Δ ranking.
    - Hopeless pair (min_wait is None) → return []. Saves chooser
      cycles; this pair's launches will all bounce.
    - Otherwise → emit candidates at {min_wait, min_wait + WAIT_BUFFER_OFFSET}
      × {cap_final, 2×cap_final, budget}. The bare-capture variant gives
      the chooser a lean option; the budget variant USES the accumulated
      ships we waited for (instead of leaving them idle on the source —
      a fix for the 2026-05-18 backward-grid bug where wait_N variants
      emitted only bare-capture amounts, wasting the accumulation and
      leaving 1-ship residue on captured planets vulnerable to opp
      recapture in 4P).

    Forward-mode rollback available via BASELINE_WAIT_GRID=forward.
    """
    if WAIT_GRID_MODE == "forward":
        return wait_then_fire_variants_forward(src, tgt, model, omega, me, world=world)
    if int(tgt.owner) == me:
        return []
    min_w = min_wait_affordable(src, tgt, model, omega, me, world=world)
    if min_w is None or min_w == 0:
        return []
    prod = max(1, int(src.production))
    variants = []
    seen: set[tuple[int, int]] = set()
    for wait_N in (min_w, min_w + WAIT_BUFFER_OFFSET):
        if wait_N >= MAX_HORIZON:
            break
        budget = int(src.ships) + prod * wait_N
        target_fleet = max(MIN_FLEET_SIZE, int(tgt.ships) + 1)
        if target_fleet > budget:
            continue
        angle, eta = aim_and_eta(src, tgt, target_fleet, omega, wait_N=wait_N, world=world)
        pred_at_arrival = float(model.ships_at(int(tgt.id), wait_N + eta) or 0.0)
        cap_final = max(MIN_FLEET_SIZE, int(math.ceil(pred_at_arrival)) + 1)
        if cap_final > budget:
            continue
        if wait_N + eta + SIM_SETTLE_TURNS > MAX_HORIZON:
            continue
        # We waited N turns to accumulate src.ships + prod*N total ships.
        # USE the accumulation — emit full budget, not bare capture+1.
        # Banding dedup ((src, tgt, wait_band) key) collapses multiple
        # ship-counts at the same wait_N to one per band since cheap_delta
        # is identical for capture-success. So we pick ONE — the budget
        # variant. This:
        #   1. Uses ships we waited for (otherwise the wait is wasted).
        #   2. Leaves residue on the captured planet (budget - defenders),
        #      defending against opp recapture in 4P (the bare-capture
        #      variant left 1-ship residue → trivially recaptured).
        final_fleet = budget
        if final_fleet < MIN_FLEET_SIZE:
            continue
        key = (wait_N, final_fleet)
        if key in seen:
            continue
        seen.add(key)
        variants.append((final_fleet, wait_N, angle, eta))
    return variants


def cheap_marginal_value(src, tgt, ships: int, eta: int, world, model,
                         me: int, wait_N: int = 0) -> float:
    """Analytic Δ for Stage-1 ranking. Replaced by fast_sim in Stage-2.

    Capture: +0.05 * tgt.prod * pv_horizon(step, arrival, gamma=0.99)
    Bounce:  -0.5 * ships
    Reinforce (mine): pv-weighted loss-prevention credit if threatened
                      within eta+30, else 0.
    """
    arrival_step = wait_N + eta
    pred_owner = model.owner_at(int(tgt.id), arrival_step)
    pred_ships = float(model.ships_at(int(tgt.id), arrival_step) or 0.0)

    if pred_owner == me:
        # PI 2026-05-21 fix — gate on BASELINE_ORBITAL_SAFETY=1, pass
        # arrival_eta so an orbiting target's position at our arrival
        # is used for the threat-distance calc. Was silently scoring
        # "rotates-into-enemy-zone" captures as safe (long horizon),
        # driving fleets into easy recaptures. Default OFF preserves
        # backwards compat with sub 52882014.
        if os.environ.get("BASELINE_ORBITAL_SAFETY", "0") == "1":
            t_to_threat = model.time_to_enemy_threat(
                int(tgt.id), me, world, arrival_eta=int(arrival_step),
            )
        else:
            t_to_threat = model.time_to_enemy_threat(int(tgt.id), me, world)
        if t_to_threat is None or t_to_threat > eta + 30:
            return 0.0
        pv = pv_horizon(int(world.step), int(t_to_threat),
                        gamma=GAMMA, t_total=EPISODE_STEPS)
        return 0.05 * float(tgt.production) * float(pv)

    if ships > pred_ships:
        pv = pv_horizon(int(world.step), int(arrival_step),
                        gamma=GAMMA, t_total=EPISODE_STEPS)
        return 0.05 * float(tgt.production) * float(pv)

    return -0.5 * float(ships)


def wait_band(wait_N: int) -> int:
    """Three buckets: fire-now (0), short-wait (1..7), long-wait (>=8)."""
    if wait_N == 0:
        return 0
    return 1 if wait_N <= 7 else 2


def _source_survives_launch_legacy(
    src, ships: int, wait_N: int, world, model, me: int,
) -> bool:
    """Pre-2026-05-27 source-drain predicate. Restored verbatim because
    the harden-larger→smaller variant regressed live (sub 53083109,
    μ=842.8 vs anchor 1144-1165). Kept as the DEFAULT path; opt in to
    the hardened variant via `BASELINE_DRAIN_HARDEN=1`."""
    # Predicted-threat garrison reserve (opt-in via env var; default
    # ALPHA=0.0 → block skipped, legacy decision byte-for-byte). Fires
    # independent of in-flight ledger force so the clause catches
    # rotating-opp waves the legacy `threat_force` sum misses.
    if THREAT_RESERVE_ALPHA > 0.0:
        growth_during_wait = int(src.production) * int(wait_N)
        residue_after_launch = (
            int(src.ships) + growth_during_wait - int(ships)
        )
        if residue_after_launch < 0:
            return False
        predicted = model.predicted_threat_force(
            int(src.id), me, world, THREAT_RESERVE_WINDOW,
        )
        reserve = int(math.ceil(THREAT_RESERVE_ALPHA * predicted))
        if residue_after_launch < reserve:
            return False
    threat_eta = model.time_to_enemy_threat(int(src.id), me, world)
    if threat_eta is None:
        return True
    threat_force = sum(
        sh
        for (eta_arr, owner, sh) in model.ledger.get(int(src.id), [])
        if owner != me and eta_arr <= int(threat_eta) + WAVE_LOOKAHEAD
    )
    if threat_force <= 0:
        return True
    if int(wait_N) >= int(threat_eta):
        return False
    growth_during_wait = int(src.production) * int(wait_N)
    residue_after_launch = int(src.ships) + growth_during_wait - int(ships)
    if residue_after_launch < 0:
        return False
    growth_after_launch_to_threat = (
        int(src.production) * (int(threat_eta) - int(wait_N))
    )
    garrison_at_threat = residue_after_launch + growth_after_launch_to_threat
    return garrison_at_threat >= int(threat_force) + 1


def _source_survives_launch_hardened(
    src, ships: int, wait_N: int, world, model, me: int,
    tgt=None,
) -> bool:
    """Source-drain protection with larger→smaller hardening
    (2026-05-27, PI direction). Three extra clauses when
    `src.production > tgt.production`:

      Clause A (stockpile floor). `residue >= STOCKPILE_PROD_MULT
        × src.production` even with no inbound threat.

      Clause B (stricter margin). Under threat,
        `SAFETY_MARGIN_DRAIN × threat_force` instead of
        `threat_force + 1`.

      Clause C (potential-launch coverage). Folds 50% of the biggest
        single opp's garrison into threat_force when ledger has no
        in-flight threat but `time_to_enemy_threat` flagged a potential.

    Live A/B 2026-05-27: this variant lost 2/32 vs the legacy
    predicate (sub 53083109 μ=842.8 vs peer 1144-1165). DEFAULT OFF.
    Opt in via `BASELINE_DRAIN_HARDEN=1` once a finer tuning is found.
    """
    threat_eta = model.time_to_enemy_threat(int(src.id), me, world)
    growth_during_wait = int(src.production) * int(wait_N)
    residue_after_launch = int(src.ships) + growth_during_wait - int(ships)
    if residue_after_launch < 0:
        return False  # nonsensical sizing; guard

    is_larger_to_smaller = (
        tgt is not None
        and int(src.production) > int(tgt.production)
    )

    # Clause A: stockpile floor for larger→smaller (no threat needed).
    if is_larger_to_smaller:
        stockpile_floor = STOCKPILE_PROD_MULT * int(src.production)
        if residue_after_launch < stockpile_floor:
            return False

    if threat_eta is None:
        return True

    threat_force = sum(
        sh
        for (eta_arr, owner, sh) in model.ledger.get(int(src.id), [])
        if owner != me and eta_arr <= int(threat_eta) + WAVE_LOOKAHEAD
    )

    # Clause C: potential-launch protection for larger→smaller.
    if threat_force <= 0:
        if not is_larger_to_smaller:
            return True
        potential = _largest_opp_potential_force(src, world, me)
        if potential <= 0:
            return True
        threat_force = int(0.5 * potential)
        if threat_force <= 0:
            return True

    if int(wait_N) >= int(threat_eta):
        return False
    growth_after_launch_to_threat = (
        int(src.production) * (int(threat_eta) - int(wait_N))
    )
    garrison_at_threat = residue_after_launch + growth_after_launch_to_threat

    # Clause B: stricter margin for larger→smaller.
    if is_larger_to_smaller:
        required = int(math.ceil(SAFETY_MARGIN_DRAIN * threat_force)) + 1
    else:
        required = int(threat_force) + 1
    return garrison_at_threat >= required


def _source_survives_launch(
    src, ships: int, wait_N: int, world, model, me: int,
    tgt=None,
) -> bool:
    """Dispatch — legacy by default, hardened when
    `BASELINE_DRAIN_HARDEN=1`. Default-legacy because the hardened
    variant regressed live (sub 53083109)."""
    if os.environ.get("BASELINE_DRAIN_HARDEN", "0").strip() == "1":
        return _source_survives_launch_hardened(
            src, ships, wait_N, world, model, me, tgt=tgt,
        )
    return _source_survives_launch_legacy(
        src, ships, wait_N, world, model, me,
    )


def _largest_opp_potential_force(src, world, me: int) -> int:
    """Largest single-opp garrison currently held — a conservative
    single-opp bound on the worst-case potential launch at `src`.
    Used by the larger→smaller source-drain protection (Clause C of
    `_source_survives_launch`)."""
    best = 0
    for opp in world.planets_by_id.values():
        if int(opp.owner) == me or int(opp.owner) < 0:
            continue
        if int(opp.id) == int(src.id):
            continue
        if int(opp.ships) > best:
            best = int(opp.ships)
    return best


def _target_holdable_after_capture_legacy(
    src, tgt, ships: int, wait_N: int, eta: int, world, model, me: int,
) -> bool:
    """Pre-2026-05-27 nearest-opp predicate. Restored verbatim — the
    v2 all-opp variant shipped with sub 53083109 (μ=842.8 vs anchor
    1144-1165) was net negative. DEFAULT path; opt in to v2 via
    `BASELINE_HOLDABILITY_V2=1`."""
    if int(tgt.owner) == me:
        return True

    arrival_step = int(wait_N) + int(eta)
    if int(tgt.owner) == -1:
        tgt_def_at_arrival = int(tgt.ships)
    else:
        tgt_def_at_arrival = int(tgt.ships) + int(tgt.production) * arrival_step

    delivered = int(ships) - tgt_def_at_arrival
    if delivered < 1:
        return True

    MIN_COUNTER_SHIPS = 20
    SAFETY_MARGIN = 1.5

    orbital_safety = os.environ.get("BASELINE_ORBITAL_SAFETY", "0") == "1"
    omega = float(getattr(world, "omega", 0.0))
    use_predict = orbital_safety and omega != 0.0 and arrival_step > 0
    if use_predict:
        tgt_x, tgt_y = _position_at(tgt, omega, arrival_step)
    else:
        tgt_x, tgt_y = float(tgt.x), float(tgt.y)

    nearest_opp = None
    nearest_opp_dist = float("inf")
    for opp in world.planets_by_id.values():
        if int(opp.owner) == me or int(opp.owner) == -1:
            continue
        if int(opp.id) == int(tgt.id):
            continue
        if int(opp.ships) < MIN_COUNTER_SHIPS:
            continue
        if use_predict:
            ox, oy = _position_at(opp, omega, arrival_step)
        else:
            ox, oy = float(opp.x), float(opp.y)
        d = math.hypot(ox - tgt_x, oy - tgt_y)
        if d < nearest_opp_dist:
            nearest_opp_dist = d
            nearest_opp = opp
    if nearest_opp is None:
        return True

    nearest_us_dist = float("inf")
    for ally in world.planets_by_id.values():
        if int(ally.owner) != me:
            continue
        if int(ally.id) == int(tgt.id):
            continue
        if use_predict:
            ax, ay = _position_at(ally, omega, arrival_step)
        else:
            ax, ay = float(ally.x), float(ally.y)
        d = math.hypot(ax - tgt_x, ay - tgt_y)
        if d < nearest_us_dist:
            nearest_us_dist = d
    if nearest_us_dist <= nearest_opp_dist:
        return True

    flight = (
        nearest_opp_dist - float(nearest_opp.radius)
        - float(tgt.radius) - 0.1
    )
    if flight <= 0:
        return True
    opp_speed = fleet_speed(int(nearest_opp.ships))
    if opp_speed <= 0:
        return True
    t_op = int(math.ceil(flight / opp_speed))
    garrison_at_recapture = delivered + int(tgt.production) * t_op
    counter_force = (
        int(nearest_opp.ships)
        + int(nearest_opp.production) * (arrival_step + t_op)
    )
    if counter_force >= SAFETY_MARGIN * garrison_at_recapture + 1:
        return False
    return True


def _target_holdable_after_capture_v2(
    src, tgt, ships: int, wait_N: int, eta: int, world, model, me: int,
) -> bool:
    """v2 hold-feasibility filter — iterates EVERY opp with ships >=
    MIN_COUNTER_SHIPS, computes recapture cost via fixed-point on
    `(opp_needed, opp_speed, t_op)`, B7-style orbital fixed-point.

    Lost A/B vs the legacy predicate (sub 53083109 panel 2/32 vs
    anchor). Kept available behind `BASELINE_HOLDABILITY_V2=1` for
    future re-tuning (over-rejected too many marginal captures —
    needs lower SAFETY_MARGIN or smarter affordability check).
    """
    if int(tgt.owner) == me:
        return True

    arrival_step = int(wait_N) + int(eta)
    if int(tgt.owner) == -1:
        tgt_def_at_arrival = int(tgt.ships)
    else:
        tgt_def_at_arrival = int(tgt.ships) + int(tgt.production) * arrival_step

    delivered = int(ships) - tgt_def_at_arrival
    if delivered < 1:
        return True

    MIN_COUNTER_SHIPS = 20
    SAFETY_MARGIN = 1.5

    orbital_safety = os.environ.get("BASELINE_ORBITAL_SAFETY", "0") == "1"
    omega = float(getattr(world, "omega", 0.0))
    use_predict = orbital_safety and omega != 0.0 and arrival_step > 0
    if use_predict:
        tgt_x, tgt_y = _position_at(tgt, omega, arrival_step)
    else:
        tgt_x, tgt_y = float(tgt.x), float(tgt.y)

    # Collect threatening opps (ships >= MIN_COUNTER_SHIPS, not the
    # target, not on our team). Track per-opp position-at-arrival so
    # the inner fixed-point doesn't repeat the predict_relative call.
    opps: list = []
    for opp in world.planets_by_id.values():
        if int(opp.owner) == me or int(opp.owner) == -1:
            continue
        if int(opp.id) == int(tgt.id):
            continue
        if int(opp.ships) < MIN_COUNTER_SHIPS:
            continue
        if use_predict:
            ox, oy = _position_at(opp, omega, arrival_step)
        else:
            ox, oy = float(opp.x), float(opp.y)
        d = math.hypot(ox - tgt_x, oy - tgt_y)
        opps.append((d, opp, ox, oy))
    if not opps:
        return True

    # "Ally closer than every threatening opp" → we'd defend faster
    # than any opp could recapture; accept globally. The min-opp
    # distance bounds the check.
    min_opp_dist = min(d for d, _o, _ox, _oy in opps)
    nearest_us_dist = float("inf")
    for ally in world.planets_by_id.values():
        if int(ally.owner) != me:
            continue
        if int(ally.id) == int(tgt.id):
            continue
        if use_predict:
            ax, ay = _position_at(ally, omega, arrival_step)
        else:
            ax, ay = float(ally.x), float(ally.y)
        d = math.hypot(ax - tgt_x, ay - tgt_y)
        if d < nearest_us_dist:
            nearest_us_dist = d
    if nearest_us_dist <= min_opp_dist:
        return True

    # Per-opp feasibility loop.
    for opp_dist, opp, ox, oy in opps:
        flight = opp_dist - float(opp.radius) - float(tgt.radius) - 0.1
        if flight <= 0:
            # Adjacent — opp can land in 1 tick. Conservative reject if
            # delivered force isn't already SAFETY_MARGIN-clear.
            if int(opp.ships) >= SAFETY_MARGIN * delivered + 1:
                return False
            continue

        # Fixed-point on opp_needed:
        #   opp_speed = fleet_speed(opp_needed)
        #   t_op      = ceil(flight / opp_speed)   (B7 fixed-point if orbital)
        #   garrison  = delivered + tgt.prod * t_op
        #   opp_needed = ceil(SAFETY_MARGIN * garrison) + 1
        opp_needed = MIN_COUNTER_SHIPS
        for _ in range(3):
            opp_speed = fleet_speed(opp_needed)
            if opp_speed <= 0:
                break
            t_op = int(math.ceil(flight / opp_speed))
            if use_predict and t_op > 0:
                # B7-style fixed-point on rendezvous point.
                for _ in range(3):
                    tx_k, ty_k = _position_at(
                        tgt, omega, arrival_step + t_op,
                    )
                    dist_k = math.hypot(tx_k - ox, ty_k - oy)
                    new_t_op = int(math.ceil(dist_k / opp_speed))
                    if abs(new_t_op - t_op) <= 1:
                        t_op = new_t_op
                        break
                    t_op = new_t_op
            garrison_at_recapture = delivered + int(tgt.production) * t_op
            new_opp_needed = (
                int(math.ceil(SAFETY_MARGIN * garrison_at_recapture)) + 1
            )
            if new_opp_needed == opp_needed:
                break
            opp_needed = new_opp_needed

        # Affordability: opp's ship budget at their launch moment
        # (just after our landing — they react then).
        opp_launch_budget = (
            int(opp.ships) + int(opp.production) * arrival_step
        )
        if opp_needed <= opp_launch_budget:
            # This opp can mount a recapture that overwhelms our
            # garrison-at-recapture by SAFETY_MARGIN. Reject the
            # candidate.
            return False

    return True


def _target_holdable_after_capture(
    src, tgt, ships: int, wait_N: int, eta: int, world, model, me: int,
) -> bool:
    """Dispatch — legacy by default, v2 (all-opp + fixed-point) when
    `BASELINE_HOLDABILITY_V2=1`. Default-legacy because v2 regressed
    live (sub 53083109 panel: 2/32 wins vs anchor, Wlo=0.017)."""
    if os.environ.get("BASELINE_HOLDABILITY_V2", "0").strip() == "1":
        return _target_holdable_after_capture_v2(
            src, tgt, ships, wait_N, eta, world, model, me,
        )
    return _target_holdable_after_capture_legacy(
        src, tgt, ships, wait_N, eta, world, model, me,
    )


def _cost_parity_margin() -> float:
    """Read COST_PARITY_MARGIN from env, falling back to the default constant."""
    raw = os.environ.get("COST_PARITY_MARGIN", "")
    if not raw:
        return COST_PARITY_MARGIN_DEFAULT
    try:
        return float(raw)
    except ValueError:
        return COST_PARITY_MARGIN_DEFAULT


def _target_cost_parity_ok(
    src, tgt, ships: int, wait_N: int, eta: int, world, model, me: int,
) -> bool:
    """Reactor-cost parity filter — Part A of reactor-aware launch selection.

    Sibling to `_target_holdable_after_capture`. Where that filter asks
    "will we still own the target after opp's counter-launch?", this asks
    the strategic-cost question: "is the cheapest opp reactor cost
    materially LESS than our launch cost?" If yes the candidate is the
    first-mover trap — we pay more than opp does to take this same
    planet, even if we hold it afterward.

    Returns True (acceptable) when:
      - tgt is our own planet (reinforce, not a race),
      - the capture itself fails (delivered < 1; other filters drop),
      - no opp planet within plausible reactor range,
      - some ally is closer to tgt than every threatening opp (we'd be
        the cheap reactor; accept the launch),
      - the cheapest opp reactor still pays ≥ ships * COST_PARITY_MARGIN.

    Margin is read per-call from env (`COST_PARITY_MARGIN`) so A/B grid
    sweeps can override without rebuilding the bundle.
    """
    if int(tgt.owner) == me:
        return True

    arrival_step = int(wait_N) + int(eta)
    if int(tgt.owner) == -1:
        tgt_def_at_arrival = int(tgt.ships)
    else:
        tgt_def_at_arrival = int(tgt.ships) + int(tgt.production) * arrival_step
    delivered = int(ships) - tgt_def_at_arrival
    if delivered < 1:
        return True  # capture fails; not our concern here

    # B2 (PI 2026-05-21 / completed 2026-05-22) — orbital safety: predict
    # tgt/ally/opp positions at arrival_step when BASELINE_ORBITAL_SAFETY=1.
    # Same modeling fix as B1 (`_target_holdable_after_capture`); the
    # reactor-cost parity verdict depends on the same rotated geometry.
    orbital_safety = os.environ.get("BASELINE_ORBITAL_SAFETY", "0") == "1"
    omega = float(getattr(world, "omega", 0.0))
    use_predict = orbital_safety and omega != 0.0 and arrival_step > 0
    if use_predict:
        tgt_x, tgt_y = _position_at(tgt, omega, arrival_step)
    else:
        tgt_x, tgt_y = float(tgt.x), float(tgt.y)

    # "Are WE closer to tgt than every threatening opp?" — analogue of
    # the hold-feasibility ally-closer safety valve. If yes, we'd be
    # the cheap second-mover and the launch is positionally fine.
    nearest_us_dist = float("inf")
    for ally in world.planets_by_id.values():
        if int(ally.owner) != me:
            continue
        if int(ally.id) == int(tgt.id):
            continue
        if int(ally.id) == int(src.id):
            continue  # already committed; can't double-count
        if use_predict:
            ax, ay = _position_at(ally, omega, arrival_step)
        else:
            ax, ay = float(ally.x), float(ally.y)
        d = math.hypot(ax - tgt_x, ay - tgt_y)
        if d < nearest_us_dist:
            nearest_us_dist = d

    min_opp_reactor_cost: int | None = None
    for opp in world.planets_by_id.values():
        if int(opp.owner) == me or int(opp.owner) == -1:
            continue
        if int(opp.id) == int(tgt.id):
            continue
        if int(opp.ships) < MIN_REACTOR_SHIPS:
            continue
        if use_predict:
            ox, oy = _position_at(opp, omega, arrival_step)
        else:
            ox, oy = float(opp.x), float(opp.y)
        d = math.hypot(ox - tgt_x, oy - tgt_y)
        # Ally-closer safety valve: if some ally is strictly closer than
        # this opp, treat the launch as positionally fine (we can reach
        # tgt to defend faster than opp can reach it to recapture).
        if nearest_us_dist < d:
            return True
        flight = d - float(opp.radius) - float(tgt.radius) - 0.1
        if flight <= 0:
            continue
        # Default (LEGACY): `fleet_speed(opp.ships)` — pre-fix shape.
        # Restored after the v2 fixed-point version regressed live
        # (sub 53083109 panel 2/32 vs anchor). Opt in to v2 via
        # `BASELINE_HOLDABILITY_V2=1` (shared env with the holdability
        # filter — they were co-shipped).
        if os.environ.get("BASELINE_HOLDABILITY_V2", "0").strip() == "1":
            opp_needed = MIN_FLEET_SIZE
            for _ in range(3):
                opp_speed = fleet_speed(opp_needed)
                if opp_speed <= 0:
                    break
                opp_eta_after_landing = int(math.ceil(flight / opp_speed))
                if use_predict and opp_eta_after_landing > 0:
                    for _ in range(3):
                        tx_k, ty_k = _position_at(
                            tgt, omega, arrival_step + opp_eta_after_landing,
                        )
                        dist_k = math.hypot(tx_k - ox, ty_k - oy)
                        new_eta = int(math.ceil(dist_k / opp_speed))
                        if abs(new_eta - opp_eta_after_landing) <= 1:
                            opp_eta_after_landing = new_eta
                            break
                        opp_eta_after_landing = new_eta
                garrison_at_recapture = (
                    delivered + int(tgt.production) * opp_eta_after_landing
                )
                new_opp_needed = int(math.ceil(garrison_at_recapture)) + 1
                if new_opp_needed == opp_needed:
                    break
                opp_needed = new_opp_needed
        else:
            opp_speed = fleet_speed(int(opp.ships))
            if opp_speed <= 0:
                continue
            opp_eta_after_landing = int(math.ceil(flight / opp_speed))
            garrison_at_recapture = (
                delivered + int(tgt.production) * opp_eta_after_landing
            )
            opp_needed = int(math.ceil(garrison_at_recapture)) + 1
        opp_launch_budget = (
            int(opp.ships) + int(opp.production) * arrival_step
        )
        if opp_needed > opp_launch_budget:
            continue  # opp can't afford the reactor; skip them
        opp_needed = max(MIN_FLEET_SIZE, opp_needed)
        if min_opp_reactor_cost is None or opp_needed < min_opp_reactor_cost:
            min_opp_reactor_cost = opp_needed

    if min_opp_reactor_cost is None:
        return True  # no affordable opp reactor; safe to launch

    margin = _cost_parity_margin()
    if float(min_opp_reactor_cost) < float(ships) * margin:
        return False  # opp pays materially less than us — wasteful first-mover
    return True


def _enumerate_reactor_candidates(
    my_planets, world, model, me: int, omega: float, baseline_len: int,
):
    """Reactor candidate generator — Part B of reactor-aware launch selection.

    For each target T not owned by us that has at least one opp fleet
    in flight, propose our own launches from a nearby source sized to
    recapture T after opp lands. The chooser then ranks these alongside
    the standard fire-now / wait_then_fire candidates.

    Output shape matches `propose()`'s prerank tuples:
        (cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N).
    Capped globally at MAX_REACTOR_CANDIDATES_PER_TURN, top-K by
    cheap_delta. Per-target source enumeration is capped at
    REACTOR_TOP_K_SOURCES_PER_TARGET closest.

    Skips:
      - targets with no opp in-flight fleets,
      - targets that opp's fleet does NOT actually capture (post-landing
        owner stays neutral or stays ours — the existing pipeline already
        handles those cases via fire-now / wait_then_fire),
      - sources that can't afford the post-landing recapture even with
        wait accumulation.
    """
    if not my_planets:
        return []

    # Identify targets with opp in-flight fleets via the ledger. Keys
    # are planet ids; values are lists of (eta, owner, ships).
    target_ids_with_opp: list[int] = []
    for tgt_id, entries in model.ledger.items():
        for (eta_arr, owner, ships_arr) in entries:
            if owner == me or owner == -1:
                continue
            if ships_arr <= 0:
                continue
            target_ids_with_opp.append(int(tgt_id))
            break

    if not target_ids_with_opp:
        return []

    candidates: list = []
    for tgt_id in target_ids_with_opp:
        tgt = world.planets_by_id.get(tgt_id)
        if tgt is None:
            continue
        if int(tgt.owner) == me:
            continue  # defensive reinforce handled elsewhere

        # Latest opp arrival to this target. Use the post-landing owner
        # to decide if a reactor is needed.
        opp_etas = [
            int(eta_arr)
            for (eta_arr, owner, ships_arr) in model.ledger.get(tgt_id, [])
            if owner != me and owner != -1 and ships_arr > 0
        ]
        if not opp_etas:
            continue
        max_opp_eta = max(opp_etas)

        post_owner = model.owner_at(int(tgt_id), max_opp_eta + 1)
        if post_owner is None:
            continue  # beyond timeline horizon
        if post_owner == me:
            continue  # we end up holding; no reactor needed
        if int(post_owner) == -1:
            # Opp's fleet bounces. Existing wait_then_fire / fire-now
            # variants handle the neutral capture; skip to avoid
            # producing duplicate candidates.
            continue

        # Top-K closest sources by straight-line distance.
        src_with_dist: list = []
        for src in my_planets:
            if int(src.ships) < MIN_FLEET_SIZE:
                continue
            if int(src.id) == int(tgt_id):
                continue
            d = math.hypot(
                float(src.x) - float(tgt.x),
                float(src.y) - float(tgt.y),
            )
            src_with_dist.append((d, src))
        if not src_with_dist:
            continue
        src_with_dist.sort(key=lambda x: x[0])
        src_with_dist = src_with_dist[:REACTOR_TOP_K_SOURCES_PER_TARGET]

        for _d, src in src_with_dist:
            # Conservative natural-eta probe at MIN_FLEET_SIZE (slowest);
            # actual launch will be larger / faster, narrowing the gap.
            _angle_probe, eta_probe = aim_and_eta(
                src, tgt, MIN_FLEET_SIZE, omega, wait_N=0, world=world,
            )
            desired_arrival = max_opp_eta + 1
            wait_N = max(0, desired_arrival - int(eta_probe))
            if wait_N + int(eta_probe) + SIM_SETTLE_TURNS > MAX_HORIZON:
                continue
            arrival_step = wait_N + int(eta_probe)
            arrival_owner = model.owner_at(int(tgt_id), arrival_step)
            if arrival_owner == me:
                continue
            arrival_ships = float(
                model.ships_at(int(tgt_id), arrival_step) or 0.0
            )
            needed = max(MIN_FLEET_SIZE, int(math.ceil(arrival_ships)) + 1)
            budget = int(src.ships) + int(src.production) * wait_N
            if needed > budget:
                continue
            # Recompute aim / eta at the actual ship count
            angle, eta = aim_and_eta(src, tgt, needed, omega, wait_N=wait_N, world=world)
            if wait_N + int(eta) + SIM_SETTLE_TURNS > MAX_HORIZON:
                continue
            # Re-sample timeline at the refined arrival step
            refined_arrival = wait_N + int(eta)
            refined_owner = model.owner_at(int(tgt_id), refined_arrival)
            if refined_owner == me:
                continue
            refined_ships = float(
                model.ships_at(int(tgt_id), refined_arrival) or 0.0
            )
            needed = max(MIN_FLEET_SIZE, int(math.ceil(refined_ships)) + 1)
            if needed > budget:
                continue
            horizon = max(int(eta) + SIM_SETTLE_TURNS, MIN_HORIZON)
            if horizon >= baseline_len:
                continue
            cheap = cheap_marginal_value(
                src, tgt, needed, int(eta), world, model, me, wait_N=wait_N,
            )
            if cheap <= CHEAP_REJECT_THRESHOLD:
                continue
            candidates.append(
                (cheap, src, tgt, needed, float(angle), int(eta), horizon, wait_N)
            )

    candidates.sort(key=lambda c: -c[0])
    return candidates[:MAX_REACTOR_CANDIDATES_PER_TURN]


def propose(my_planets, target_pool, world, model, me: int,
            omega: float, baseline_len: int):
    """Build the pre-rank list of candidates, then dedup by
    (src_id, tgt_id, wait_band) keeping the top cheap-Δ per bucket.

    Returns: list of tuples
        (cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N)
    sorted by cheap_delta descending.
    """
    prerank = []
    for src in my_planets:
        if int(src.ships) < MIN_FLEET_SIZE:
            continue
        for tgt in nearest_k(target_pool, src, NUM_TARGETS_PER_SOURCE):
            if int(tgt.id) == int(src.id):
                continue

            for ships in enumerate_ship_counts(src, tgt, model, omega, me, world):
                if ships < MIN_FLEET_SIZE or ships > int(src.ships):
                    continue
                angle, eta = aim_and_eta(src, tgt, ships, omega, world=world)
                horizon = max(eta + SIM_SETTLE_TURNS, MIN_HORIZON)
                if horizon >= baseline_len:
                    horizon = baseline_len - 1
                cheap = cheap_marginal_value(
                    src, tgt, ships, eta, world, model, me, wait_N=0,
                )
                if cheap > CHEAP_REJECT_THRESHOLD:
                    prerank.append(
                        (cheap, src, tgt, ships, angle, eta, horizon, 0)
                    )

            for w_ships, w_wait, w_angle, w_eta in wait_then_fire_variants(
                src, tgt, model, omega, me, world=world,
            ):
                w_horizon = max(w_wait + w_eta + SIM_SETTLE_TURNS, MIN_HORIZON)
                if w_horizon >= baseline_len:
                    continue
                w_cheap = cheap_marginal_value(
                    src, tgt, w_ships, w_eta, world, model, me, wait_N=w_wait,
                )
                if w_cheap > CHEAP_REJECT_THRESHOLD:
                    prerank.append(
                        (w_cheap, src, tgt, w_ships, w_angle, w_eta,
                         w_horizon, w_wait)
                    )

    # Reactor candidate generator (Part B of reactor-aware launch selection,
    # 2026-05-19 PM). For each opp fleet in flight to a non-our target,
    # propose our own launches sized to recapture after opp lands. These
    # extend the standard prerank list and participate in the existing
    # (src, tgt, wait_band) dedup. The chooser then scores them alongside
    # fire-now / wait_then_fire candidates. Opt out via
    # PROPOSER_REACTOR_CANDIDATES=off for ablation A/B.
    if os.environ.get("PROPOSER_REACTOR_CANDIDATES", "").strip().lower() != "off":
        prerank.extend(_enumerate_reactor_candidates(
            my_planets, world, model, me, omega, baseline_len,
        ))

    best_per_band: dict[tuple[int, int, int], tuple] = {}
    for entry in prerank:
        cheap, src, tgt, _ships, _angle, _eta, _horizon, w = entry
        key = (int(src.id), int(tgt.id), wait_band(int(w)))
        prev = best_per_band.get(key)
        if prev is None or cheap > prev[0]:
            best_per_band[key] = entry

    deduped = list(best_per_band.values())

    # Trajectory admissibility filter (opt-in via env var, default off).
    # Drops candidates whose straight-line trajectory hits the sun, OOB,
    # or a comet/wrong-planet before reaching the intended target — all
    # are deterministic-zero-success launches the chooser would otherwise
    # waste rollout time on (and the existing leaf-value heads don't
    # always penalise them). Uses lib.trajectory.predict_fleet_fate,
    # which mirrors the engine's swept-pair / point-to-segment-distance
    # rules. Filter runs ONLY on fire-now candidates (wait_N==0); wait-N
    # variants have time-shifted geometry the static fate-predictor
    # doesn't model.
    #
    # Origin: PI critique 2026-05-17 PM. Full design at
    # knowledge-base/concepts/trajectory-first-architecture.md;
    # in-chooser variant (chooser_trajectory.py) lost A/B vs v15
    # because it discarded strategic depth; this proposer-side filter
    # keeps the K-step rollout and only PRUNES doomed candidates.
    # Default-on as of 2026-05-17 after the SUN_SAFETY=0 fix in
    # lib.trajectory closed the false-reject leak: Option 1 prefilter
    # A/B vs v15 went from 36/64 (56.2pct) pre-fix to 42/64 (65.6pct)
    # post-fix — at parity-or-better with composite_a2 alone, plus
    # deterministic 0pct sun/oob/comet failures. Set
    # PROPOSER_TRAJECTORY_FILTER=off to bypass.
    if os.environ.get("PROPOSER_TRAJECTORY_FILTER", "").strip().lower() != "off":
        filtered: list = []
        for entry in deduped:
            _cheap, src, tgt, ships, angle, eta, _horizon, w = entry
            if int(w) != 0:
                # Wait-then-fire: trajectory geometry depends on the
                # launch-time orbital state; the static fate-predictor
                # would mis-classify. Pass through unfiltered.
                filtered.append(entry)
                continue
            fate = predict_fleet_fate(src, tgt, float(angle), int(ships), world)
            if fate.outcome != "target":
                continue  # sun / oob / hits wrong planet / timeout — drop
            # Target reached. If it's a comet, also gate on lifetime.
            if int(tgt.id) in world.comet_ids:
                life = comet_remaining_lifetime(int(tgt.id), world)
                if life is None or life <= int(fate.step):
                    continue
            filtered.append(entry)
        deduped = filtered

    # Bug #4 fix (2026-05-18 PM): drop candidates whose launch would
    # leave the SOURCE vulnerable to a known inbound enemy threat
    # before our garrison + production accrual can defend. The
    # chooser's leaf rollout (horizon 25) doesn't see threats landing
    # 30+ ticks later, so the chooser drains exposed sources. This
    # pre-cut catches that class of decision before the chooser even
    # scores the candidate. Opt out via PROPOSER_DRAIN_FILTER=off to
    # A/B against the pre-fix breadth.
    if os.environ.get("PROPOSER_DRAIN_FILTER", "").strip().lower() != "off":
        deduped = [
            entry for entry in deduped
            if _source_survives_launch(
                entry[1],         # src
                int(entry[3]),    # ships
                int(entry[7]),    # wait_N
                world, model, me,
                tgt=entry[2],     # tgt — enables larger→smaller protection
            )
        ]

    # Tier 2 hold-feasibility filter (2026-05-18 PM): drop candidates
    # whose captured target would be recaptured by a nearby strong opp
    # planet before our garrison + production accrual can defend. The
    # chooser's rollout (horizon 25) misses long-distance captures whose
    # arrival lands near/past horizon, so the leaf state under-credits
    # opp counter. This pre-cut catches the wasted-ships pattern PI
    # observed in live games. Opt out via PROPOSER_HOLD_FEASIBILITY=off.
    if os.environ.get("PROPOSER_HOLD_FEASIBILITY", "").strip().lower() != "off":
        deduped = [
            entry for entry in deduped
            if _target_holdable_after_capture(
                entry[1],  # src
                entry[2],  # tgt
                int(entry[3]),  # ships
                int(entry[7]),  # wait_N
                int(entry[5]),  # eta
                world, model, me,
            )
        ]

    # Cost-parity filter (Part A of reactor-aware launch selection,
    # 2026-05-19 PM). Drops candidate launches where the cheapest opp
    # reactor pays materially fewer ships than we do. Holdability
    # (above) and cost-parity ask different questions; both can drop
    # the same candidate or one can drop what the other accepts. Opt
    # out via PROPOSER_COST_PARITY=off for ablation A/B.
    if os.environ.get("PROPOSER_COST_PARITY", "").strip().lower() != "off":
        deduped = [
            entry for entry in deduped
            if _target_cost_parity_ok(
                entry[1],  # src
                entry[2],  # tgt
                int(entry[3]),  # ships
                int(entry[7]),  # wait_N
                int(entry[5]),  # eta
                world, model, me,
            )
        ]

    deduped.sort(key=lambda e: -e[0])
    return deduped

# === inlined: agents/baseline/value.py ===
"""Leaf value function: F1 + F2 favor with PV-discounted production.

F1 = my_ships - opp_ships_agg          (in-flight + on-planet)
F2 = (my_prod - opp_prod_agg) * pv     (pv = pv_horizon discount)

PV-discount keeps F2 on a comparable scale to F1; without it the future-
production term over-weights captures by ~100x in late game and the
chooser stops valuing ship preservation. opp aggregation is max-of-opps
in 2P (unchanged from baseline) and weighted-sum-of-opps in 4P
(weakest opp 1.5x).

A2 (4P weakness exploitation) derives from
romantamrazov/orbit-star-wars-lb-max-1224 (peak LB μ=1224, +109 above
our v15 ceiling).

  - 4P: 1.5x bias on the WEAKEST opponent's contribution; other opps
    unweighted. Biases leaf valuation toward states that further
    weaken (or eliminate) them.
  - Elimination bonus: +55 when weakest's strength (ships + 15*prod)
    <= 110 AND my_strength >= 0.9 * weakest's (only fire when WE can
    finish — no elim-then-die bias). 4P only.

History — 2P bias was tested and rolled back: a uniform 1.25x
multiplier on the single opp regressed h2h vs v15 in 2P (25/64,
39.1%, Wlo=0.281, Whi=0.513 INCONCLUSIVE) because v15 is well-tuned
and biasing the chooser toward attacks degrades its calibration.
The "weakness exploitation" thesis is 4P-specific (per-weakest, not
uniform); the 2P path is unchanged from the original baseline.

Opt-in alternative head: `BASELINE_VALUE_HEAD=composite` switches the
chooser to `lib.value_heads.composite_capture_value` (waste +
capture-aware per-fleet credit). 2P-only — composite does not
distinguish opp identity in 4P. Default remains `favor` with A2.
"""


import math
import os

# from lib.scoring import pv_horizon  # inlined by bundle_agent.py

EPISODE_STEPS = 500
DEFAULT_GAMMA = 0.99

ELIMINATION_BONUS = 55.0
WEAK_ENEMY_THRESHOLD = 110.0
WEAKEST_ENEMY_MULT_4P = 1.5
ELIMINATION_GATE_RATIO = 0.9
STRENGTH_PROD_WEIGHT = 15.0

# Spatial leaf params (favor_hybrid_spatial only).
# Idle-trajectory audit 2026-05-17 on submission 52754310 (mu=1271.8)
# showed 43.8% of our ship-turns were on planets >50 units from any
# non-our planet. Spatial term rewards positioning ships near
# capturable targets so the chooser naturally drains rear/isolated
# garrisons forward.
SPATIAL_WEIGHT = float(os.environ.get("BASELINE_SPATIAL_WEIGHT", "0.5"))
SPATIAL_DECAY = float(os.environ.get("BASELINE_SPATIAL_DECAY", "30.0"))


def _read(obs, attr, default):
    if hasattr(obs, attr):
        return getattr(obs, attr)
    return obs.get(attr, default) if isinstance(obs, dict) else default


def favor(obs, me: int, num_seats: int = 2, gamma: float = DEFAULT_GAMMA) -> float:
    planets = _read(obs, "planets", []) or []
    fleets = _read(obs, "fleets", []) or []
    step = int(_read(obs, "step", 0))

    ships_by_owner: dict[int, float] = {}
    prod_by_owner: dict[int, float] = {}
    for p in planets:
        owner = int(p[1])
        if owner < 0:
            continue
        ships_by_owner[owner] = ships_by_owner.get(owner, 0.0) + float(p[5])
        prod_by_owner[owner] = prod_by_owner.get(owner, 0.0) + float(p[6])
    for f in fleets:
        owner = int(f[1])
        if owner < 0:
            continue
        ships_by_owner[owner] = ships_by_owner.get(owner, 0.0) + float(f[6])

    my_ships = ships_by_owner.get(me, 0.0)
    my_prod = prod_by_owner.get(me, 0.0)

    opps = sorted(
        o for o in (set(ships_by_owner) | set(prod_by_owner))
        if o != me and o >= 0
    )

    elim_bonus = 0.0
    if num_seats <= 2 or len(opps) < 2:
        # 2P (or degenerate <=1 opp survives): UNCHANGED from baseline —
        # max-of-opps, no bias, no bonus. The 2P uniform bias was tested
        # and rolled back (regresses vs v15).
        opp_ships = max((ships_by_owner.get(o, 0.0) for o in opps), default=0.0)
        opp_prod = max((prod_by_owner.get(o, 0.0) for o in opps), default=0.0)
    else:
        # 4P: weighted sum (weakest 1.5x) + elim bonus when we can finish.
        opp_strengths = {
            o: ships_by_owner.get(o, 0.0)
               + prod_by_owner.get(o, 0.0) * STRENGTH_PROD_WEIGHT
            for o in opps
        }
        weakest = min(opps, key=lambda o: opp_strengths[o])
        weakest_str = opp_strengths[weakest]
        opp_ships = sum(
            ships_by_owner.get(o, 0.0)
            * (WEAKEST_ENEMY_MULT_4P if o == weakest else 1.0)
            for o in opps
        )
        opp_prod = sum(
            prod_by_owner.get(o, 0.0)
            * (WEAKEST_ENEMY_MULT_4P if o == weakest else 1.0)
            for o in opps
        )
        my_strength = my_ships + my_prod * STRENGTH_PROD_WEIGHT
        if (weakest_str <= WEAK_ENEMY_THRESHOLD
                and my_strength >= ELIMINATION_GATE_RATIO * weakest_str):
            elim_bonus = ELIMINATION_BONUS

    pv = pv_horizon(step, 0, gamma=gamma, t_total=EPISODE_STEPS)
    return (my_ships - opp_ships) + (my_prod - opp_prod) * pv + elim_bonus


def favor_composite(obs, me: int, num_seats: int = 2,
                    gamma: float = DEFAULT_GAMMA) -> float:
    """`composite_capture_value` adapted to the (obs, me, num_seats, gamma)
    signature `chooser` expects. `gamma` is intentionally ignored —
    composite uses linear time-remaining weighting instead of γ-discount.
    `num_seats` is ignored — composite doesn't differentiate opps.

    Prior live evidence (iter_v1 sub 52661990, 2026-05-14):
    composite head on the v7_0 chooser → ladder μ 1034.7 (vs v15 1108.4).
    Wire only as an opt-in A/B; do NOT default this on. The clean
    baseline value is `favor` (with A2 4P-weakness exploitation).
    """
    # from lib.value_heads import composite_capture_value  # inlined by bundle_agent.py
    return composite_capture_value(obs, me)


def _positional_ship_value(obs, me: int) -> float:
    """Sum over my ships (on-planet + in-flight) of
    1.0 / (1.0 + d_min / SPATIAL_DECAY), where d_min = distance to
    nearest non-our planet. Value ranges 0..1 per ship:
    1.0 when adjacent (d=0), 0.5 at d=SPATIAL_DECAY, ~0.2 at d=120.

    Returns 0.0 if no non-our planet remains (degenerate end-state).
    """
    planets = _read(obs, "planets", []) or []
    fleets = _read(obs, "fleets", []) or []
    non_our = [(float(p[2]), float(p[3])) for p in planets if int(p[1]) != me]
    if not non_our:
        return 0.0
    total = 0.0
    for p in planets:
        if int(p[1]) != me:
            continue
        x, y = float(p[2]), float(p[3])
        d_min = min(math.hypot(x - tx, y - ty) for tx, ty in non_our)
        weight = 1.0 / (1.0 + d_min / SPATIAL_DECAY)
        total += float(p[5]) * weight
    for f in fleets:
        if int(f[1]) != me:
            continue
        x, y = float(f[2]), float(f[3])
        d_min = min(math.hypot(x - tx, y - ty) for tx, ty in non_our)
        weight = 1.0 / (1.0 + d_min / SPATIAL_DECAY)
        total += float(f[6]) * weight
    return total


def favor_hybrid_spatial(obs, me: int, num_seats: int = 2,
                         gamma: float = DEFAULT_GAMMA) -> float:
    """favor_hybrid + positional pull toward non-our planets (2P only).

    Layered on top of the validated hybrid head (composite in 2P,
    A2-favor in 4P). The spatial term is applied ONLY in 2P games —
    in 4P, the A2 weakness-exploitation already biases toward the
    weakest opp's positions, and the bv33jlzwj A/B (3/32 first-place,
    max=1503ms) showed spatial regresses 4P substantially. 2P-only
    keeps the validated A2-4P path identical to favor_hybrid.

    The spatial term is purely additive — when SPATIAL_WEIGHT=0 or
    num_seats > 2 it equals favor_hybrid exactly.
    """
    base = favor_hybrid(obs, me, num_seats, gamma)
    if SPATIAL_WEIGHT == 0.0 or num_seats > 2:
        return base
    return base + SPATIAL_WEIGHT * _positional_ship_value(obs, me)


def favor_hybrid(obs, me: int, num_seats: int = 2,
                 gamma: float = DEFAULT_GAMMA) -> float:
    """2P uses composite (waste-aware, validated by audit-workflow A/B:
    93.8% vs v9_scavenge, 67.2% vs v15). 4P uses `favor` with A2
    4P-weakness exploitation. Domains are disjoint by construction —
    composite has no 4P opp aggregation (`composite-value-head-2p-only.md`
    flag), and A2's per-weakest multiplier + elim bonus only fire when
    num_seats > 2.
    """
    if num_seats <= 2:
        return favor_composite(obs, me, num_seats, gamma)
    return favor(obs, me, num_seats, gamma)


def select_favor_fn():
    """Pick the leaf value function.

    Env var `BASELINE_VALUE_HEAD`:
      - unset / anything else -> `favor` (default, v15 baseline + A2 4P).
      - "composite"           -> `favor_composite` (2P waste-aware,
                                  composite_capture_value head).
      - "hybrid"              -> `favor_hybrid` (composite in 2P,
                                  A2-favor in 4P).

    The chooser uses the same function for both `build_idle_baseline` and
    `score_action` so the Δ stays well-defined.
    """
    choice = os.environ.get("BASELINE_VALUE_HEAD", "").strip().lower()
    if choice == "composite":
        return favor_composite
    if choice == "hybrid":
        return favor_hybrid
    if choice == "hybrid_spatial":
        return favor_hybrid_spatial
    return favor

# === inlined: agents/baseline/chooser.py ===
"""Chooser: reactive idle baseline + per-candidate Δ, emit greedy non-dogpile.

Pipeline:
  baseline[h] = favor at horizon h with me idle + opp reactive
  candidate Δ = favor(me_action @ wait_N + opp reactive) - baseline[h]
  emit       = candidates with Δ>0, greedy by Δ desc,
               1 launch per source AND 1 per target per turn.
               wait_N>0 winners RESERVE source+target but emit nothing.

Opp seats play lib.opp_model.lite_greedy_policy reactively inside every
rollout (not a precomputed trajectory), so my captures trigger opp
counter-launches and fragile leaves are correctly penalised.
"""


import os
import time

# from lib.fast_sim import clone as fs_clone  # inlined by bundle_agent.py
fs_clone = clone
# from lib.fast_sim import step as fs_step  # inlined by bundle_agent.py
fs_step = step
# from lib.opp_model import lite_greedy_policy, top_tier_mirror_policy  # inlined by bundle_agent.py

# from agents.baseline.value import select_favor_fn  # inlined by bundle_agent.py

WALLCLOCK_BUDGET_MS = 600.0
N_VALIDATE = 60
PER_CANDIDATE_SAFETY = 1.5
RESERVED_OVERHEAD_MS = 50.0


def _select_opp_policy():
    """Tier 3 (2026-05-18 PM): asymmetric opp model selection.

    BASELINE_OPP_TIER env var:
      - "0" or unset → lite_greedy_policy (default, ~1-2ms/call).
      - "1" → top_tier_mirror_policy (~5-10ms/call; ladder-realistic
              opp using v3.5.1 aggressive snipe pipeline). Bench gate
              FIRST before A/B — per-call cost is 5-10× lite_greedy.

    Per-call selection (not cached at import time) so env-var overrides
    inside test fixtures take effect without re-importing the module.
    """
    return (
        top_tier_mirror_policy
        if os.environ.get("BASELINE_OPP_TIER", "0").strip() == "1"
        else lite_greedy_policy
    )


def opp_actions_for_snap(snap, me: int, num_seats: int) -> list[list]:
    """One reactive opp action set per non-me seat. Opp policy is
    selected via BASELINE_OPP_TIER — see `_select_opp_policy`."""
    opp_policy = _select_opp_policy()
    actions: list[list] = [[] for _ in range(num_seats)]
    for opp_id in range(num_seats):
        if opp_id == me:
            continue
        try:
            actions[opp_id] = opp_policy(snap.state[opp_id].observation) or []
        except Exception:
            actions[opp_id] = []
    return actions


def build_idle_baseline(snap_base, me: int, num_seats: int,
                        max_horizon: int, gamma: float) -> list[float]:
    """favor at every horizon 0..max_horizon under (me-idle, opp-reactive)."""
    favor_fn = select_favor_fn()
    snap = fs_clone(snap_base)
    out = [favor_fn(snap.state[me].observation, me, num_seats, gamma=gamma)]
    for _ in range(max_horizon):
        if snap.fake_env.done:
            out.append(out[-1])
            continue
        actions = opp_actions_for_snap(snap, me, num_seats)
        snap = fs_step(snap, actions, in_place=True)
        out.append(favor_fn(snap.state[me].observation, me, num_seats, gamma=gamma))
    return out


def score_action(snap_base, me: int, num_seats: int,
                 src_id: int, angle: float, ships: int,
                 horizon: int, baseline_favors: list[float],
                 wait_N: int, gamma: float) -> float:
    """Δ favor at horizon = leaf(my_action@wait_N) − baseline."""
    favor_fn = select_favor_fn()
    snap = fs_clone(snap_base)
    for step_i in range(horizon):
        if snap.fake_env.done:
            break
        actions = opp_actions_for_snap(snap, me, num_seats)
        if step_i == int(wait_N):
            actions[me] = [[int(src_id), float(angle), int(ships)]]
        snap = fs_step(snap, actions, in_place=True)
    leaf = favor_fn(snap.state[me].observation, me, num_seats, gamma=gamma)
    return leaf - baseline_favors[horizon]


def affordable_validate_cap(snap_base, me: int, num_seats: int,
                            max_horizon: int, wallclock_ms: float,
                            min_horizon: int, gamma: float,
                            ) -> tuple[int, float]:
    """Probe per-step + per-leaf cost on the current board, derive a
    safe candidate cap and the per-candidate cost estimate.

    Returns `(cap, per_cand_ms)`. `cap` is bounded below by 8. The
    `per_cand_ms` value is used by `choose()` to pre-bail before
    entering a candidate that would push past the deadline.

    Probing per-leaf cost matters because the leaf eval cost varies
    by ~50x between value heads (favor ~100µs vs composite_capture_value
    ~2-5ms — it builds a World + ray-casts every fleet). Without the
    leaf probe the cap stayed sized for favor and composite blew the
    1000ms env budget on heavy turns (max 1292ms vs v15 / v9_scavenge,
    2026-05-17 A/B).
    """
    favor_fn = select_favor_fn()
    t0 = time.perf_counter()
    probe = fs_clone(snap_base)
    probe = fs_step(probe, [[] for _ in range(num_seats)], in_place=True)
    per_step_ms = max(0.05, (time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    favor_fn(probe.state[me].observation, me, num_seats, gamma=gamma)
    per_leaf_ms = max(0.05, (time.perf_counter() - t0) * 1000.0)

    avg_K = (min_horizon + max_horizon) / 2.0
    per_cand_ms = (per_step_ms * avg_K + per_leaf_ms) * PER_CANDIDATE_SAFETY
    budget = wallclock_ms - RESERVED_OVERHEAD_MS
    cap = max(8, int(budget / per_cand_ms))
    return cap, per_cand_ms


def choose(snap_base, prerank, baseline_favors: list[float],
           me: int, num_seats: int, wallclock_ms: float,
           min_horizon: int, max_horizon: int, gamma: float,
           world=None,
           reserved_srcs: set[int] | None = None,
           reserved_for_new_commits: set[int] | None = None,
           ) -> tuple[list[list], list[dict]]:
    """Validate top candidates with fast_sim, emit greedy non-dogpile moves.

    Returns `(moves, commits)`. See `chooser_trajectory.choose_trajectory`
    for the full ledger-aware contract; this is the parallel composite
    implementation (default chooser is trajectory).
    """
    if reserved_srcs is None:
        reserved_srcs = set()
    if reserved_for_new_commits is None:
        reserved_for_new_commits = reserved_srcs
    if not prerank:
        return [], []

    n_aff, per_cand_ms = affordable_validate_cap(
        snap_base, me, num_seats, max_horizon, wallclock_ms,
        min_horizon, gamma,
    )
    top = prerank[: min(N_VALIDATE, n_aff)]

    deadline = time.perf_counter() + wallclock_ms / 1000.0
    # Pre-bail headroom: don't ENTER a candidate that would push us past
    # the deadline. score_action is uninterruptible (runs the full K-step
    # rollout once entered), so checking AT the deadline is too late.
    # Closes the long-tail max-turn-ms overrun seen in the 2026-05-17 A/B.
    safe_deadline = deadline - (per_cand_ms / 1000.0)
    validated: list[tuple] = []
    for _cheap, src, tgt, ships, angle, _eta, horizon, wait_N in top:
        if time.perf_counter() > safe_deadline:
            break
        sid_ = int(src.id)
        if int(wait_N) > 0:
            if sid_ in reserved_for_new_commits:
                continue
        else:
            if sid_ in reserved_srcs:
                continue
        delta = score_action(
            snap_base, me, num_seats,
            int(src.id), float(angle), int(ships),
            int(horizon), baseline_favors, int(wait_N), gamma,
        )
        if delta > 0:
            validated.append((delta, src, tgt, ships, angle, wait_N))

    if not validated:
        return [], []

    validated.sort(key=lambda c: -c[0])
    used_srcs: set[int] = set()
    used_tgts: set[int] = set()
    moves: list[list] = []
    commits: list[dict] = []
    commit_step = int(world.step) if world is not None else 0
    for _delta, src, tgt, ships, angle, wait_N in validated:
        sid, tid = int(src.id), int(tgt.id)
        if sid in used_srcs or tid in used_tgts:
            continue
        used_srcs.add(sid)
        used_tgts.add(tid)
        if int(wait_N) == 0:
            moves.append([sid, float(angle), int(ships)])
        else:
            commits.append({
                "src_id": sid,
                "tgt_id": tid,
                "ships_planned": int(ships),
                "angle_original": float(angle),
                "wait_remaining": int(wait_N),
                "commit_step": commit_step,
            })
    return moves, commits

# === inlined: agents/baseline/chooser_roi.py ===
"""chooser_roi — ROI-prior + opp-modifier chooser.

Architectural pivot (2026-05-19, PI-directed): replace the trajectory rollout
foundation with a closed-form ROI prior + thin opp-vulnerability posterior.
Dispatched via BASELINE_CHOOSER=roi from agents/baseline/main.py.

Pipeline per turn:
  1. solo_roi(src, tgt, ships, eta, wait_N) computes a closed-form ROI per
     proposer candidate using lib/scoring + lib/world_model primitives.
  2. coalition_roi enumerates N-way joint launches per target via merged
     arrival-ledger walk; emit if joint > max(solo) + slack.
  3. (Phase 4, pending) opp_modifier_check scans emit set for exposed
     sources and downsizes/drops candidates the opp would profitably
     counter-target.

Current implementation: Phase 3 (solo_roi + 2..4-leg coalitions).

See /root/.claude/plans/okay-we-can-do-elegant-lampson.md.
"""


import math
import os
import time
from itertools import combinations

# from lib.fleet import speed as fleet_speed  # inlined by bundle_agent.py
fleet_speed = speed
# Single-line imports — the bundler's line-by-line regex chokes on
# multi-line parenthesised imports (friction
# `bundler-modular-agent-namespace-access-breaks-bundle`, see
# agents/baseline/main.py:71-76).
# from lib.scoring import T_TOTAL_DEFAULT  # inlined by bundle_agent.py
# from lib.scoring import expected_hold  # inlined by bundle_agent.py
# from lib.scoring import margin_multiplier  # inlined by bundle_agent.py
# from lib.scoring import pv_horizon  # inlined by bundle_agent.py
# Tier 2 imports: fast_sim rollout posterior for top-K candidates.
# from lib.fast_sim import clone as fs_clone  # inlined by bundle_agent.py
fs_clone = clone
# from lib.fast_sim import step as fs_step  # inlined by bundle_agent.py
fs_step = step
# from lib.fast_sim import delta_us_minus_them as fs_delta  # inlined by bundle_agent.py
fs_delta = delta_us_minus_them
# from lib.opp_model import lite_greedy_policy  # inlined by bundle_agent.py


# Cost coefficients — to be calibrated via bench in Phase 6. Initial
# values are deliberately conservative so the ROI prior favours
# small-cost moves over speculative captures while we tune.
SHIP_COST_COEF: float = float(os.environ.get("ROI_SHIP_COST", "0.05"))
WAIT_COST_COEF: float = float(os.environ.get("ROI_WAIT_COST", "0.5"))

# Coalition-only knobs.
COALITION_MAX_SOURCES: int = int(os.environ.get("ROI_COALITION_MAX", "4"))
# Coalition wins only if its ROI exceeds the best constituent solo by
# at least this slack. Prevents 3-source over-firing on targets a
# 1-source solo could handle.
COALITION_SLACK: float = float(os.environ.get("ROI_COALITION_SLACK", "1.0"))
# Per-leg residue reserve when re-enumerating coalition seeds (solo
# candidates that the proposer dropped because cap > budget). The
# leg sends src.ships - MIN_COALITION_RESIDUE; smaller residues are
# caught by _source_survives_launch downstream.
MIN_COALITION_RESIDUE: int = int(os.environ.get("ROI_MIN_RESIDUE", "5"))
MIN_LEG_SHIPS: int = 2
# Minimum residue on solo emits — prevents draining a source to zero.
# Mirror of MIN_COALITION_RESIDUE for the solo path. Proposer dedup
# typically picks the budget-sized variant, which would drain the
# source; choose_roi downsizes to honor this floor.
MIN_SOLO_RESIDUE: int = int(os.environ.get("ROI_MIN_SOLO_RESIDUE", "5"))

# Source-vulnerability knobs (Phase 4). Folded into solo/coalition
# ROI so the comparison sees the true cost of exposing a source.
# Disable with ROI_OPP_MODIFIER=off to fall back to Phase 3 behaviour.
OPP_MODIFIER_ENABLED: bool = (
    os.environ.get("ROI_OPP_MODIFIER", "on").strip().lower() != "off"
)
# Don't consider opp planets with fewer than this many ships as a
# counter-attack threat. Matches the proposer's hold-feasibility
# MIN_COUNTER_SHIPS (proposer.py:446) for symmetry.
VULN_MIN_OPP_SHIPS: int = 20

# Endgame elimination bonus: when capturing this target removes the
# LAST planet owned by a non-me, non-neutral player, the strategic
# value is "uncontested production for the rest of the game" — far
# beyond the per-tick PV math. Bonus = our_total_prod × pv_full,
# representing the unopposed advantage we get after elimination.
ENDGAME_BONUS_ENABLED: bool = (
    os.environ.get("ROI_ENDGAME_BONUS", "on").strip().lower() != "off"
)

# Defensive-coalition post-pass: after solos and attack-coalitions
# emit, for each exposed source (vuln_loss > 0) try adding an ally
# reinforce leg. If the reinforcement neutralises opp's counter AND
# the joint ROI (attack + reinforce) beats the bare attack, emit the
# reinforce as an additional move. Disable with
# ROI_DEFENSIVE_COALITION=off.
DEFENSIVE_COALITION_ENABLED: bool = (
    os.environ.get("ROI_DEFENSIVE_COALITION", "on").strip().lower() != "off"
)

# Tier 2 — forward-sim posterior. After ROI ranks candidates, the
# top-K solos get re-scored via a K-tick fast_sim rollout: our launch
# now, opp acts via lite_greedy_policy, then both seats idle. Compare
# the rollout's terminal `us_minus_them` to a baseline rollout where
# we did nothing. Replaces the closed-form score with measured
# margin delta — closes the "closed-form vuln is over-pessimistic
# OR too aggressive" gap that G3 exposed. Disable with
# ROI_TIER2_ROLLOUT=off; tune K via ROI_ROLLOUT_K (default 8).
TIER2_ROLLOUT_ENABLED: bool = (
    os.environ.get("ROI_TIER2_ROLLOUT", "on").strip().lower() != "off"
)
TIER2_ROLLOUT_K: int = int(os.environ.get("ROI_ROLLOUT_K", "8"))
TIER2_TOP_K_SOLOS: int = int(os.environ.get("ROI_TIER2_TOP_K", "6"))


def _terminal_value(snap, me: int) -> float:
    """Terminal scoring head for the Tier 2 rollout. Returns ship-delta
    in normal play; returns a decisive +/-1e6 bonus when the game ends
    (opp eliminated or we got eliminated) so capture-eliminations
    don't tie a "no-op + ship growth" baseline.
    """
    # from lib.fast_sim import ship_totals  # inlined by bundle_agent.py
    totals = ship_totals(snap)
    us = totals.get(int(me), 0.0)
    them = sum(v for k, v in totals.items() if int(k) != int(me))
    if getattr(snap, "fake_env", None) is not None and snap.fake_env.done:
        if us > them + 1e-6:
            return 1e6 + (us - them)
        if them > us + 1e-6:
            return -1e6 - (them - us)
        return 0.0
    return us - them


def _rollout_baseline(snap_base, me: int, num_seats: int, K: int) -> float:
    """Roll K ticks from `snap_base` with us idle and opp(s) playing
    lite_greedy_policy. Return `_terminal_value`. The reference
    against which candidate rollouts are compared.
    """
    snap = fs_clone(snap_base)
    for _ in range(int(K)):
        if snap.fake_env.done:
            break
        actions: list = [[] for _ in range(num_seats)]
        for seat in range(num_seats):
            if seat == me:
                continue
            try:
                actions[seat] = lite_greedy_policy(snap.state[seat].observation)
            except Exception:
                actions[seat] = []
        snap = fs_step(snap, actions, in_place=True)
    return _terminal_value(snap, int(me))


def _rollout_with_action(snap_base, our_action: list, me: int,
                          num_seats: int, K: int) -> float:
    """Roll K ticks with our `our_action` on tick 0 (idle afterwards)
    and opp(s) playing lite_greedy_policy throughout. Return
    `_terminal_value`. Compared against `_rollout_baseline` to get the
    measured margin delta of taking `our_action`.
    """
    snap = fs_clone(snap_base)
    for tick in range(int(K)):
        if snap.fake_env.done:
            break
        actions: list = [[] for _ in range(num_seats)]
        actions[me] = our_action if tick == 0 else []
        for seat in range(num_seats):
            if seat == me:
                continue
            try:
                actions[seat] = lite_greedy_policy(snap.state[seat].observation)
            except Exception:
                actions[seat] = []
        snap = fs_step(snap, actions, in_place=True)
    return _terminal_value(snap, int(me))


def _endgame_finish_bonus(
    target,
    capture_step: int,
    world,
    me: int,
    step: int,
    gamma: float,
) -> float:
    """Bonus PV when `target` is the last planet owned by some opp
    player. After capture, that player is eliminated and we play
    uncontested for the rest of the game.

    Bonus = sum(my_planets.production) × pv_horizon(step, capture_step).
    For two-opp scenarios this only fires if BOTH conditions hold:
    target.owner has no other planets. The bonus reflects the
    margin lead we'd accrue once that player is gone.
    """
    if not ENDGAME_BONUS_ENABLED:
        return 0.0
    if int(target.owner) == me or int(target.owner) == -1:
        return 0.0
    target_owner = int(target.owner)
    for p in world.planets_by_id.values():
        if int(p.id) == int(target.id):
            continue
        if int(p.owner) == target_owner:
            return 0.0  # target_owner has another planet → not last
    our_total_prod = sum(
        float(p.production) for p in world.planets_by_id.values()
        if int(p.owner) == me
    )
    if our_total_prod <= 0:
        return 0.0
    pv_full = pv_horizon(int(step), int(capture_step), gamma=gamma,
                         t_total=T_TOTAL_DEFAULT)
    return our_total_prod * pv_full


def _cheapest_opp_counter(
    src,
    residue: int,
    world,
    me: int,
    max_horizon: int,
    extra_defense_ships: int = 0,
    extra_defense_eta: int | None = None,
):
    """Find the worst-case opp counter-attack against `src` left with
    `residue` ships. Returns `(opp, opp_eta, opp_force, our_defense)`
    for the opp planet whose recapture would cost us the most, or
    `None` if no opp can profitably counter.

    `extra_defense_ships` + `extra_defense_eta` model a hypothetical
    reinforcement leg arriving at `src` at tick `extra_defense_eta`.
    If opp's counter ETA is later than the reinforcement, those ships
    add to `src`'s defense at counter-arrival time. Used by the
    defensive-coalition pass to test "does B's reinforcement save A?"
    """
    if int(src.production) <= 0:
        return None

    worst = None
    worst_loss_proxy = -1  # use opp_force-our_defense gap as the tiebreaker
    for opp in world.planets_by_id.values():
        if int(opp.owner) == me or int(opp.owner) == -1:
            continue
        if int(opp.ships) < VULN_MIN_OPP_SHIPS:
            continue
        d = math.hypot(float(opp.x) - float(src.x),
                       float(opp.y) - float(src.y))
        flight = d - float(opp.radius) - float(src.radius) - 0.1
        if flight <= 0:
            continue
        spd = fleet_speed(int(opp.ships))
        if spd <= 0:
            continue
        opp_eta = int(math.ceil(flight / spd))
        if opp_eta > int(max_horizon):
            continue
        opp_force = int(opp.ships) + int(opp.production) * opp_eta
        our_defense = max(0, int(residue)) + int(src.production) * opp_eta
        if (extra_defense_ships > 0
                and extra_defense_eta is not None
                and int(extra_defense_eta) <= opp_eta):
            our_defense += int(extra_defense_ships)
        if opp_force <= our_defense + 1:
            continue  # we hold
        gap = opp_force - our_defense
        if gap > worst_loss_proxy:
            worst_loss_proxy = gap
            worst = (opp, opp_eta, opp_force, our_defense)
    return worst


def _source_vulnerability_loss(
    src,
    residue: int,
    world,
    model,
    me: int,
    step: int,
    max_horizon: int,
    gamma: float = 0.99,
    extra_defense_ships: int = 0,
    extra_defense_eta: int | None = None,
) -> float:
    """Expected production loss if the cheapest opp planet recaptures
    `src` after it's left with `residue` ships.

    Symmetric to _target_holdable_after_capture (proposer.py:407) but
    applied to the SOURCE: after our launch drains src, can any nearby
    opp planet profitably counter-attack? If yes, this returns the
    PV-discounted production stream src would have produced for us over
    the remaining game. Returns 0 if the source is safe (residue
    ≥ safe_garrison or no plausible opp threat).

    No fast_sim. Considers each opp planet's straight-line counter:
    opp arrives with opp.ships + opp.production · opp_eta; we defend
    with residue + src.production · opp_eta. If opp_force > defense + 1
    AND opp_eta ≤ max_horizon, the recapture is feasible.

    `extra_defense_ships` + `extra_defense_eta` simulate a planned ally
    reinforcement of `src`: if the reinforcement arrives ≤ opp's
    counter ETA, those ships add to defense. Used by the defensive-
    coalition pass to test whether a reinforcement neutralises the
    vulnerability.
    """
    if not OPP_MODIFIER_ENABLED:
        return 0.0
    threat = _cheapest_opp_counter(
        src, int(residue), world, me, int(max_horizon),
        extra_defense_ships=int(extra_defense_ships),
        extra_defense_eta=extra_defense_eta,
    )
    if threat is None:
        return 0.0
    _opp, opp_eta, _opp_force, _our_defense = threat
    # Vulnerability is a TRANSIENT cost. Opp holds src for roughly
    # `opp_eta` ticks before we counter-counter-attack (symmetric
    # round-trip). Use a finite loss window, NOT the full remaining
    # game, otherwise the closed-form math says every drained capture
    # is a permanent loss and the chooser refuses to emit.
    # Default loss window = opp_eta (one round-trip). 2× margin
    # because src flipping from ours → opp's shifts margin by 2×
    # tgt.production during the loss window.
    loss_end = int(step) + 2 * int(opp_eta)
    loss_pv = pv_horizon(int(step), int(opp_eta), gamma=gamma,
                         t_total=loss_end)
    return 2.0 * float(src.production) * loss_pv


def solo_roi(
    src,
    tgt,
    ships: int,
    eta: int,
    wait_N: int,
    world,
    model,
    me: int,
    step: int,
    max_horizon: int,
    gamma: float = 0.99,
) -> float:
    """Closed-form ROI score for one (src, tgt, ships, eta, wait_N).

    Returns -inf for refused candidates (past horizon, bounced combat).
    Captures bounce with a small negative ship-cost penalty so the
    chooser's "do nothing" baseline (ROI=0) outranks them. Reinforce
    moves (target already ours) return 0 — no margin gained.
    """
    arrival = int(wait_N) + int(eta)
    if arrival > max_horizon:
        return float("-inf")

    arrivals = list(model.ledger.get(int(tgt.id), []))
    arrivals.append((arrival, int(me), int(ships)))

    # from lib.world_model import predict_garrison_at  # local — keeps lib/ env-free  # inlined by bundle_agent.py
    owner_arr, _garrison_arr = predict_garrison_at(tgt, arrival, arrivals)

    if owner_arr != me:
        return -SHIP_COST_COEF * int(ships)

    if int(tgt.owner) == me:
        # Reinforce. The proposer emits these only when tgt is
        # threatened (`time_to_enemy_threat` + positive shortfall via
        # `capture_size`). Compute the value of HOLDING the planet:
        # if we reinforce, we keep tgt's production for the rest of
        # the game AND deny opp the gain (2× margin).
        #
        # Sanity check: does the reinforcement actually defend? The
        # proposer's capture_size sized this candidate to cover the
        # shortfall, but rare edge cases (cap > budget, ship truncation)
        # can leave a candidate that arrives but doesn't hold. We use
        # `_cheapest_opp_counter` with the reinforcement folded in via
        # `extra_defense_*` to confirm. If still threatened post-
        # reinforcement, this is a band-aid that won't actually save
        # the planet — score it as wasted ships.
        post_residue = int(tgt.ships)  # tgt's current ships pre-arrival
        threat_post = _cheapest_opp_counter(
            tgt, post_residue, world, int(me), int(max_horizon),
            extra_defense_ships=int(ships),
            extra_defense_eta=int(arrival),
        )
        if threat_post is not None:
            # Reinforcement insufficient — tgt still falls. Wasted.
            return -SHIP_COST_COEF * int(ships)
        # Confirm tgt WAS threatened before reinforcement; otherwise
        # we're wasting ships on a safe planet.
        threat_pre = _cheapest_opp_counter(
            tgt, post_residue, world, int(me), int(max_horizon),
        )
        if threat_pre is None:
            return -SHIP_COST_COEF * int(ships)
        # We save the planet. Value = production stream we preserve.
        # 2× margin (we keep + opp denied) over the rest of the game,
        # discounted from now (we incur ship cost now, save the
        # margin stream from the threat ETA onward).
        _opp_pre, opp_eta_pre, _opp_force, _defense = threat_pre
        save_pv = pv_horizon(int(step), int(opp_eta_pre), gamma=gamma,
                             t_total=T_TOTAL_DEFAULT)
        gross_reinforce = 2.0 * float(tgt.production) * save_pv
        ship_cost = SHIP_COST_COEF * int(ships)
        wait_cost = WAIT_COST_COEF * int(wait_N) * float(src.production)
        residue_src = int(src.ships) - int(ships)
        vuln_loss = _source_vulnerability_loss(
            src, residue_src, world, model, me, step, int(max_horizon),
            gamma=gamma,
        )
        return gross_reinforce - ship_cost - wait_cost - vuln_loss

    hold = expected_hold(int(tgt.id), arrival, world, model, t_total=T_TOTAL_DEFAULT)
    if hold <= 0:
        return -SHIP_COST_COEF * int(ships)

    pv_held = pv_horizon(int(step), arrival, gamma=gamma,
                         t_total=int(step) + arrival + hold)
    # 2P-contested-neutral assumption: in an adversarial 2P game,
    # neutrals are competed for. If we don't capture, opp likely will.
    # Treat neutrals the same as enemy planets for margin purposes
    # (gain + deny opp's potential gain). This symmetrises gross vs
    # the 1× vuln-loss multiplier so the chooser actually emits
    # against contested boards. margin_multiplier(tgt, me) returns
    # 1 for neutrals; we promote to 2 here.
    mult = margin_multiplier(tgt, me)
    if mult == 1 and int(tgt.owner) == -1:
        mult = 2

    gross = mult * float(tgt.production) * pv_held
    endgame_bonus = _endgame_finish_bonus(
        tgt, arrival, world, me, step, gamma,
    )
    ship_cost = SHIP_COST_COEF * int(ships)
    wait_cost = WAIT_COST_COEF * int(wait_N) * float(src.production)

    residue = int(src.ships) - int(ships)
    vuln_loss = _source_vulnerability_loss(
        src, residue, world, model, me, step, int(max_horizon), gamma=gamma,
    )

    return gross + endgame_bonus - ship_cost - wait_cost - vuln_loss


def _coalition_legs_for_target(target, my_planets, world, model, me: int,
                               max_horizon: int):
    """Build coalition leg seeds: one per my-planet that could fire at
    `target` within `max_horizon`, sized to keep residue ≥
    MIN_COALITION_RESIDUE.

    Returns [(src, ships, eta, angle), ...]. Empty if no my-planet
    can plausibly reach.
    """
    # from agents.baseline.proposer import aim_and_eta, _source_survives_launch  # inlined by bundle_agent.py
    # from lib.trajectory import predict_fleet_fate  # inlined by bundle_agent.py

    legs = []
    for src in my_planets:
        if int(src.ships) < MIN_LEG_SHIPS + MIN_COALITION_RESIDUE:
            continue
        if int(src.id) == int(target.id):
            continue
        leg_ships = int(src.ships) - MIN_COALITION_RESIDUE
        if leg_ships < MIN_LEG_SHIPS:
            continue
        angle, eta = aim_and_eta(src, target, leg_ships, world.omega, wait_N=0)
        if int(eta) > max_horizon:
            continue
        if not _source_survives_launch(src, leg_ships, 0, world, model, me):
            continue
        # Trajectory admissibility (fire-now only).
        fate = predict_fleet_fate(src, target, float(angle), int(leg_ships), world)
        if fate.outcome != "target":
            continue
        legs.append((src, int(leg_ships), int(eta), float(angle)))
    return legs


def coalition_roi(target, legs, world, model, me: int, step: int,
                  max_horizon: int, gamma: float = 0.99):
    """Closed-form ROI for a multi-leg coalition launch.

    Builds the merged arrival ledger (existing in-flight arrivals at
    target + each leg) and walks arrival ticks until the target flips
    to me. PV-discounts production over the resulting hold horizon.

    Returns (roi_score, capture_step). roi_score is -inf if the
    coalition fails to capture; capture_step is None in that case.
    """
    if not legs:
        return float("-inf"), None

    base_arrivals = list(model.ledger.get(int(target.id), []))
    leg_arrivals = [(int(eta), int(me), int(ships))
                    for (_src, ships, eta, _angle) in legs]
    merged = base_arrivals + leg_arrivals

    arrival_ticks = sorted({a[0] for a in leg_arrivals})
    if not arrival_ticks or arrival_ticks[-1] > max_horizon:
        return float("-inf"), None

    # from lib.world_model import predict_garrison_at  # inlined by bundle_agent.py
    capture_step = None
    for tick in arrival_ticks:
        owner_t, _ = predict_garrison_at(target, tick, merged)
        if owner_t == me:
            capture_step = tick
            break
    if capture_step is None:
        return float("-inf"), None

    hold = expected_hold(int(target.id), capture_step, world, model,
                         t_total=T_TOTAL_DEFAULT)
    if hold <= 0:
        return float("-inf"), None

    pv_held = pv_horizon(int(step), capture_step, gamma=gamma,
                         t_total=int(step) + capture_step + hold)
    mult = margin_multiplier(target, me)
    total_ships = sum(int(ships) for (_src, ships, _eta, _angle) in legs)
    gross = mult * float(target.production) * pv_held
    endgame_bonus = _endgame_finish_bonus(
        target, capture_step, world, me, step, gamma,
    )
    ship_cost = SHIP_COST_COEF * total_ships

    # Coalition vulnerability is capped by opp's counter-capacity.
    # Each opp planet can launch at most one counter-attack per turn.
    # Count plausible opp counter-sources; cap loss at top-K per-leg
    # vulnerabilities, K = num_opp_planets_above_threshold. With one
    # opp planet (the common 2P case), opp can only recapture ONE of
    # our drained sources — taking MAX, not SUM.
    per_leg_vuln: list[float] = []
    for src, ships, _eta, _angle in legs:
        residue = int(src.ships) - int(ships)
        per_leg_vuln.append(_source_vulnerability_loss(
            src, residue, world, model, me, step, int(max_horizon), gamma=gamma,
        ))
    opp_counter_capacity = sum(
        1 for p in world.planets_by_id.values()
        if int(p.owner) != me and int(p.owner) != -1
        and int(p.ships) >= VULN_MIN_OPP_SHIPS
    )
    per_leg_vuln.sort(reverse=True)
    vuln_loss = sum(per_leg_vuln[:max(1, opp_counter_capacity)])

    return gross + endgame_bonus - ship_cost - vuln_loss, capture_step


def _best_reinforcement_for(
    src,
    ships_emitted: int,
    world,
    model,
    me: int,
    step: int,
    max_horizon: int,
    gamma: float,
    used_allies: set,
):
    """Find the best ally that, by reinforcing `src` post-launch, neutralises
    opp's counter-attack profitably.

    Returns `(ally, ally_ships, eta_ally, angle_ally, joint_delta)` for the
    winning reinforcement, or `None` if no ally helps.

    `joint_delta` is the ROI improvement vs. emitting just the attack leg:
    `joint_delta = base_vuln_loss − ally_ship_cost − ally_vuln_loss`. Caller
    accepts when `joint_delta > 0`.
    """
    # from agents.baseline.proposer import aim_and_eta, _source_survives_launch  # inlined by bundle_agent.py
    # from lib.trajectory import predict_fleet_fate  # inlined by bundle_agent.py

    residue = int(src.ships) - int(ships_emitted)
    base_vuln = _source_vulnerability_loss(
        src, residue, world, model, me, step, int(max_horizon), gamma=gamma,
    )
    if base_vuln <= 0.0:
        return None  # source already safe

    # Find the threatening opp counter so we can gate reinforce ETA.
    threat = _cheapest_opp_counter(
        src, residue, world, me, int(max_horizon),
    )
    if threat is None:
        return None
    _opp, opp_eta, _opp_force, _our_defense = threat

    best = None
    best_joint_delta = 0.0
    for ally in world.planets_by_id.values():
        if int(ally.owner) != me:
            continue
        if int(ally.id) == int(src.id):
            continue
        if int(ally.id) in used_allies:
            continue
        if int(ally.ships) < MIN_LEG_SHIPS + MIN_COALITION_RESIDUE:
            continue
        ally_ships = int(ally.ships) - MIN_COALITION_RESIDUE
        if ally_ships < MIN_LEG_SHIPS:
            continue
        angle_ally, eta_ally = aim_and_eta(
            ally, src, ally_ships, world.omega, wait_N=0,
        )
        if int(eta_ally) > int(opp_eta):
            continue  # reinforcement arrives too late
        if not _source_survives_launch(ally, ally_ships, 0, world, model, me):
            continue
        fate = predict_fleet_fate(
            ally, src, float(angle_ally), int(ally_ships), world,
        )
        if fate.outcome != "target":
            continue

        # Recompute vuln with the reinforcement folded in. If it
        # neutralises the threat, new_vuln drops to 0; if it merely
        # reduces (e.g. another opp planet still threatens), new_vuln
        # is the residual.
        new_vuln = _source_vulnerability_loss(
            src, residue, world, model, me, step, int(max_horizon),
            gamma=gamma,
            extra_defense_ships=ally_ships,
            extra_defense_eta=int(eta_ally),
        )
        ally_residue = int(ally.ships) - ally_ships
        ally_vuln = _source_vulnerability_loss(
            ally, ally_residue, world, model, me, step, int(max_horizon),
            gamma=gamma,
        )
        joint_delta = base_vuln - new_vuln - SHIP_COST_COEF * ally_ships - ally_vuln
        if joint_delta > best_joint_delta:
            best_joint_delta = joint_delta
            best = (ally, int(ally_ships), int(eta_ally),
                    float(angle_ally), float(joint_delta))
    return best


def _best_coalition_for_target(target, my_planets, world, model, me: int,
                               step: int, max_horizon: int, gamma: float):
    """Enumerate 2..COALITION_MAX_SOURCES leg subsets; return the
    (roi, legs) of the best-scoring coalition for `target`, or
    (-inf, []) if no coalition captures.

    Caps enumeration at COALITION_MAX_SOURCES seeds total. With 4 seeds
    that's C(4,2)+C(4,3)+C(4,4) = 6+4+1 = 11 subsets per target —
    bounded constant overhead.
    """
    seeds = _coalition_legs_for_target(
        target, my_planets, world, model, me, max_horizon,
    )
    if len(seeds) < 2:
        return float("-inf"), []
    seeds.sort(key=lambda leg: leg[2])  # by ETA ascending
    seeds = seeds[:COALITION_MAX_SOURCES]

    best_roi = float("-inf")
    best_legs: list = []
    for r in range(2, len(seeds) + 1):
        for combo in combinations(seeds, r):
            roi, _cap = coalition_roi(
                target, combo, world, model, me, step, max_horizon, gamma=gamma,
            )
            if roi > best_roi:
                best_roi = roi
                best_legs = list(combo)
    return best_roi, best_legs


def choose_roi(
    snap_base,
    prerank,
    me: int,
    num_seats: int,
    wallclock_ms: float,
    min_horizon: int,
    max_horizon: int,
    gamma: float,
    world,
    model,
    step: int,
) -> list:
    """ROI-prior chooser: solo scoring + N-way coalition, greedy emit.

    1. Score every prerank candidate with solo_roi; keep positives.
    2. For each opp/neutral target, enumerate 2..N-way coalitions of
       my-planet seeds; if best coalition ROI > best constituent
       solo ROI + COALITION_SLACK, swap the coalition in.
    3. Greedy commit: sort by ROI desc, one emit per (src, tgt) pair.

    Coalition legs are emitted as independent fire-now launches at
    the same target — the engine handles concurrent arrivals via
    its arrival resolver. wait_N>0 candidates remain solo-only at
    this phase (mixed wait coalition geometry isn't validated by
    predict_fleet_fate).
    """
    # Wallclock budget — insurance against dense-board coalition
    # enumeration. Bench at G2 showed max=687ms on 2P self-play, well
    # under the 1000ms cap, so this is structurally cheap. Solo scoring
    # and the defensive post-pass are O(prerank × opps) and bounded;
    # the deadline check sits at the head of coalition enumeration
    # where the cost lives.
    deadline = time.perf_counter() + max(50.0, float(wallclock_ms)) / 1000.0

    # --- Pass 1: solo scoring with ship-count variants ---
    # For each prerank entry, enumerate candidate ship counts and let
    # solo_roi pick the best. Variants:
    #   (a) original (proposer's pick — usually cap or full budget).
    #   (b) at-fire-time max_safe (src.ships + wait_N × src.production
    #       − MIN_SOLO_RESIDUE) when it differs from (a) and still
    #       captures (≥ MIN_LEG_SHIPS).
    # CRITICAL: max_safe must use AT-FIRE-TIME ships for wait_N>0
    # candidates. With wait_N=11 and src.ships=10, effective fire-time
    # ships = 21; max_safe = 16. Computing max_safe off the current
    # src.ships alone clamps the launch to 5 ships, which always
    # bounces against the wait_N=11 cap. That bug made ROI emit nothing
    # for half a game (G3 vs v7_0 lost 0/32).
    # No hard rejection on (a) — vuln_loss is the principled cost
    # of draining the source, and is enforced inside solo_roi.
    solo_scored: list = []  # (score, src, tgt, ships, angle, wait_N)
    solo_by_target: dict[int, list] = {}
    for entry in prerank:
        _cheap, src, tgt, ships_orig, angle, eta, _horizon, wait_N = entry
        src_ships_at_fire = int(src.ships) + int(wait_N) * int(src.production)
        max_safe_at_fire = max(0, src_ships_at_fire - MIN_SOLO_RESIDUE)

        ship_variants: list[int] = []
        if int(ships_orig) >= MIN_LEG_SHIPS:
            ship_variants.append(int(ships_orig))
        if (max_safe_at_fire >= MIN_LEG_SHIPS
                and max_safe_at_fire != int(ships_orig)):
            ship_variants.append(max_safe_at_fire)
        if not ship_variants:
            continue

        best_score = float("-inf")
        best_ships = ship_variants[0]
        for s in ship_variants:
            score = solo_roi(
                src, tgt, s, int(eta), int(wait_N),
                world, model, int(me), int(step), int(max_horizon),
                gamma=gamma,
            )
            if score > best_score:
                best_score = score
                best_ships = s
        if best_score == float("-inf") or best_score <= 0.0:
            continue
        rec = (best_score, src, tgt, int(best_ships), float(angle), int(wait_N))
        solo_scored.append(rec)
        solo_by_target.setdefault(int(tgt.id), []).append(rec)

    # --- Pass 2: coalition enumeration per opp/neutral target ---
    my_planets = [p for p in world.planets_by_id.values() if int(p.owner) == me]
    opp_targets = [p for p in world.planets_by_id.values()
                   if int(p.owner) != me]

    coalitions: list = []  # (score, target, legs)
    for target in opp_targets:
        if time.perf_counter() > deadline:
            break  # wallclock budget exhausted; emit what we have
        c_roi, c_legs = _best_coalition_for_target(
            target, my_planets, world, model, me, step, max_horizon, gamma,
        )
        if c_roi <= 0.0 or not c_legs:
            continue
        # Coalition beats best solo on this target only if strictly
        # better by the slack margin. Otherwise the solo path is
        # preferred (smaller ship commitment).
        best_solo_on_tgt = max(
            (s[0] for s in solo_by_target.get(int(target.id), [])),
            default=0.0,
        )
        if c_roi <= best_solo_on_tgt + COALITION_SLACK:
            continue
        coalitions.append((c_roi, target, c_legs))

    # --- Pass 2.5: Tier 2 rollout posterior on top-K solos ---
    # ROI's closed-form scoring is the prior; for the top-K solos
    # we replace the score with a measured fast_sim rollout delta
    # vs an idle-us baseline. Coalitions and wait_N>0 solos stay on
    # closed-form (rollout integration for those is more complex).
    # The rollout sees opp's actual lite_greedy reaction within K
    # ticks — closes the closed-form vuln calibration gap.
    if TIER2_ROLLOUT_ENABLED and solo_scored and time.perf_counter() <= deadline:
        # Sort solos by closed-form score (descending). Validate top-K.
        solo_scored.sort(key=lambda r: -r[0])
        # Determine rollout horizon — must be long enough for the
        # furthest top-K candidate to ARRIVE and resolve. fast_sim's
        # `ship_totals` counts in-flight fleets in their owner's total,
        # so if we terminate the rollout before the fleet lands, the
        # candidate is scored as a no-op vs idle baseline.
        max_eta_top_k = 0
        for i in range(min(TIER2_TOP_K_SOLOS, len(solo_scored))):
            _score, src, tgt, ships, angle, wait_N = solo_scored[i]
            # eta isn't stored in the rec; recover via fleet_speed and
            # straight-line distance (approximation; ignores wait_N
            # candidates which are skipped below).
            d = math.hypot(float(tgt.x) - float(src.x),
                           float(tgt.y) - float(src.y))
            spd = fleet_speed(int(ships))
            if spd > 0:
                max_eta_top_k = max(max_eta_top_k, int(math.ceil(d / spd)))
        K_rollout = max(int(TIER2_ROLLOUT_K), max_eta_top_k + 5)

        try:
            baseline_delta = _rollout_baseline(
                snap_base, int(me), int(num_seats), K_rollout,
            )
        except Exception:
            baseline_delta = None
        if baseline_delta is not None:
            top_k = min(TIER2_TOP_K_SOLOS, len(solo_scored))
            rescored: list = []
            for i in range(top_k):
                if time.perf_counter() > deadline:
                    break
                _score, src, tgt, ships, angle, wait_N = solo_scored[i]
                if int(wait_N) != 0:
                    # Wait-N solo: leave on closed-form score.
                    rescored.append(solo_scored[i])
                    continue
                action = [[int(src.id), float(angle), int(ships)]]
                try:
                    cand_delta = _rollout_with_action(
                        snap_base, action, int(me), int(num_seats),
                        K_rollout,
                    )
                except Exception:
                    rescored.append(solo_scored[i])
                    continue
                roll_score = float(cand_delta) - float(baseline_delta)
                # Drop candidates the rollout says are worse than idle.
                # ROI's prior was over-optimistic on these (e.g.,
                # didn't see opp's actual counter-attack landing).
                if roll_score <= 0.0:
                    continue
                rescored.append((roll_score, src, tgt, int(ships),
                                  float(angle), int(wait_N)))
            # Keep the rescored top-K (replaces their original
            # closed-form scores); the un-validated tail stays as-is.
            solo_scored = rescored + solo_scored[top_k:]
            # Rebuild solo_by_target with new scores.
            solo_by_target = {}
            for rec in solo_scored:
                solo_by_target.setdefault(int(rec[2].id), []).append(rec)

    # --- Pass 3: greedy emit ---
    # Sort all candidates (solo + coalition) by score desc.
    # Coalitions claim ALL their legs' sources atomically.
    combined: list = []
    for rec in solo_scored:
        combined.append(("solo", rec[0], rec))
    for coal in coalitions:
        combined.append(("coalition", coal[0], coal))
    combined.sort(key=lambda c: -c[1])

    used_srcs: set[int] = set()
    used_tgts: set[int] = set()
    moves: list[list] = []
    for kind, _score, payload in combined:
        if kind == "coalition":
            _c_roi, target, legs = payload
            tid = int(target.id)
            if tid in used_tgts:
                continue
            if any(int(leg[0].id) in used_srcs for leg in legs):
                continue
            used_tgts.add(tid)
            for src, ships, _eta, angle in legs:
                sid = int(src.id)
                used_srcs.add(sid)
                moves.append([sid, float(angle), int(ships)])
            continue
        # solo
        _score, src, tgt, ships, angle, wait_N = payload
        sid, tid = int(src.id), int(tgt.id)
        if sid in used_srcs or tid in used_tgts:
            continue
        used_srcs.add(sid)
        used_tgts.add(tid)
        if wait_N == 0:
            moves.append([sid, float(angle), int(ships)])

    # --- Pass 4: defensive-coalition post-pass ---
    # For each emitted move whose source ends up vulnerable, see if an
    # idle ally can reinforce before opp's counter arrives. The reinforce
    # is emitted as an additional move.
    if DEFENSIVE_COALITION_ENABLED and moves:
        used_allies: set[int] = set(used_srcs)  # already-emitting srcs can't reinforce
        # Iterate a snapshot so we don't reinforce-reinforce.
        for src_id, _angle, ships in [tuple(m) for m in moves]:
            src = world.planets_by_id.get(int(src_id))
            if src is None:
                continue
            reinforce = _best_reinforcement_for(
                src, int(ships), world, model, me, step,
                int(max_horizon), gamma, used_allies,
            )
            if reinforce is None:
                continue
            ally, ally_ships, _eta_ally, angle_ally, _delta = reinforce
            used_allies.add(int(ally.id))
            moves.append([int(ally.id), float(angle_ally), int(ally_ships)])

    return moves

# === inlined: agents/baseline/chooser_trajectory.py ===
"""Trajectory-first chooser — drop-in alternative to `chooser.choose`.

Replaces the K-step fast_sim rollout + composite-leaf-value approach
with deterministic trajectory analysis + single-tick combat prediction.

Pipeline per turn:
  1. Iterate proposer's prerank candidates in cheap-Δ order.
  2. For each candidate `(src, tgt, ships, angle, eta, wait_N)`:
     a. `predict_fleet_fate(src, tgt, angle, ships, world)` — drop
        on `sun` / `oob` / `timeout` / `planet` (path-blocked by a
        different planet) / `comet_collision` (predicted-hit comet).
     b. For surviving "target" outcomes that ARE comets, drop if the
        comet's remaining lifetime ≤ ETA (expires before arrival).
     c. `predict_garrison_at(tgt, eta, ledger[tgt.id] + [our arrival])`
        — single-tick combat result.
     d. Score:
          captured (was-enemy → now-us)  → production × time_remaining,
                                            capped at comet life.
          reinforced (already-ours)      → skip (no extra credit;
                                            threat reinforcement is
                                            handled at proposer.propose).
          bounced (still-not-us)         → -ships (waste penalty).
  3. Sort surviving by score desc; greedy non-dogpile dedup by
     (src_id, tgt_id); emit `wait_N==0` winners only (`wait_N>0`
     reserves src+tgt, emits nothing this turn).

No K-step rollout, no leaf value-function approximation, no fast_sim
state cloning. Cost is O(candidates × (trajectory_steps + eta + arrivals)).

PI critique 2026-05-17: "we should be thinking in fleet trajectories";
"sun-deaths should be 0% with proper trajectory analysis". See
`knowledge-base/concepts/trajectory-first-architecture.md`.
"""


import math
import os
import time

# from agents.baseline.chooser import affordable_validate_cap, opp_actions_for_snap  # inlined by bundle_agent.py
# from agents.baseline.value import DEFAULT_GAMMA, select_favor_fn  # inlined by bundle_agent.py
# from lib.fast_sim import clone as fs_clone  # inlined by bundle_agent.py
fs_clone = clone
# from lib.fast_sim import step as fs_step  # inlined by bundle_agent.py
fs_step = step
# from lib.opp_model import lite_greedy_policy as _me_policy  # inlined by bundle_agent.py
_me_policy = lite_greedy_policy
# from lib.opp_model import me_defensive_action as _me_defends_policy  # inlined by bundle_agent.py
_me_defends_policy = me_defensive_action
# from lib.trajectory import predict_fleet_fate  # inlined by bundle_agent.py
# from lib.world_model import comet_remaining_lifetime, predict_garrison_at  # inlined by bundle_agent.py


EPISODE_STEPS_TOTAL: int = 500
WASTE_WEIGHT: float = 0.5
CAPTURE_REWARD_WEIGHT: float = 0.05

# Leader-focus bonus (2026-05-21). In 4P, multiply capture score by
# LEADER_FOCUS_WEIGHT when the target's owner is the current leader
# (player with highest planet-production-owned). Pushes focal to attack
# the strongest opp rather than spreading attacks across all three.
# Disabled (=1.0) in 2P automatically since there is no leader to
# distinguish. Default 1.0 (no change); opt-in via env var.
LEADER_FOCUS_WEIGHT: float = float(os.environ.get("BASELINE_LEADER_FOCUS", "1.0"))

# Neutral-capture bonus (2026-05-21). In 4P trace of seed=5 (a loss),
# focal made 73 captures-from-enemy but only 6 captures-from-neutral
# while phase_c snowballed to 36 planets via aggressive neutral grab.
# This bonus tilts the chooser toward neutral targets relative to
# enemy targets — neutrals don't have a defender (cheaper) and grow
# production without risk of attrition. Stronger in the opening phase
# where territorial grab dominates outcomes. Default 1.0 (no change);
# opt-in via env var BASELINE_NEUTRAL_BONUS.
NEUTRAL_BONUS_WEIGHT: float = float(os.environ.get("BASELINE_NEUTRAL_BONUS", "1.0"))
NEUTRAL_EARLY_HORIZON: int = int(os.environ.get("BASELINE_NEUTRAL_EARLY_HORIZON", "50"))
NEUTRAL_EARLY_EXTRA: float = float(os.environ.get("BASELINE_NEUTRAL_EARLY_EXTRA", "1.0"))

# Follow-on hold bonus (Fix 2b — 2026-05-27 plan). Opt-in scoring
# bonus on captures that enable a profitable follow-on launch from the
# newly-captured base. Surfaces the B3/B4 modeling-correct snipe
# helpers (`_best_followon` / `_followon_hold_estimate` in
# `lib/missions/snipe.py`) into the live trajectory chooser path —
# previously B3/B4 were in dead code from this agent's perspective.
# Default 0.0 (no-op); bundle wrapper opts in once local A/B confirms
# lift. Calibrated against `CAPTURE_REWARD_WEIGHT=0.05`.
FOLLOWON_BONUS_WEIGHT: float = float(
    os.environ.get("BASELINE_FOLLOWON_BONUS", "0.0"),
)
FOLLOWON_RADIUS: float = float(
    os.environ.get("BASELINE_FOLLOWON_RADIUS", "35.0"),
)

# Score floor for emit (2026-05-27 — concentration knob). Today every
# candidate with `score > 0.0` fires; in midgame this scatters small
# marginal launches across every owned planet. `MIN_DELTA` raises the
# floor so only candidates above a tunable threshold survive — natural
# concentration without an arbitrary count-cap. Default 0.0 preserves
# byte-for-byte legacy (strict `> 0.0`); positive values install a
# strict `>=` floor. Units are PV-discounted delta (see
# `score_candidate_v4`); tune via local A/B.
MIN_DELTA: float = float(os.environ.get("BASELINE_MIN_DELTA", "0.0"))

# Ship-turn opportunity-cost penalty (2026-05-27 Step 2B). Today the
# leaf `favor` returns ~297-340 for any positive-prod capture regardless
# of eta — pv_horizon(leaf_step, 0) ≈ 99 for any leaf_step in 25..50
# with γ=0.99, t_total=500. Result: slow captures (eta=40) score ~88%
# of fast captures (eta=10) when in reality they tie up ships 4x
# longer. Penalty subtracts κ × ships × (wait_N + eta) from delta to
# price the time the ships are committed and unable to defend/redirect.
# Default 0.0 preserves byte-for-byte legacy. Tune via local A/B.
SHIP_TURN_KAPPA: float = float(
    os.environ.get("BASELINE_SHIP_TURN_KAPPA", "0.0"),
)


def _leader_owner_from_world(world, me: int) -> int | None:
    """Return the player id (other than `me`) with the highest total
    planet production owned. Returns None when leader is undefined
    (no opps, single opp, or production tie).
    """
    if world is None:
        return None
    prod_by_owner: dict[int, int] = {}
    try:
        plist = list(world.planets_by_id.values())
    except AttributeError:
        return None
    for p in plist:
        o = int(getattr(p, "owner", -1))
        if o < 0 or o == int(me):
            continue
        prod_by_owner[o] = prod_by_owner.get(o, 0) + int(getattr(p, "production", 0))
    if len(prod_by_owner) < 2:
        return None  # 2P or only one opp; no leader distinction
    best = max(prod_by_owner.values())
    leaders = [o for o, v in prod_by_owner.items() if v == best]
    if len(leaders) > 1:
        return None  # tie, no clear leader
    return leaders[0]

# Bug #14 fix attempt — CHEAP MIRROR. NEGATIVE RESULT 2026-05-18 PM.
#
# Premise: at each tick of the leaf rollout, drive ME with
# `lite_greedy_policy` (the same policy used for opp seats) instead
# of standing still after the single injected launch. Pre-fix the
# rollout was asymmetric: opp reacted each tick but WE didn't, so
# every candidate was scored against a worst-case "I make this move
# and then sit on my hands for 25 ticks while opp keeps playing"
# baseline. The asymmetry is documented in the bug catalog at
# `audit/2026-05-18-bug-catalog.md#14`.
#
# Empirical result with `BASELINE_ME_REACTS=1`:
# - The 3 xfail oracles (cleanup / coordinated / solo) did NOT flip
#   to pass — the mirror didn't unlock the expected coordination.
# - The 2 working defense oracles (defense_against_incoming_multi_fleet,
#   defense_wide_gap_multi_wave) REGRESSED to FAIL. The likely cause:
#   `lite_greedy_policy` is too greedy/attack-biased — in the baseline
#   it emits attack launches from the would-be reinforcer planet
#   (e.g. P1 with 200 ships → launches at opp), so the baseline lets
#   the threatened planet fall too. The candidate's reinforce can't
#   look more attractive than a baseline that's already failing in
#   the same way.
#
# PI's caveat was exactly this: "our chooser is meant to be SMARTER
# than lite_greedy. Using lite_greedy as our rollout policy
# UNDER-rates our skill." Under-rates badly enough to break working
# tests. The fix isn't viable as written.
#
# Next steps (deferred): try the "future-capture credit at the leaf"
# alternative (catalog option #3) — don't simulate us in the rollout
# at all; instead at the leaf add bonus credit for OUR in-flight
# captures past the leaf horizon. Captures the "we'll defend"
# intuition via accounting, not simulation, and doesn't depend on
# lite_greedy's tactical quality.
#
# The toggle and `_me_reactive_action` helper are kept so the
# experiment is reproducible. Default OFF — env var
# `BASELINE_ME_REACTS=1` to re-enable.
_ME_REACTS_ENABLED = os.environ.get("BASELINE_ME_REACTS", "0") != "0"


def _me_reactive_action(snap, me: int) -> list:
    """`lite_greedy_policy` driven from ME's observation. Same call
    shape as `opp_actions_for_snap` for non-me seats; isolated here so
    rollout sites stay readable."""
    try:
        return _me_policy(snap.state[me].observation) or []
    except Exception:
        return []


# Bug #14 fix — OPTION 5: PURELY DEFENSIVE policy for ME in the rollout
# (2026-05-18 PM). Supersedes the failed cheap-mirror (option 1 above).
# At each rollout tick, ME runs `lib.opp_model.me_defensive_action`:
# scan inbound enemy fleets, find under-defended owned planets, emit a
# reinforce launch from the nearest viable sister planet. Never attacks.
#
# Rationale vs the cheap-mirror failure: lite_greedy is too attack-biased
# — in the baseline path it emitted attack launches from the would-be
# reinforcer, so the baseline let the threatened planet fall too. A
# purely-defensive policy avoids that pathology because it never
# misallocates the reinforcer's ships to offense. The chooser's own
# attack moves are made on its real next turn; the rollout's job is to
# model opp's reaction (which implies us defending), not us attacking
# again.
#
# Default OFF (env var `BASELINE_ME_DEFENDS=1` to enable). When both
# DEFENDS and REACTS are set, DEFENDS takes precedence (REACTS is the
# deprecated cheap-mirror experiment kept only for reproducibility).
_ME_DEFENDS_ENABLED = os.environ.get("BASELINE_ME_DEFENDS", "0") != "0"


def _me_defensive_action(snap, me: int) -> list:
    """Defensive policy on ME's observation. Same call shape as
    `_me_reactive_action`; isolated here so the three rollout sites
    stay readable."""
    try:
        return _me_defends_policy(snap.state[me].observation, me) or []
    except Exception:
        return []

# How many ticks AFTER fleet arrival to keep simulating before reading
# the leaf. Long enough to see immediate combat aftermath (production
# tick, opp counter-arrivals already in flight); short enough that we
# don't run a full v15-style 40-step rollout. v3 trade-off vs v2:
# v2 used predict_garrison_at (single-tick static math); v3 uses
# fast_sim along the actual trajectory so the leaf reflects opp's
# reactive launches (via lite_greedy_policy each tick).
SETTLE_TURNS: int = 3

# Multi-launch budget (Step A, 2026-05-17 v2): the v1 chooser hard-
# capped at 1 launch per source per turn. v15 routinely emits
# parallel launches (3-5 sources × 1-2 fleets each); the 1/source
# limit was a load-bearing reason v1 lost 0/32 vs v15. v2 tracks
# remaining ships per source and emits multiple launches until the
# source's ships fall below the next candidate's requirement.
#
# MIN_SOURCE_RESERVE = 0: v15-line baseline does NOT hold a reserve;
# proposer's MIN_FLEET_SIZE filter ensures we never emit absurdly
# small launches. Holding even 2 ships back blocks the very common
# early-game case where a 10-ship home wants to send all 10 to a
# 5-ship neutral. Adopt v15's "spend it all if it captures" stance.
MIN_SOURCE_RESERVE: int = 0

# Opp 1-turn lookahead (Step C, 2026-05-17 v2): predict each opp
# source's likely best 1 launch; inject into ledger before scoring.
# Knobs first-pass; tune if A/B is borderline.
OPP_NEAREST_K: int = 4
OPP_SHIP_FRACTION: float = 0.8
OPP_MIN_SHIPS: int = 4

# Wallclock budgeting (2026-05-17 wait_N session): mirror composite
# chooser. v4 with wait_N>0 routinely blew the 1000ms env cap on heavy
# turns (max=2416ms in n=64 A/B vs v15). Composite stays within cap via
# affordable_validate_cap + safe_deadline pre-bail.
#
# N_VALIDATE=200 (vs composite's 60): trajectory v4's per-candidate
# cost is shorter on average (prop_horizon clamps to MIN_HORIZON=25 for
# most candidates; composite's avg horizon is closer to 32). The pre-
# wallclock-fix A/B (no cap) hit 65.6%; the N_VALIDATE=60 cap dropped
# it to 57.8% — confirming candidate breadth matters. Let safe_deadline
# bind the actual budget; N_VALIDATE is just a generous upper bound.
N_VALIDATE: int = 200
RESERVED_OVERHEAD_MS: float = 50.0

# Direction B — joint candidate evaluation (2026-05-18).
# Caps env-var configurable so a variant can be A/B'd without rebuilding
# the bundle. Default values match the original 2026-05-18 v3 production
# constants. `BASELINE_JOINT_AGGR=1` ALSO lifts the `used_tgts` lock on
# both solo and joint emits (multi-source-same-target stacking — combat
# rule 1 exploit). Origin: 2026-05-20 PI directive to find a STRUCTURAL
# lift over the n=8 ablation plateau. Risk: ship-waste from over-emit to
# same target; controlled by leaving `used_srcs` lock in place.
# Verified (C)+(E) via scripts/verify_solo_vs_joint.py on live episodes
# of 52754310 (mu=1271.8): solo launches from idle planets capture only
# 21pct of nearest targets (production growth out-paces accumulation);
# joint launches with a neighbor capture 89pct (+68pp lift).
# Opt-in via BASELINE_JOINT=1. Production stays on solo-only path.
JOINT_TOP_K_PER_TARGET: int = int(
    os.environ.get("BASELINE_JOINT_TOP_K", "3")
)
JOINT_MAX_PAIRS: int = int(
    os.environ.get("BASELINE_JOINT_MAX_PAIRS", "20")
)
JOINT_LIFT_USED_TGTS: bool = (
    os.environ.get("BASELINE_JOINT_AGGR", "0").strip() == "1"
)


def score_candidate_dyn(snap_base, src, tgt, ships: int, angle: float,
                        me: int, num_seats: int, world,
                        settle_turns: int = SETTLE_TURNS,
                        ) -> tuple[float, str, int | None]:
    """v3 dynamic scoring: fast_sim along the trajectory.

    Same admissibility gate as v2 (sun / oob / comet-collision /
    comet-expired-by-arrival are rejected deterministically by
    `predict_fleet_fate`). For surviving candidates, runs `fs_step`
    for `eta + settle_turns` ticks with our action injected at tick 0
    and `lite_greedy_policy` driving every opp seat reactively. Reads
    the target planet's ACTUAL owner from the simulated leaf —
    capturing whatever happens during flight (opp counter-launches,
    other fleets arriving, production accumulation, multi-fleet
    combat resolution).

    This is the convergence of trajectory thinking (deterministic
    admissibility filter, no expensive leaf value function) with
    the K-step rollout's strategic depth (reactive opp via fast_sim
    + lite_greedy). Cost per candidate ≈ (eta + settle) × per-step
    fast_sim cost (~0.5 ms). For eta=10, that's ~6.5 ms — vs v15's
    composite chooser ~20 ms (40-step rollout + 2-5 ms composite leaf).

    Returns `(score, status, eta)`. Statuses: same vocabulary as v2.
    """
    fate = predict_fleet_fate(src, tgt, angle, ships, world)
    if fate.outcome == "sun":
        return (float("-inf"), "sun", fate.step)
    if fate.outcome == "oob":
        return (float("-inf"), "oob", fate.step)
    if fate.outcome == "timeout":
        return (float("-inf"), "timeout", fate.step)
    if fate.outcome == "planet":
        if fate.hit_planet_id in world.comet_ids:
            return (float("-inf"), "comet_collision", fate.step)
        return (float("-inf"), "path_blocked", fate.step)

    eta = int(fate.step)
    if int(tgt.id) in world.comet_ids:
        life = comet_remaining_lifetime(int(tgt.id), world)
        if life is None or life <= eta:
            return (float("-inf"), "comet_expired", eta)

    # Run fast_sim eta + settle ticks; inject our action at tick 0.
    snap = fs_clone(snap_base)
    horizon = eta + settle_turns
    for t in range(horizon):
        if snap.fake_env.done:
            break
        actions = opp_actions_for_snap(snap, me, num_seats)
        if t == 0:
            actions[me] = [[int(src.id), float(angle), int(ships)]]
        snap = fs_step(snap, actions, in_place=True)

    # Read target's leaf state from the simulated obs.
    leaf_obs = snap.state[me].observation
    leaf_planets = (
        leaf_obs.get("planets", []) if isinstance(leaf_obs, dict)
        else getattr(leaf_obs, "planets", [])
    )
    target_pid = int(tgt.id)
    leaf_owner: int = -2  # sentinel: target not found (e.g. expired comet)
    for p in leaf_planets:
        if int(p[0]) == target_pid:
            leaf_owner = int(p[1])
            break

    # Score from leaf outcome. The fast_sim leaf reflects opp's
    # reactive launches and production over `eta + settle` ticks, so
    # "owner == me at leaf" is a much stronger signal than the static
    # predict_garrison_at v2 used.
    if leaf_owner == me:
        # Was it ours BEFORE the launch?
        if int(tgt.owner) == me:
            # Already ours, still ours — pure reinforcement (no extra credit).
            return (0.0, "reinforced", eta)
        # Captured (or recaptured a planet that would have fallen).
        time_remaining = max(0, EPISODE_STEPS_TOTAL - int(world.step) - eta)
        held = time_remaining
        if target_pid in world.comet_ids:
            life = comet_remaining_lifetime(target_pid, world)
            if life is not None:
                held = min(held, max(0, life - eta))
        return (CAPTURE_REWARD_WEIGHT * float(tgt.production) * float(held),
                "captured", eta)

    # Leaf shows target NOT ours: either bounce (still enemy/neutral)
    # or the planet vanished (comet expired). Both → waste.
    return (-WASTE_WEIGHT * ships, "bounced", eta)


# ---------------------------------------------------------------------------
# v4 (Direction A, 2026-05-17 PM): favor leaf + idle-baseline Δ-scoring.
# ---------------------------------------------------------------------------
#
# v3 lost 0/32 vs v15 with a BINARY leaf (target.owner == me at leaf?).
# Hypothesis: binary scoring collapses ~bits of strategic info that the
# v15-style continuous `favor` leaf preserves (ship balance + production
# balance × pv_horizon). v4 replaces v3's binary check with v15's
# Δ-from-idle-baseline-favor scoring, keeping v3's eta-bounded
# trajectory rollout (cheaper than v15's fixed K=40).
#
# If v4 reaches v15 parity: information-collapse hypothesis confirmed;
# trajectory chooser was architecturally fine, leaf was the bug.
# If v4 still loses: scoring isn't the binding constraint; pivot to
# joint-action or sequential planning (Directions B/C in concept doc).


def build_trajectory_baseline(snap_base, me: int, num_seats: int,
                              horizon: int, favor_fn, gamma: float,
                              ) -> list[float]:
    """Idle-baseline favor at each tick in [0, horizon]. Used by v4 to
    subtract the do-nothing alternative from each candidate's leaf
    favor (mirrors `chooser.build_idle_baseline`).

    Returns a list of length `horizon + 1`. Cost: `horizon` calls to
    `fs_step` + `(horizon + 1)` calls to `favor_fn`. Runs ONCE per
    chooser invocation, not per candidate.

    Bug #14 fix (2026-05-18): when `BASELINE_ME_REACTS=1`, the
    baseline ALSO has ME play `lite_greedy_policy` reactively each
    tick — same policy used for opp seats. Reason: if only the
    candidate path has ME reactive but the baseline doesn't, the Δ
    captures "value of ME playing at all" rather than "value of THIS
    candidate's specific move." Symmetric framing isolates the
    candidate's marginal contribution.
    """
    snap = fs_clone(snap_base)
    out: list[float] = [
        favor_fn(snap.state[me].observation, me, num_seats, gamma=gamma),
    ]
    for _ in range(horizon):
        if snap.fake_env.done:
            out.append(out[-1])
            continue
        actions = opp_actions_for_snap(snap, me, num_seats)
        # Baseline IS asymmetric on purpose (ME idle, opp reactive) —
        # we measure the candidate's marginal value above the worst-
        # case "I do nothing this turn AND on every future turn"
        # outcome. Auto-defense applies in CANDIDATE rollouts only
        # (sites B/C below), where it represents "future me reacting
        # to opp's response to MY move." Adding auto-defense here
        # makes the baseline too capable and zeros the candidate-Δ
        # for defensive launches (auto-defense already handles them),
        # so the chooser refuses to emit real defense.
        if _ME_REACTS_ENABLED:
            actions[me] = _me_reactive_action(snap, me)
        snap = fs_step(snap, actions, in_place=True)
        out.append(
            favor_fn(snap.state[me].observation, me, num_seats, gamma=gamma),
        )
    return out


def score_candidate_v4(snap_base, src, tgt, ships: int, angle: float,
                       me: int, num_seats: int, world,
                       baseline_favors: list[float],
                       favor_fn, gamma: float,
                       horizon: int,
                       skip_admissibility: bool = False,
                       wait_N: int = 0,
                       model=None,
                       ) -> tuple[float, str, int | None]:
    """v4 scoring: same admissibility filter + fast_sim rollout as v3,
    but the leaf is `favor_fn` instead of a binary owner-check, and the
    score is `Δ = leaf_with_action − baseline_favors[horizon]`.

    Returns `(delta, status, eta)`. Statuses match v3 plus "scored"
    (the success case for v4, since "captured/reinforced/bounced" are
    no longer first-class outcomes — favor implicitly encodes them).

    `skip_admissibility=True` bypasses the predict_fleet_fate filter
    (env var TRAJECTORY_SKIP_ADMISSIBILITY=on) to isolate whether the
    filter is false-rejecting valid candidates that composite_a2 lets
    through.

    `wait_N>0` defers action injection to step `wait_N` in the rollout
    (matches composite chooser's `score_action` pattern at
    `agents/baseline/chooser.py:60-73`). Admissibility filter only runs
    for `wait_N==0` (the source planet orbits between now and the wait
    point, so the pre-launch trajectory analysis is stale); for wait>0
    candidates, fast_sim's collision resolution catches real sun/oob/
    comet hits inside the rollout.
    """
    eta = 0
    if not skip_admissibility and int(wait_N) == 0:
        fate = predict_fleet_fate(src, tgt, angle, ships, world)
        if fate.outcome == "sun":
            return (float("-inf"), "sun", fate.step)
        if fate.outcome == "oob":
            return (float("-inf"), "oob", fate.step)
        if fate.outcome == "timeout":
            return (float("-inf"), "timeout", fate.step)
        if fate.outcome == "planet":
            if fate.hit_planet_id in world.comet_ids:
                return (float("-inf"), "comet_collision", fate.step)
            return (float("-inf"), "path_blocked", fate.step)

        eta = int(fate.step)
        if int(tgt.id) in world.comet_ids:
            life = comet_remaining_lifetime(int(tgt.id), world)
            if life is None or life <= eta:
                return (float("-inf"), "comet_expired", eta)

    # Clamp horizon to baseline length (caller pre-sized).
    if horizon >= len(baseline_favors):
        horizon = len(baseline_favors) - 1

    snap = fs_clone(snap_base)
    # Bug #14 option 5: precompute defensive reinforces ONCE from the
    # candidate's tick-0 observation. In real game the chooser emits
    # ALL of this turn's moves (candidate + any reactive defense)
    # simultaneously — we model that by merging them at `wait_N` and
    # NOT re-evaluating defense on every rollout tick. Per-call cost
    # drops from `horizon × per-candidate` to `1 × per-candidate`,
    # which is the difference between bench-WATCH (10 outliers >1s
    # at horizon=25 with the per-tick variant) and bench-PASS.
    me_defense_emits: list = []
    if _ME_DEFENDS_ENABLED:
        me_defense_emits = _me_defensive_action(snap, me)

    for t in range(horizon):
        if snap.fake_env.done:
            break
        actions = opp_actions_for_snap(snap, me, num_seats)
        if t == int(wait_N):
            # Candidate first (the chooser's primary decision), then
            # defensive emits. If the candidate drains the planet
            # defense wanted as reinforcer, fs_step will cap the
            # defensive launch at remaining ships.
            actions[me] = (
                [[int(src.id), float(angle), int(ships)]]
                + list(me_defense_emits)
            )
        elif _ME_REACTS_ENABLED:
            actions[me] = _me_reactive_action(snap, me)
        snap = fs_step(snap, actions, in_place=True)

    leaf = favor_fn(snap.state[me].observation, me, num_seats, gamma=gamma)
    delta = leaf - baseline_favors[horizon]

    # Plumb NEUTRAL_BONUS + LEADER_FOCUS into the live scoring path.
    # The earlier dead-code path (`score_candidate`, v2 static-garrison
    # scorer) read these env vars but was never called; v4 ignored them.
    # Multiply POSITIVE deltas only — the bonus is a tilt toward
    # preferred targets, not a punishment for bad candidates that
    # happen to be neutral/leader. See plan
    # `/root/.claude/plans/fix-one-and-two-cuddly-dewdrop.md` Fix 1.
    if delta > 0.0:
        bonus = 1.0
        if NEUTRAL_BONUS_WEIGHT != 1.0 and int(tgt.owner) == -1:
            bonus *= NEUTRAL_BONUS_WEIGHT
            if int(world.step) < NEUTRAL_EARLY_HORIZON:
                bonus *= NEUTRAL_EARLY_EXTRA
        if LEADER_FOCUS_WEIGHT != 1.0:
            leader = _leader_owner_from_world(world, me)
            if leader is not None and int(tgt.owner) == int(leader):
                bonus *= LEADER_FOCUS_WEIGHT
        delta *= bonus

    # Follow-on hold bonus (Fix 2b — opt-in, env-gated). Surfaces the
    # B3/B4 modeling-correct snipe helpers (`_best_followon` predicts
    # follow-on + target positions at our arrival). Off by default
    # (`BASELINE_FOLLOWON_BONUS=0.0`); the bundle wrapper opts in once
    # the A/B confirms lift. Restricted to fresh captures
    # (`tgt.owner != me`) and positive-delta candidates so the bonus
    # only sweetens already-attractive captures.
    if (FOLLOWON_BONUS_WEIGHT > 0.0 and delta > 0.0
            and int(tgt.owner) != me):
        try:
            # from lib.missions.snipe import _best_followon  # local: heavy import  # inlined by bundle_agent.py
            foothold = _best_followon(
                tgt, world, model, me, FOLLOWON_RADIUS,
                arrival_eta=int(eta),
            )
        except Exception:
            foothold = None
        if foothold is not None:
            _f_target, _f_cost, _f_eta_from_t, f_hold = foothold
            delta += (
                FOLLOWON_BONUS_WEIGHT
                * float(_f_target.production)
                * float(f_hold)
            )

    if SHIP_TURN_KAPPA > 0.0:
        delta -= SHIP_TURN_KAPPA * float(ships) * float(int(wait_N) + int(eta))

    return (delta, "scored", eta)


def score_candidate_v4_joint(snap_base, launches, me: int, num_seats: int,
                              world,
                              baseline_favors: list[float],
                              favor_fn, gamma: float,
                              horizon: int,
                              skip_admissibility: bool = False,
                              ) -> tuple[float, str]:
    """Direction B: score a JOINT candidate of multiple launches in one
    fast_sim rollout. `launches` is a list of
    `(src, tgt, ships, angle, wait_N)` tuples — all injected at their
    respective wait_N steps in the SAME rollout. Leaf scoring identical
    to v4 solo: Δ = leaf − baseline_favors[horizon].

    Returns `(delta, status)` where status is:
      - 'admissibility_fail' if any wait_N==0 leg fails predict_fleet_fate
      - 'comet_expired' if any leg targets a comet that expires before eta
      - 'scored' otherwise

    For v1 simplicity, all legs typically have wait_N==0 (fire-now joint).
    Multi-wait joints are valid by construction but not enumerated yet
    (see proposer path in `choose_trajectory`).
    """
    # Per-leg admissibility filter (only meaningful for wait_N==0 legs).
    # Collect per-leg eta for the ship-turn penalty (skip/wait>0 legs → 0).
    leg_etas: list[int] = []
    for src, tgt, ships, angle, wait_N in launches:
        if skip_admissibility or int(wait_N) != 0:
            leg_etas.append(0)
            continue
        fate = predict_fleet_fate(src, tgt, angle, ships, world)
        if fate.outcome == "sun":
            return (float("-inf"), "admissibility_fail")
        if fate.outcome == "oob":
            return (float("-inf"), "admissibility_fail")
        if fate.outcome == "timeout":
            return (float("-inf"), "admissibility_fail")
        if fate.outcome == "planet":
            return (float("-inf"), "admissibility_fail")
        if int(tgt.id) in world.comet_ids:
            life = comet_remaining_lifetime(int(tgt.id), world)
            if life is None or life <= int(fate.step):
                return (float("-inf"), "comet_expired")
        leg_etas.append(int(fate.step))

    # Clamp horizon to baseline length.
    if horizon >= len(baseline_favors):
        horizon = len(baseline_favors) - 1

    # Build the inject schedule keyed by wait_N step.
    inject_at: dict[int, list] = {}
    for src, tgt, ships, angle, wait_N in launches:
        inject_at.setdefault(int(wait_N), []).append(
            [int(src.id), float(angle), int(ships)],
        )

    snap = fs_clone(snap_base)
    # Defensive emits computed once from tick-0 obs, attached to the
    # earliest inject step (typically wait_N=0). Matches the
    # score_candidate_v4 wiring — see comment there.
    me_defense_emits: list = []
    if _ME_DEFENDS_ENABLED:
        me_defense_emits = _me_defensive_action(snap, me)
    earliest_inject_t = min(inject_at.keys()) if inject_at else -1

    for t in range(horizon):
        if snap.fake_env.done:
            break
        actions = opp_actions_for_snap(snap, me, num_seats)
        if t in inject_at:
            base_actions = list(inject_at[t])
            if t == earliest_inject_t and me_defense_emits:
                base_actions = base_actions + list(me_defense_emits)
            actions[me] = base_actions
        elif _ME_REACTS_ENABLED:
            actions[me] = _me_reactive_action(snap, me)
        snap = fs_step(snap, actions, in_place=True)

    leaf = favor_fn(snap.state[me].observation, me, num_seats, gamma=gamma)
    delta = leaf - baseline_favors[horizon]

    # NEUTRAL_BONUS / LEADER_FOCUS for joints: apply when EVERY leg
    # targets the preferred owner. This keeps the joint Δ unitary
    # without per-leg attribution (the joint EV is one shared rollout).
    if delta > 0.0 and launches:
        bonus = 1.0
        if NEUTRAL_BONUS_WEIGHT != 1.0:
            if all(int(L[1].owner) == -1 for L in launches):
                bonus *= NEUTRAL_BONUS_WEIGHT
                if int(world.step) < NEUTRAL_EARLY_HORIZON:
                    bonus *= NEUTRAL_EARLY_EXTRA
        if LEADER_FOCUS_WEIGHT != 1.0:
            leader = _leader_owner_from_world(world, me)
            if (leader is not None
                    and all(int(L[1].owner) == int(leader) for L in launches)):
                bonus *= LEADER_FOCUS_WEIGHT
        delta *= bonus

    if SHIP_TURN_KAPPA > 0.0:
        penalty = 0.0
        for (src, tgt, ships, angle, wait_N), eta_leg in zip(launches, leg_etas):
            penalty += float(ships) * float(int(wait_N) + int(eta_leg))
        delta -= SHIP_TURN_KAPPA * penalty

    return (delta, "scored")


def predict_opp_responses(world, me: int, num_seats: int,
                          ) -> list[tuple[int, int, int, int]]:
    """1-turn opp lookahead: project each enemy source's likely best
    launch into a list of (target_pid, eta, opp_owner, ships) tuples
    that can be merged into our arrival ledger.

    Heuristic per opp source: scan its `OPP_NEAREST_K` nearest non-opp
    targets; pick the first one whose straight-line trajectory is
    admissible (no sun, no oob, no comet collision). Project a fleet
    of `OPP_SHIP_FRACTION × src.ships` ships.

    Closes Gap 2 of v1's 0/32 failure: composite-head A/B's K-step
    rollout simulates opp counter-launches via lib.opp_model.
    lite_greedy_policy; v1 trajectory chooser had no opp model so
    every candidate was scored as if the opp would play idle. With
    this projection, our score_candidate sees the ledger with the
    enemy's likely counter-fleet already accounted for.

    First-pass heuristic; not a full opp model. May:
      - Overestimate opp competence (assumes they pick optimal target).
      - Underestimate launches (only 1 per source).
      - Miss multi-target threats (e.g. gang-ups).
    All tolerable for a first-cut A/B; refine if results are promising.
    """
    opp_arrivals: list[tuple[int, int, int, int]] = []
    all_planets = list(world.planets_by_id.values())
    for opp_id in range(num_seats):
        if opp_id == me:
            continue
        opp_planets = [p for p in all_planets if int(p.owner) == opp_id]
        for opp_src in opp_planets:
            if int(opp_src.ships) < OPP_MIN_SHIPS:
                continue
            # Nearest non-opp targets.
            others = sorted(
                ((math.hypot(p.x - opp_src.x, p.y - opp_src.y), p)
                 for p in all_planets
                 if int(p.owner) != opp_id and int(p.id) != int(opp_src.id)),
                key=lambda d_p: d_p[0],
            )
            ships = max(1, int(int(opp_src.ships) * OPP_SHIP_FRACTION))
            for _d, opp_tgt in others[:OPP_NEAREST_K]:
                angle = math.atan2(opp_tgt.y - opp_src.y,
                                   opp_tgt.x - opp_src.x)
                fate = predict_fleet_fate(opp_src, opp_tgt, angle,
                                          ships, world)
                if fate.outcome == "target":
                    opp_arrivals.append(
                        (int(opp_tgt.id), int(fate.step), opp_id, ships),
                    )
                    break  # 1 projection per opp source is enough
    return opp_arrivals


def merge_ledgers(base: dict, projected: list[tuple[int, int, int, int]],
                  ) -> dict:
    """Add projected (target_pid, eta, owner, ships) tuples into a copy
    of `base` (per-planet list of (eta, owner, ships))."""
    out = {pid: list(v) for pid, v in base.items()}
    for tgt_pid, eta, owner, ships in projected:
        out.setdefault(tgt_pid, []).append((eta, owner, ships))
    return out


def choose_trajectory(snap_base, prerank, baseline_favors,
                      me: int, num_seats: int, wallclock_ms: float,
                      min_horizon: int, max_horizon: int, gamma: float,
                      world, model,
                      reserved_srcs: set[int] | None = None,
                      reserved_for_new_commits: set[int] | None = None,
                      ) -> tuple[list[list], list[dict]]:
    """Drop-in alternative to `chooser.choose`.

    Returns `(moves, commits)`:
      `moves`   — fire-now action list `[[src_id, angle, ships], ...]`
                  to emit this turn.
      `commits` — `wait_N > 0` winners that the agent should remember
                  across turns. Each is a dict with keys `src_id`,
                  `tgt_id`, `ships_planned`, `angle_original`,
                  `wait_remaining`, `commit_step`. The agent's ledger
                  (`agents/baseline/main._PENDING_LAUNCHES`) ticks these
                  down and fires them when `wait_remaining` reaches 0
                  (gated on `BASELINE_LEDGER=on`). When the ledger is
                  off, commits are discarded — behaviour identical to
                  the pre-ledger chooser.

    `reserved_srcs` — set of source ids that the chooser should not
    fire-now-emit from this turn (ledger is firing them via due_moves,
    or hard-ledger blocks them entirely while a commit is in flight).
    `reserved_for_new_commits` — set of source ids that already have a
    surviving ledger entry. The chooser must not add a SECOND wait
    commit for these (stacking causes duplicate emits at fire time).
    When `None`, defaults to `reserved_srcs` (hard semantics).

    The `snap_base` / `baseline_favors` / `min_horizon` / `max_horizon`
    / `gamma` args are unused (kept for signature parity with
    `chooser.choose` so the dispatcher in `main.py` is a simple swap).
    The trajectory chooser doesn't roll out and doesn't need an idle
    baseline.

    v2 (2026-05-17 PM):
    - 1-turn opp lookahead: predict_opp_responses projects each opp
      source's best launch; ledger merged BEFORE scoring (every
      predict_garrison_at sees the pessimistic state).
    - Multi-launch budget: drops "1 launch per source" dedup; tracks
      ship sub-budget per source.
    """
    if reserved_srcs is None:
        reserved_srcs = set()
    if reserved_for_new_commits is None:
        reserved_for_new_commits = reserved_srcs
    if not prerank:
        return [], []

    deadline = time.perf_counter() + wallclock_ms / 1000.0

    # v4 (default, 2026-05-17 PM): Δ-from-idle-baseline scoring with
    # favor leaf. Replaces v3's binary owner-check leaf — see concept
    # at knowledge-base/concepts/probability-of-winning-framework.md.
    # Use BASELINE_CHOOSER=trajectory_v3 to force the v3 (binary leaf)
    # path for A/B comparison.
    use_v3 = (
        os.environ.get("BASELINE_CHOOSER", "").strip().lower()
        == "trajectory_v3"
    )
    skip_filter = (
        os.environ.get("TRAJECTORY_SKIP_ADMISSIBILITY", "").strip().lower()
        == "on"
    )
    favor_fn = select_favor_fn()  # honours BASELINE_VALUE_HEAD env var

    # Pre-pass: find the largest horizon we'll need so the baseline runs
    # deep enough for every candidate (including wait_N>0, whose proposer
    # horizon already accounts for the wait via
    # `w_horizon = max(w_wait + w_eta + SIM_SETTLE_TURNS, MIN_HORIZON)`).
    max_horizon_seen = 0
    for cheap_delta, src, tgt, ships, angle, eta_hint, h, wait_N in prerank:
        if int(h) > max_horizon_seen:
            max_horizon_seen = int(h)

    baseline_favors: list[float] = []
    if not use_v3 and max_horizon_seen > 0:
        baseline_favors = build_trajectory_baseline(
            snap_base, me, num_seats, max_horizon_seen, favor_fn, gamma,
        )

    # Wallclock budgeting (mirror composite chooser pattern). Probe per-
    # step + per-leaf cost to size the safe_deadline pre-bail. The hard
    # cap stays at N_VALIDATE (generous); safe_deadline is the real
    # binder so score_candidate_v4's uninterruptible rollout never
    # starts past the cliff. Closes the n=64 A/B max=2416ms overrun
    # (1000ms env cap) without the N_VALIDATE=60 candidate-breadth
    # regression (57.8% vs pre-fix 65.6% in the post-N=60 A/B).
    cap = N_VALIDATE
    per_cand_ms = 0.0
    if not use_v3:
        remaining_ms = max(50.0, (deadline - time.perf_counter()) * 1000.0)
        _, per_cand_ms = affordable_validate_cap(
            snap_base, me, num_seats, max_horizon, remaining_ms,
            min_horizon, gamma,
        )
    safe_deadline = deadline - (per_cand_ms / 1000.0)

    scored: list[tuple] = []
    solo_winners: set[int] = set()  # src_ids whose solo scored Δ>0
    cand_count = 0
    for cheap_delta, src, tgt, ships, angle, eta_hint, prop_horizon, wait_N in prerank:
        if cand_count >= cap:
            break
        if not use_v3 and time.perf_counter() > safe_deadline:
            break
        # Skip candidates the ledger has already accounted for. A
        # wait_N>0 candidate from a src with a surviving commit would
        # stack a second commit — duplicate emit at fire time. A
        # wait_N==0 candidate from a reserved src would conflict with
        # the ledger's fire-now this turn (hard mode) or has no impact
        # in soft mode (where reserved_srcs only includes srcs firing
        # this turn).
        sid_ = int(src.id)
        if int(wait_N) > 0:
            if sid_ in reserved_for_new_commits:
                continue
        else:
            if sid_ in reserved_srcs:
                continue
        cand_count += 1
        if use_v3:
            # v3 path: fire-now-only (binary leaf doesn't generalise to
            # wait_N>0 trivially). Skip wait_N>0 in the v3 path.
            if int(wait_N) != 0:
                continue
            score, status, _ = score_candidate_dyn(
                snap_base, src, tgt, int(ships), float(angle),
                me, num_seats, world,
            )
            if status in ("captured",) and score > 0.0:
                scored.append((score, src, tgt, ships, angle, wait_N))
        else:
            score, status, _ = score_candidate_v4(
                snap_base, src, tgt, int(ships), float(angle),
                me, num_seats, world,
                baseline_favors, favor_fn, gamma,
                horizon=int(prop_horizon),
                skip_admissibility=skip_filter,
                wait_N=int(wait_N),
                model=model,
            )
            passes = (
                score > MIN_DELTA if MIN_DELTA == 0.0
                else score >= MIN_DELTA
            )
            if status == "scored" and passes:
                scored.append((score, src, tgt, ships, angle, wait_N))
                # Track sources with viable solo (for joint gating).
                solo_winners.add(int(src.id))

    # Direction B v3 (2026-05-18 PM): 2P-only gate added after v2's
    # 4P regression (4/32 first-place = 12.5pct in 8-seed × 4-seat
    # rotation vs 3x hybrid). 4P joint commits 2 srcs against one of
    # 3 opponents, leaving the other 2 opps free to attack our weakened
    # planets. 2P-only is the same defensive shape as favor_hybrid_spatial
    # in commit 558bd61. 2P joint v2 A/B 38/64 = 59.4pct (Wlo=0.471,
    # INCONCL-but-positive vs hybrid).
    # 2026-05-21: 4P gate lifted when BASELINE_JOINT_AGGR=1 OR when the
    # explicit BASELINE_JOINT_4P=1 env var is set. Without this, AGGR's
    # `used_tgts` lift creates a silent double-count in 4P: solo emits
    # can stack on the same target but each is scored in an independent
    # rollout that assumed it was alone. Lifting the gate runs the real
    # joint scoring so combined-EV is computed once. Defensive fallout
    # (the 2026-05-18 audit's concern) is now handled by the reinforce
    # post-pass in `agents/baseline/main.emit_threat_reinforcements`.
    joint_4p_allowed = (
        JOINT_LIFT_USED_TGTS
        or os.environ.get("BASELINE_JOINT_4P", "0").strip() == "1"
    )
    joint_enabled = (
        os.environ.get("BASELINE_JOINT", "0").strip() == "1"
        and (int(num_seats) <= 2 or joint_4p_allowed)
    )
    if (joint_enabled and not use_v3
            and time.perf_counter() <= safe_deadline):
        # Group prerank by target_id. Take top-K solo candidates per
        # target by cheap_delta; pair-enumerate.
        by_tgt: dict[int, list] = {}
        for cd, src, tgt, ships, angle, eta_hint, ph, wn in prerank:
            if int(wn) != 0:
                continue  # v1: fire-now joints only
            if int(src.id) in reserved_srcs:
                continue  # ledger is firing from this src this turn
            by_tgt.setdefault(int(tgt.id), []).append(
                (float(cd), src, tgt, int(ships), float(angle), int(ph)),
            )
        joint_count = 0
        for tgt_id, cands in by_tgt.items():
            if len(cands) < 2:
                continue
            cands.sort(key=lambda c: -c[0])
            top = cands[:JOINT_TOP_K_PER_TARGET]
            for i in range(len(top)):
                if joint_count >= JOINT_MAX_PAIRS:
                    break
                if time.perf_counter() > safe_deadline:
                    break
                for j in range(i + 1, len(top)):
                    if joint_count >= JOINT_MAX_PAIRS:
                        break
                    if time.perf_counter() > safe_deadline:
                        break
                    ca, cb = top[i], top[j]
                    if int(ca[1].id) == int(cb[1].id):
                        continue  # same source → not a joint
                    # Gate: at least one constituent must be a FAILING
                    # solo. If both srcs already have viable solo
                    # captures, joint over-bundles them and the emit
                    # logic would lose the cheaper independent path.
                    if (int(ca[1].id) in solo_winners
                            and int(cb[1].id) in solo_winners):
                        continue
                    launches = [
                        (ca[1], ca[2], ca[3], ca[4], 0),
                        (cb[1], cb[2], cb[3], cb[4], 0),
                    ]
                    jh = max(int(ca[5]), int(cb[5]))
                    j_score, j_status = score_candidate_v4_joint(
                        snap_base, launches, me, num_seats, world,
                        baseline_favors, favor_fn, gamma,
                        horizon=jh, skip_admissibility=skip_filter,
                    )
                    joint_count += 1
                    j_passes = (
                        j_score > MIN_DELTA if MIN_DELTA == 0.0
                        else j_score >= MIN_DELTA
                    )
                    if j_status == "scored" and j_passes:
                        scored.append((j_score, "joint", launches))

    if not scored:
        return [], []

    scored.sort(key=lambda c: -c[0])

    # Emit logic — match composite chooser (`agents/baseline/chooser.choose`)
    # for parity. 1 launch per source per turn, 1 per target. For joints
    # (tagged 'joint' tuples), require ALL of its sources and targets to
    # be free; commit all legs together.
    used_srcs: set[int] = set()
    used_tgts: set[int] = set()
    moves: list[list] = []
    commits: list[dict] = []
    commit_step = int(world.step) if world is not None else 0
    for entry in scored:
        # Joint candidates are 3-tuples: (score, 'joint', launches).
        if len(entry) == 3 and entry[1] == "joint":
            _score, _tag, launches = entry
            if any(int(L[0].id) in used_srcs for L in launches):
                continue
            if (not JOINT_LIFT_USED_TGTS
                    and any(int(L[1].id) in used_tgts for L in launches)):
                continue
            for src, tgt, ships, angle, wait_N in launches:
                used_srcs.add(int(src.id))
                used_tgts.add(int(tgt.id))
                if int(wait_N) == 0:
                    moves.append([int(src.id), float(angle), int(ships)])
            continue
        # Solo: legacy 6-tuple (score, src, tgt, ships, angle, wait_N).
        _score, src, tgt, ships, angle, wait_N = entry
        sid, tid = int(src.id), int(tgt.id)
        if sid in used_srcs:
            continue
        if not JOINT_LIFT_USED_TGTS and tid in used_tgts:
            continue
        used_srcs.add(sid)
        used_tgts.add(tid)
        if int(wait_N) == 0:
            moves.append([sid, float(angle), int(ships)])
        else:
            # Wait-N winner — emit nothing this turn; instead surface
            # as a commit. The agent's ledger (when BASELINE_LEDGER=on)
            # will tick this down and fire at wait_N == 0.
            commits.append({
                "src_id": sid,
                "tgt_id": tid,
                "ships_planned": int(ships),
                "angle_original": float(angle),
                "wait_remaining": int(wait_N),
                "commit_step": commit_step,
            })
    return moves, commits

# === agent ===
"""baseline — clean modular re-implementation of v15 (live champion μ=1115.5).

Pipeline (per turn):
  1. proposer.propose       enumerate fire-now + multi-wait grid, cheap-rank,
                            dedup by (src, tgt, wait_band).
  2. chooser.build_idle_baseline   precompute favor under (me-idle, opp-reactive).
  3. chooser.choose         validate top candidates with fast_sim K-step rollout,
                            emit greedy non-dogpile moves.

Knobs (env var overrides, all optional):
  BASELINE_GAMMA              PV-discount γ for favor() and cheap-rank.   default 0.99
  BASELINE_WALLCLOCK_MS       per-turn validate budget (env actTimeout=1000).
                                                                          default 600
  ORBIT_WARS_PARITY_WALLCLOCK_MS    bundle-parity override (very large
                                    value disables mid-loop deadline bail
                                    so the agent is a pure function of obs).
"""


import math
import os

# Production default: hybrid value head (composite in 2P, A2-favor in 4P).
# `setdefault` lets local A/B drivers (fast.py) override via env var without
# patching source, while submission-bundle / Kaggle-runner sees hybrid out
# of the box. See agents/baseline/value.select_favor_fn for the dispatch.
os.environ.setdefault("BASELINE_VALUE_HEAD", "hybrid")

# Production default: trajectory chooser. v4 with wait_N>0 + wallclock
# budgeting hits 42/64 = 65.6pct Wlo=0.534 vs v15 (n=64), point-estimate
# +3pp over composite_a2's 40/64 = 62.5pct in the same A/B, with better
# max-turn-ms (1077 vs 1292). The trajectory path is deterministic on
# sun/oob/expired-comet failure modes (predict_fleet_fate filter) and
# was the architectural reframe completed in this session. Local A/B
# drivers can force the composite path by setting BASELINE_CHOOSER to
# any value other than "trajectory" (e.g. "composite").
os.environ.setdefault("BASELINE_CHOOSER", "trajectory")

# Direction B v3 (2026-05-18 PM): joint candidate enumeration enabled
# by default. 2P A/B: joint vs hybrid = 38/64 = 59.4pct, Wlo=0.471,
# Whi=0.705 (INCONCLUSIVE-but-positive). 2P-only gate in chooser
# (num_seats <= 2 check) preserves 4P behaviour (4P regressed without
# gate at 12.5pct first-place). Wallclock OK: bench max=891ms,
# p95=703ms, zero >1000ms. Set BASELINE_JOINT=0 to disable.
os.environ.setdefault("BASELINE_JOINT", "1")

# H1 — post-chooser idle drain (2026-05-18) — DISABLED BY DEFAULT.
# Audit `audit/replays/idle-trajectory-2026-05-17.md` measured 43.8pct
# isolated ship-turns in trajectory champion (mu=1271.8). H1 attempted
# to drain rear sources via post-chooser reinforce launches. A/B vs
# hybrid reference at n=32: **11/32 = 34.4pct, Wlo=0.204, max-ms=1528
# — FAIL**. The chooser's decision to leave rear planets idle is
# CORRECTLY calibrated reserve-holding; H1's forced emissions weaken
# defense without compensating capture-EV. Spatial-leaf head (commit
# b5f5296) failed for the same root cause. The 43.8 pct isolated is
# not a leak — it's correctly-held reserve. See audit/2026-05-18-
# spatial-leaf-negative-result.md and audit/2026-05-18-h1-idle-drain-
# negative-result.md. Default OFF; opt-in via BASELINE_IDLE_DRAIN=1.
IDLE_DRAIN_THRESHOLD = int(os.environ.get("BASELINE_IDLE_DRAIN_THRESHOLD", "30"))
IDLE_REAR_THRESHOLD = float(os.environ.get("BASELINE_IDLE_REAR_THRESHOLD", "35.0"))
IDLE_DRAIN_RESERVE = int(os.environ.get("BASELINE_IDLE_DRAIN_RESERVE", "5"))
IDLE_DRAIN_ENABLED = os.environ.get("BASELINE_IDLE_DRAIN", "0") == "1"

# Reinforce-emit post-pass (2026-05-21). Wires `propose_reinforce_missions`
# (lib/missions/reinforce.py) into the chooser's emit path. Distinct from
# `drain_idle_rear` (which the 2026-05-18 audit falsified as weakening
# defense): this only fires for OUR planets predicted to flip to enemy
# within model.horizon. Triggered by PI live-game observation (4P seed
# 914393430): a +5 prod planet fell while rear sources held reserves.
# Default OFF; opt-in via BASELINE_REINFORCE_EMIT=1.
REINFORCE_EMIT_ENABLED = os.environ.get("BASELINE_REINFORCE_EMIT", "0") == "1"
REINFORCE_MIN_PROD = int(os.environ.get("BASELINE_REINFORCE_MIN_PROD", "2"))
REINFORCE_MAX_LAUNCHES = int(os.environ.get("BASELINE_REINFORCE_MAX", "3"))

# Anticipated-threat (preemptive) reinforce — direction (b) from
# PI 2026-05-21 directive "mobilize idle planets toward planets that
# need them." Fires for friendly destinations with inbound enemy fleets
# that thin defenders below safety margin, even if T_loss isn't predicted
# yet. Distinct from strict propose_reinforce_missions (T_loss < horizon
# only) and from drain_idle_rear (blanket "rear -> closer friend").
ANTICIPATE_ENABLED = os.environ.get("BASELINE_REINFORCE_ANTICIPATE", "0") == "1"
ANTICIPATE_MIN_PROD = int(os.environ.get("BASELINE_REINFORCE_ANTICIPATE_MIN_PROD", "3"))
ANTICIPATE_MARGIN = float(os.environ.get("BASELINE_REINFORCE_ANTICIPATE_MARGIN", "1.3"))

# Smart stagnant-rear drain (2026-05-21). Distinct from `drain_idle_rear`
# (falsified 2026-05-18 at n=32, 11/32=34.4pct: fixed thresholds drained
# correctly-held reserves). This version uses DYNAMIC expected_reserve
# scaled by production, requires src.ships > 2x reserve (so only genuine
# excess is drained), and hard-gates on "zero inbound enemy" (not the
# looser time_to_enemy_threat heuristic). Origin: PI 2026-05-21 trace
# of sub 52882014 showing 37/40 planets sat on 50+ idle ships for 20+
# turns. Each drain launch is physics-filtered via predict_fleet_fate.
# Default OFF; opt-in via BASELINE_STAGNANT_DRAIN=1.
STAGNANT_DRAIN_ENABLED = os.environ.get("BASELINE_STAGNANT_DRAIN", "0") == "1"
STAGNANT_RESERVE_MULT = int(os.environ.get("BASELINE_STAGNANT_RESERVE_MULT", "5"))
STAGNANT_RESERVE_FLOOR = int(os.environ.get("BASELINE_STAGNANT_RESERVE_FLOOR", "10"))
STAGNANT_THRESHOLD_MULT = float(os.environ.get("BASELINE_STAGNANT_THRESHOLD_MULT", "2.0"))
STAGNANT_MIN_IMPROVEMENT = float(os.environ.get("BASELINE_STAGNANT_MIN_IMPROVEMENT", "8.0"))
STAGNANT_MAX_LAUNCHES = int(os.environ.get("BASELINE_STAGNANT_MAX_LAUNCHES", "4"))

# Combat-stack drain (2026-05-21). Different target choice from
# drain_stagnant_rear: instead of "drain to a friendly closer to front,"
# this stacks excess directly onto a NON-OUR planet we're currently
# attacking (has friendly inbound in the ledger). Directly addresses
# PI's image observation: "our large planet sits fleets away from
# combat, we do not cluster at combat." Same dynamic-reserve guards.
COMBAT_STACK_ENABLED = os.environ.get("BASELINE_COMBAT_STACK", "0") == "1"
COMBAT_STACK_MAX_LAUNCHES = int(os.environ.get("BASELINE_COMBAT_STACK_MAX_LAUNCHES", "4"))

# Sniper bundle (2026-05-21). PI directive: "when we have idle planets
# and when it's clear that they can bundle to really shoot even across
# the whole map, fast to attack one of the biggest opponent planets,
# then do it." Trigger: total reserve > SNIPER_TOTAL_RESERVE AND at least
# one source has SNIPER_MIN_SOURCE_SHIPS idle AND a non-our planet with
# production >= SNIPER_MIN_TGT_PROD is in range. Action: largest single
# source fires the capture solo (sized to take predicted garrison +
# safety margin); optional follow-on from other idle sources to bolster
# post-capture garrison. Physics-filtered. Bounded by SNIPER_MAX_LAUNCHES.
SNIPER_ENABLED = os.environ.get("BASELINE_SNIPER", "0") == "1"
SNIPER_MIN_SOURCE_SHIPS = int(os.environ.get("BASELINE_SNIPER_MIN_SOURCE", "80"))
SNIPER_TOTAL_RESERVE = int(os.environ.get("BASELINE_SNIPER_TOTAL_RESERVE", "300"))
SNIPER_MIN_TGT_PROD = int(os.environ.get("BASELINE_SNIPER_MIN_TGT_PROD", "4"))
SNIPER_MAX_TARGETS = int(os.environ.get("BASELINE_SNIPER_MAX_TARGETS", "3"))
SNIPER_MARGIN = float(os.environ.get("BASELINE_SNIPER_MARGIN", "1.2"))
SNIPER_MAX_LAUNCHES = int(os.environ.get("BASELINE_SNIPER_MAX_LAUNCHES", "4"))
SNIPER_RESERVE_FRAC = float(os.environ.get("BASELINE_SNIPER_RESERVE_FRAC", "0.4"))

# Stateful commit ledger (2026-05-20). When `BASELINE_LEDGER=on`, the
# chooser's wait_N>0 winners are remembered across turns instead of
# being silently dropped. Each entry ticks down each turn; when
# wait_remaining hits 0 the agent emits the launch (re-aimed against
# current src/tgt geometry). See plan
# /root/.claude/plans/so-now-research-and-zany-widget.md and audit
# audit/2026-05-20-filter-rejection-trace.md.
#
# Module-level state keyed by `obs.player` so independent seats in the
# same process (eg local A/B harnesses spinning up both seats) don't
# share commitments. Cleared on `obs.step == 0` (new-match detection).
LEDGER_ENABLED = os.environ.get("BASELINE_LEDGER", "off").strip().lower() == "on"
# Mode for the ledger: "hard" (default) reserves the src across the
# wait, blocking chooser emits from it. "soft" leaves the src free
# (chooser can fire fire-now from it) and only requires enough ships
# at emit time. Set via env var BASELINE_LEDGER_MODE.
LEDGER_MODE = os.environ.get("BASELINE_LEDGER_MODE", "hard").strip().lower()
_PENDING_LAUNCHES: dict[int, list[dict]] = {}

# Opening override (2026-05-21). Cherry-picked from analytical track
# (origin/claude/strategy-axis-decision-3437). For step < OPENING_HORIZON
# (=30), run the one-shot multi-turn MILP `opening_plan` and emit
# fire_step==step_now entries from its schedule. Same three-case dispatch
# as `lib/pipeline/opening.opening_default`: (a) emit schedule entries
# fired now, (b) empty fire-now list, (c) empty schedule → fall through
# to standard chooser. Default OFF; opt-in via BASELINE_OPENING_MILP=1.
OPENING_MILP_ENABLED = os.environ.get("BASELINE_OPENING_MILP", "0") == "1"

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

# from lib.fast_sim import from_obs as fs_from_obs  # inlined by bundle_agent.py
fs_from_obs = from_obs
# from lib.fleet import speed as fleet_speed  # inlined by bundle_agent.py
fleet_speed = speed
# from lib.intent import World  # inlined by bundle_agent.py
# from lib.joint_solver.opening_planner import OPENING_HORIZON, opening_plan  # inlined by bundle_agent.py
# from lib.missions.reinforce import propose_reinforce_missions  # inlined by bundle_agent.py
# from lib.orbit import predict_relative  # inlined by bundle_agent.py
# from lib.trajectory import predict_fleet_fate  # inlined by bundle_agent.py
# from lib.world_model import WorldModel  # inlined by bundle_agent.py

# Import by explicit names so the bundler's per-line import-stripping regex
# can handle them. Single-line form is mandatory — the regex matches one
# line at a time, so multi-line parenthesised imports would leak their
# continuation lines as indented orphans. Friction tag
# `bundler-modular-agent-namespace-access-breaks-bundle` (2026-05-17).
# from agents.baseline.chooser import build_idle_baseline, choose, WALLCLOCK_BUDGET_MS  # inlined by bundle_agent.py
# from agents.baseline.proposer import propose, MAX_HORIZON, MIN_HORIZON  # inlined by bundle_agent.py


_PARITY_ENV_VAR = "ORBIT_WARS_PARITY_WALLCLOCK_MS"


def _as_dict(obs) -> dict:
    if isinstance(obs, dict):
        return obs
    return {
        "player": getattr(obs, "player", 0),
        "step": getattr(obs, "step", 0),
        "planets": list(getattr(obs, "planets", []) or []),
        "fleets": list(getattr(obs, "fleets", []) or []),
        "comets": list(getattr(obs, "comets", []) or []),
        "comet_planet_ids": list(getattr(obs, "comet_planet_ids", []) or []),
        "angular_velocity": float(getattr(obs, "angular_velocity", 0.0)),
    }


def _num_seats(planets, fleets) -> int:
    max_owner = -1
    for p in planets:
        if int(p.owner) > max_owner:
            max_owner = int(p.owner)
    for f in fleets:
        if int(f.owner) > max_owner:
            max_owner = int(f.owner)
    return 4 if max_owner >= 2 else 2


def _wallclock_ms() -> float:
    override = os.environ.get(_PARITY_ENV_VAR)
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    try:
        return float(os.environ.get("BASELINE_WALLCLOCK_MS", WALLCLOCK_BUDGET_MS))
    except ValueError:
        return WALLCLOCK_BUDGET_MS


def _gamma() -> float:
    try:
        return float(os.environ.get("BASELINE_GAMMA", 0.99))
    except ValueError:
        return 0.99


def _tick_ledger(me: int, world, model, omega: float) -> tuple[list[list], list[dict]]:
    """Tick pending wait commitments for `me`.

    Returns `(due_moves, surviving_pending)`:
      `due_moves`           — actions to emit this turn (one per due commit
                              that validated successfully). Re-aimed against
                              current src/tgt geometry.
      `surviving_pending`   — entries still in flight (wait_remaining > 0
                              after the decrement) plus entries whose
                              wait_remaining hit 0 but failed validation
                              (NOT included — silently dropped).

    Tick semantics:
      - Decrement every entry's `wait_remaining` by 1.
      - If `wait_remaining` reaches 0 (or already <= 0):
          * Drop if src no longer ours.
          * Drop if tgt now ours (capture goal moot — chooser may have
            redirected or another src took it).
          * Drop if src has 0 ships (nothing to send).
          * Otherwise re-aim using `proposer.aim_and_eta` and emit
            `min(ships_planned, src.ships)` toward tgt.
      - Else: keep entry alive (decrement only).

    Re-aim is essential because planets orbit between commit time and
    emit time. The proposer's original `angle_original` was correct for
    geometry at commit time; firing at the same angle N turns later
    would miss.
    """
    pending = _PENDING_LAUNCHES.get(int(me), [])
    if not pending:
        return [], []

    # from agents.baseline.proposer import aim_and_eta as _aim_and_eta  # inlined by bundle_agent.py
    _aim_and_eta = aim_and_eta

    due_moves: list[list] = []
    survivors: list[dict] = []
    for entry in pending:
        entry["wait_remaining"] = int(entry["wait_remaining"]) - 1
        if entry["wait_remaining"] > 0:
            survivors.append(entry)
            continue

        # Time to fire — validate. Record drop reason on the entry for
        # downstream telemetry (the entry is otherwise discarded after
        # this loop).
        sid = int(entry["src_id"])
        tid = int(entry["tgt_id"])
        src = world.planets_by_id.get(sid)
        tgt = world.planets_by_id.get(tid)
        if src is None or tgt is None:
            entry["drop_reason"] = "planet_missing"
            continue
        if int(src.owner) != int(me):
            entry["drop_reason"] = "src_lost"
            continue
        if int(tgt.owner) == int(me):
            entry["drop_reason"] = "tgt_now_ours"
            continue
        available = int(src.ships)
        if available <= 0:
            entry["drop_reason"] = "src_empty"
            continue
        ships = min(int(entry["ships_planned"]), available)
        if ships <= 0:
            entry["drop_reason"] = "size_zero"
            continue
        # Re-aim against the geometry that holds RIGHT NOW (planets
        # have orbited during the wait). wait_N=0 because we're firing
        # this turn.
        try:
            angle, _eta = _aim_and_eta(src, tgt, ships, omega, wait_N=0,
                                       world=world)
        except Exception:
            entry["drop_reason"] = "aim_failed"
            continue
        entry["fired_at_step"] = int(world.step)
        entry["fired_ships"] = int(ships)
        due_moves.append([sid, float(angle), int(ships)])

    return due_moves, survivors


def emit_threat_reinforcements(
    moves, planets, my_id: int, world, model, omega: float,
) -> list:
    """Append reinforce launches for OUR planets predicted to fall.

    Defense-directed: uses `propose_reinforce_missions` which scans the
    WorldModel timeline for the first `T_loss` per friendly planet, then
    proposes (src, defended) candidates feasible to arrive before the
    flip. Skips sources already in `moves` so the chooser's offensive
    plan isn't disrupted. Caps total reinforce launches at
    REINFORCE_MAX_LAUNCHES per turn.
    """
    if not REINFORCE_EMIT_ENABLED:
        return moves
    candidates = propose_reinforce_missions(world, model)
    if not candidates:
        return moves
    used_srcs: set[int] = set()
    for m in moves:
        try:
            used_srcs.add(int(m[0]))
        except (TypeError, IndexError):
            pass
    planet_by_id = {int(p.id): p for p in planets}

    def tgt_prod(M):
        p = planet_by_id.get(int(M.target_id))
        return float(p.production) if p is not None else 0.0

    candidates.sort(key=lambda M: (-tgt_prod(M), -float(M.score)))
    extras = []
    fired = 0
    for mission in candidates:
        if fired >= REINFORCE_MAX_LAUNCHES:
            break
        if mission.mission_class != "reinforce":
            continue
        sid = int(mission.src_id)
        if sid in used_srcs:
            continue
        src = planet_by_id.get(sid)
        tgt = planet_by_id.get(int(mission.target_id))
        if src is None or tgt is None:
            continue
        if int(tgt.production) < REINFORCE_MIN_PROD:
            continue
        ships = int(mission.ships)
        if int(src.ships) < ships:
            continue
        try:
            tx, ty = predict_relative(tgt, omega, int(mission.eta))
        except Exception:
            tx, ty = float(tgt.x), float(tgt.y)
        angle = math.atan2(float(ty) - float(src.y), float(tx) - float(src.x))
        # Physics safety (Rule 47): drop reinforce launches that would
        # hit the sun, go OOB, hit a non-target planet, or never arrive.
        try:
            fate = predict_fleet_fate(src, tgt, angle, ships, world)
            if fate.outcome != "target":
                continue
        except Exception:
            pass
        extras.append([sid, float(angle), int(ships)])
        used_srcs.add(sid)
        fired += 1

    if ANTICIPATE_ENABLED and fired < REINFORCE_MAX_LAUNCHES:
        extras2 = _propose_anticipated_reinforces(
            planets, used_srcs, my_id, world, model, omega,
            slots_left=REINFORCE_MAX_LAUNCHES - fired,
        )
        extras.extend(extras2)
    return list(moves) + extras


def _propose_anticipated_reinforces(
    planets, used_srcs: set[int], my_id: int, world, model, omega: float,
    slots_left: int,
) -> list:
    """Preemptive reinforce: defenders thinned by inbound enemy fleets.

    For each friendly D with prod >= ANTICIPATE_MIN_PROD and at least one
    inbound enemy fleet within model.horizon, check whether projected
    defenders cover the inbound threat by ANTICIPATE_MARGIN. If not,
    propose a launch from the nearest viable friendly source whose
    arrival ETA precedes the earliest enemy ETA.
    """
    if slots_left <= 0:
        return []
    my_planets = [p for p in planets if int(p.owner) == my_id]
    if len(my_planets) < 2:
        return []
    horizon = int(getattr(model, "horizon", 40))
    out = []
    fired = 0
    # Pre-compute friendly index for source iteration.
    friendly_by_id = {int(p.id): p for p in my_planets}
    # Score destinations by (production desc, thinness ratio asc — most-thin first).
    destinations: list[tuple] = []
    for d in my_planets:
        if int(d.production) < ANTICIPATE_MIN_PROD:
            continue
        arrivals = (model.ledger.get(int(d.id)) or []) if hasattr(model, "ledger") else []
        if not arrivals:
            continue
        # Pass 1: find earliest enemy ETA (within horizon).
        earliest_enemy_eta: int | None = None
        for (eta_arr, owner_arr, ships_arr) in arrivals:
            if int(ships_arr) <= 0:
                continue
            if int(eta_arr) > horizon:
                continue
            if int(owner_arr) != my_id:
                if earliest_enemy_eta is None or int(eta_arr) < earliest_enemy_eta:
                    earliest_enemy_eta = int(eta_arr)
        if earliest_enemy_eta is None:
            continue
        # Pass 2: enemy + friendly inbound that lands at or before the
        # first wave (later arrivals don't help defend this wave).
        enemy_inbound = 0
        friendly_inbound = 0
        for (eta_arr, owner_arr, ships_arr) in arrivals:
            if int(ships_arr) <= 0:
                continue
            if int(eta_arr) > earliest_enemy_eta:
                continue
            if int(owner_arr) == my_id:
                friendly_inbound += int(ships_arr)
            else:
                enemy_inbound += int(ships_arr)
        if enemy_inbound <= 0:
            continue
        # Projected defenders at earliest enemy arrival (ignoring
        # accruing production from the enemy's perspective; production
        # accrues for us between now and arrival).
        proj_defenders = (
            int(d.ships) + int(d.production) * int(earliest_enemy_eta)
            + friendly_inbound
        )
        # Already comfortable margin → skip.
        if proj_defenders >= enemy_inbound * ANTICIPATE_MARGIN:
            continue
        deficit = int(enemy_inbound * ANTICIPATE_MARGIN) - proj_defenders + 1
        if deficit <= 0:
            continue
        destinations.append((deficit, d, earliest_enemy_eta))
    # Highest deficit first.
    destinations.sort(key=lambda x: -x[0])
    for deficit, d, earliest_enemy_eta in destinations:
        if fired >= slots_left:
            break
        # Find nearest friendly source not already used, with enough
        # ships AND able to arrive before earliest_enemy_eta.
        best_src = None
        best_eta = None
        for s in my_planets:
            if int(s.id) == int(d.id):
                continue
            if int(s.id) in used_srcs:
                continue
            if int(s.ships) < deficit:
                continue
            dist = math.hypot(float(d.x) - float(s.x), float(d.y) - float(s.y))
            v = fleet_speed(deficit)
            if v <= 0:
                continue
            eta = int(math.ceil(dist / v))
            if eta >= int(earliest_enemy_eta):
                continue
            if best_src is None or eta < best_eta:
                best_src = s
                best_eta = eta
        if best_src is None:
            continue
        try:
            tx, ty = predict_relative(d, omega, int(best_eta))
        except Exception:
            tx, ty = float(d.x), float(d.y)
        angle = math.atan2(
            float(ty) - float(best_src.y), float(tx) - float(best_src.x),
        )
        # Physics safety (Rule 47): drop reinforce launches that would
        # hit the sun, go OOB, hit a non-target planet, or never arrive.
        try:
            fate = predict_fleet_fate(best_src, d, angle, deficit, world)
            if fate.outcome != "target":
                continue
        except Exception:
            pass
        out.append([int(best_src.id), float(angle), int(deficit)])
        used_srcs.add(int(best_src.id))
        fired += 1
    return out


def drain_idle_rear(moves, planets, my_id: int, world, model) -> list:
    """H1: append reinforce launches for rear sources the chooser didn't use.

    Idempotent post-chooser pass. Fires only when ALL of:
      - source is one of MY planets AND not in `moves`
      - source.ships > IDLE_DRAIN_THRESHOLD
      - source's min-distance to any non-our planet > IDLE_REAR_THRESHOLD
      - source has no enemy threat (model.time_to_enemy_threat is None)
      - there is an own planet strictly closer to the action than source
    Emits one launch toward that closer own planet, ships = source.ships
    minus IDLE_DRAIN_RESERVE. Each `move` is `[src_id, angle, ships]`.
    """
    if not IDLE_DRAIN_ENABLED:
        return moves
    used_srcs = set()
    for m in moves:
        try:
            used_srcs.add(int(m[0]))
        except (TypeError, IndexError):
            pass
    non_our_xy = [(float(p.x), float(p.y)) for p in planets
                  if int(p.owner) != my_id]
    if not non_our_xy:
        return moves
    my_planets = [p for p in planets if int(p.owner) == my_id]
    if len(my_planets) < 2:
        return moves  # no closer own target available

    def d_action(p):
        return min(math.hypot(float(p.x) - tx, float(p.y) - ty)
                   for tx, ty in non_our_xy)

    extras = []
    for src in my_planets:
        if int(src.id) in used_srcs:
            continue
        if int(src.ships) <= IDLE_DRAIN_THRESHOLD:
            continue
        src_d = d_action(src)
        if src_d <= IDLE_REAR_THRESHOLD:
            continue
        if model.time_to_enemy_threat(int(src.id), my_id, world) is not None:
            continue
        best_target = None
        best_d = src_d  # strict-less-than → require improvement
        for q in my_planets:
            if int(q.id) == int(src.id):
                continue
            qd = d_action(q)
            if qd >= best_d:
                continue
            best_d = qd
            best_target = q
        if best_target is None:
            continue
        ships = int(src.ships) - IDLE_DRAIN_RESERVE
        if ships < 1:
            continue
        angle = math.atan2(float(best_target.y) - float(src.y),
                           float(best_target.x) - float(src.x))
        extras.append([int(src.id), float(angle), int(ships)])
    return list(moves) + extras


def drain_stagnant_rear(moves, planets, my_id: int, world, model) -> list:
    """Drain genuinely-excess rear reserves toward closer-to-front friendlies.

    PI 2026-05-21 spec — drains only when ALL of:
      - source is one of MY planets AND not already in `moves`
      - source has ZERO inbound enemy fleets (incoming_enemy_eta is None)
      - source.ships > THRESHOLD_MULT * expected_reserve(src)
      - a strictly-closer-to-front friendly exists (action distance
        improvement >= STAGNANT_MIN_IMPROVEMENT board units)

    expected_reserve(src) = max(production * RESERVE_MULT, RESERVE_FLOOR).
    For prod=5 with defaults (5x, 10 floor): reserve=25, drain at >50.
    For prod=2: reserve=10, drain at >20.

    Each launch is filtered by predict_fleet_fate (Rule 47). At most
    STAGNANT_MAX_LAUNCHES emissions per turn (wallclock guard).

    Target choice: among friendlies that improve action-distance, pick
    the one with the LOWEST ships (proxy for "needs reinforcement").
    """
    if not STAGNANT_DRAIN_ENABLED:
        return moves
    used_srcs = set()
    for m in moves:
        try:
            used_srcs.add(int(m[0]))
        except (TypeError, IndexError):
            pass
    my_planets = [p for p in planets if int(p.owner) == my_id]
    other_planets = [p for p in planets if int(p.owner) != my_id]
    if len(my_planets) < 2 or not other_planets:
        return moves

    def d_action(p):
        return min(
            math.hypot(float(p.x) - float(q.x), float(p.y) - float(q.y))
            for q in other_planets
        )

    extras = []
    fired = 0
    for src in my_planets:
        if fired >= STAGNANT_MAX_LAUNCHES:
            break
        if int(src.id) in used_srcs:
            continue
        # Hard gate: zero inbound enemy fleet ETAs.
        if model.incoming_enemy_eta(int(src.id), my_id) is not None:
            continue
        prod = int(src.production)
        expected_reserve = max(prod * STAGNANT_RESERVE_MULT,
                               STAGNANT_RESERVE_FLOOR)
        if int(src.ships) <= STAGNANT_THRESHOLD_MULT * expected_reserve:
            continue
        src_d = d_action(src)
        # Closer-to-front friendly required; pick the lowest-ships one
        # among candidates that meet the improvement floor.
        candidates = []
        for q in my_planets:
            if int(q.id) == int(src.id):
                continue
            qd = d_action(q)
            if src_d - qd < STAGNANT_MIN_IMPROVEMENT:
                continue
            candidates.append((int(q.ships), q))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        target = candidates[0][1]
        ships_to_send = int(src.ships) - expected_reserve
        if ships_to_send < 1:
            continue
        angle = math.atan2(float(target.y) - float(src.y),
                           float(target.x) - float(src.x))
        try:
            fate = predict_fleet_fate(src, target, angle, ships_to_send, world)
            if fate.outcome != "target":
                continue
        except Exception:
            pass
        extras.append([int(src.id), float(angle), int(ships_to_send)])
        used_srcs.add(int(src.id))
        fired += 1
    return list(moves) + extras


def drain_combat_stack(moves, planets, my_id: int, world, model) -> list:
    """Stack idle excess directly onto attacks-in-progress.

    Drain target = NON-OUR planet that already has friendly fleets
    inbound (`model.ledger` entry with owner == my_id). Each excess
    rear source fires an extra launch toward the contested target
    closest to it, joining the existing wave.

    Same reserve / threshold / inbound-enemy guards as
    drain_stagnant_rear. Directly addresses PI 2026-05-21 image
    observation: "our large planet sits fleets away from combat,
    we do not cluster at combat."
    """
    if not COMBAT_STACK_ENABLED:
        return moves
    used_srcs = set()
    for m in moves:
        try:
            used_srcs.add(int(m[0]))
        except (TypeError, IndexError):
            pass
    my_planets = [p for p in planets if int(p.owner) == my_id]
    if len(my_planets) < 2:
        return moves

    contested = []
    for p in planets:
        if int(p.owner) == my_id:
            continue
        arrivals = model.ledger.get(int(p.id), [])
        friendly_in = sum(int(s) for (eta, o, s) in arrivals if o == my_id)
        if friendly_in > 0:
            contested.append(p)
    if not contested:
        return moves

    extras = []
    fired = 0
    for src in my_planets:
        if fired >= COMBAT_STACK_MAX_LAUNCHES:
            break
        if int(src.id) in used_srcs:
            continue
        if model.incoming_enemy_eta(int(src.id), my_id) is not None:
            continue
        prod = int(src.production)
        reserve = max(prod * STAGNANT_RESERVE_MULT, STAGNANT_RESERVE_FLOOR)
        if int(src.ships) <= STAGNANT_THRESHOLD_MULT * reserve:
            continue
        ships_to_send = int(src.ships) - reserve
        if ships_to_send < 1:
            continue
        target = min(
            contested,
            key=lambda t: math.hypot(float(t.x) - float(src.x),
                                     float(t.y) - float(src.y)),
        )
        angle = math.atan2(float(target.y) - float(src.y),
                           float(target.x) - float(src.x))
        try:
            fate = predict_fleet_fate(src, target, angle, ships_to_send, world)
            if fate.outcome != "target":
                continue
        except Exception:
            pass
        extras.append([int(src.id), float(angle), int(ships_to_send)])
        used_srcs.add(int(src.id))
        fired += 1
    return list(moves) + extras


def emit_sniper_strikes(moves, planets, my_id: int, world, model) -> list:
    """One-shot decisive strike at high-value enemy planets from idle reserves.

    Trigger: total reserve > SNIPER_TOTAL_RESERVE AND at least one source
    has SNIPER_MIN_SOURCE_SHIPS idle AND a non-our planet with production
    >= SNIPER_MIN_TGT_PROD is reachable.

    Action: largest single source fires the capture solo sized to take
    `predicted_garrison_at_arrival * SNIPER_MARGIN`. Optional follow-on
    launches from OTHER idle sources reinforce the capture (joining our
    garrison post-flip). Each launch physics-filtered via predict_fleet_fate.

    Affordability: total sniper ships <= SNIPER_RESERVE_FRAC * total reserve.
    """
    if not SNIPER_ENABLED:
        return moves
    used_srcs = set()
    for m in moves:
        try:
            used_srcs.add(int(m[0]))
        except (TypeError, IndexError):
            pass
    my_planets = [p for p in planets if int(p.owner) == my_id]
    enemy_planets = [
        p for p in planets
        if int(p.owner) != my_id and int(p.owner) != -1
    ]
    if not my_planets or not enemy_planets:
        return moves
    total_my_ships = sum(int(p.ships) for p in my_planets)
    if total_my_ships < SNIPER_TOTAL_RESERVE:
        return moves

    targets = sorted(
        [p for p in enemy_planets if int(p.production) >= SNIPER_MIN_TGT_PROD],
        key=lambda p: -int(p.production),
    )[:SNIPER_MAX_TARGETS]
    if not targets:
        return moves

    extras = []
    fired = 0
    sniper_ship_budget = int(total_my_ships * SNIPER_RESERVE_FRAC)
    sniper_ships_used = 0

    for tgt in targets:
        if fired >= SNIPER_MAX_LAUNCHES:
            break

        # Score every idle source by ACTUAL arrival ETA (PI 2026-05-21:
        # "close + big = fast"). ETA = dist / fleet_speed(ships); bigger
        # source = faster fleet, closer = shorter dist. Both effects baked
        # into FleetFate.step. Physics-safety filter applied here, so the
        # fastest viable arrival emerges naturally.
        scored = []
        for src in my_planets:
            if int(src.id) in used_srcs:
                continue
            if int(src.ships) < SNIPER_MIN_SOURCE_SHIPS:
                continue
            if model.incoming_enemy_eta(int(src.id), my_id) is not None:
                continue
            src_reserve = max(int(src.production) * 5, 10)
            available = int(src.ships) - src_reserve
            if available < SNIPER_MIN_SOURCE_SHIPS:
                continue
            angle = math.atan2(float(tgt.y) - float(src.y),
                               float(tgt.x) - float(src.x))
            try:
                fate = predict_fleet_fate(src, tgt, angle, available, world)
            except Exception:
                continue
            if fate.outcome != "target":
                continue
            scored.append((int(fate.step), src, float(angle), int(available)))
        if not scored:
            continue
        scored.sort(key=lambda s: s[0])

        # Primary = fastest arrival. Required = predicted garrison at
        # primary's arrival * margin.
        eta_p, primary, angle_p, primary_avail = scored[0]
        predicted = model.ships_at(int(tgt.id), eta_p)
        if predicted is None:
            predicted = float(tgt.ships) + eta_p * int(tgt.production)
        required = int(math.ceil(float(predicted) * SNIPER_MARGIN)) + 1
        if required > primary_avail:
            continue
        if sniper_ships_used + required > sniper_ship_budget:
            continue

        extras.append([int(primary.id), float(angle_p), int(required)])
        used_srcs.add(int(primary.id))
        sniper_ships_used += required
        fired += 1

        # Follow-on reinforcements from remaining sources in ETA order
        # (fastest first → joins our garrison closest to capture moment).
        for eta_f, src, angle_f, available in scored[1:]:
            if fired >= SNIPER_MAX_LAUNCHES:
                break
            if sniper_ships_used + available > sniper_ship_budget:
                continue
            extras.append([int(src.id), float(angle_f), int(available)])
            used_srcs.add(int(src.id))
            sniper_ships_used += available
            fired += 1
    return list(moves) + extras


def agent(obs, configuration=None):
    obs_d = _as_dict(obs)
    me = int(obs_d.get("player", 0))
    step = int(obs_d.get("step", 0))

    # New-match detection — clear this seat's commit ledger on step 0.
    # Both `LEDGER_ENABLED` and `BASELINE_LEDGER=on` are checked at call
    # time so harnesses can flip the env var mid-process without
    # restarting the agent module.
    ledger_on = (
        LEDGER_ENABLED
        or os.environ.get("BASELINE_LEDGER", "off").strip().lower() == "on"
    )
    if ledger_on and step == 0:
        _PENDING_LAUNCHES.pop(me, None)

    raw_planets = obs_d.get("planets", []) or []
    raw_fleets = obs_d.get("fleets", []) or []
    if not raw_planets:
        return []

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if int(p.owner) == me]
    other_planets = [p for p in planets if int(p.owner) != me]
    if not my_planets or not other_planets:
        return []

    world = World.from_obs(obs_d)
    model = WorldModel.from_world(world)
    omega = float(obs_d.get("angular_velocity", 0.0))
    num_seats = _num_seats(planets, fleets)
    gamma = _gamma()
    wallclock_ms = _wallclock_ms()

    threatened_mine = [
        p for p in my_planets
        if model.time_to_enemy_threat(int(p.id), me, world) is not None
    ]
    target_pool = other_planets + threatened_mine

    # Opening override (2026-05-21, hybrid). For step < OPENING_HORIZON
    # AND when opening_plan produced fire_step==step_now entries, emit
    # those. Cases (b) "MILP wants to wait" and (c) "empty schedule"
    # both fall through to AGGR's standard chooser — AGGR's aggressive
    # opening attacks outperform the MILP's "wait" recommendations in
    # empirical 4P testing (variant_open n=16 5/16 vs pre-patch 6/16
    # when intentional-waits were honoured).
    if OPENING_MILP_ENABLED and int(step) < OPENING_HORIZON:
        try:
            op = opening_plan(world, model, me, num_seats)
        except Exception:
            op = None
        if op is not None and op.schedule:
            opening_moves = [
                [int(e.src_id), float(e.angle), int(e.ships)]
                for e in op.schedule if int(e.fire_step) == int(step)
            ]
            if opening_moves:
                # Case (a): MILP has fire-now entries — emit and return.
                return opening_moves
            # Cases (b) and (c) fall through.

    snap_base = fs_from_obs(obs, num_seats=num_seats)

    # Trajectory-first chooser opt-in (2026-05-17). Deterministic
    # admissibility + single-tick combat prediction; no K-step rollout,
    # no leaf-value approximation. See knowledge-base/concepts/
    # trajectory-first-architecture.md. Default chooser remains the
    # K-step rollout for backward compat with the v15-line A/B baseline.
    if os.environ.get("BASELINE_CHOOSER", "").strip().lower() == "trajectory":
        # Trajectory chooser doesn't need baseline_favors (no idle baseline);
        # propose still wants a baseline_len for shape but value doesn't
        # affect the trajectory chooser's scoring.
        prerank = propose(
            my_planets, target_pool, world, model, me, omega,
            baseline_len=MAX_HORIZON + 1,
        )
        # from agents.baseline.chooser_trajectory import choose_trajectory  # inlined by bundle_agent.py

        # 1. Tick + emit the ledger's due commitments (if any). Build
        #    the reserved-srcs set so the chooser doesn't double-commit
        #    on srcs we've already scheduled.
        #
        # Mode "hard" (default): reserve src for the whole wait window —
        # chooser cannot emit anything from that src until the commit
        # fires.
        # Mode "soft": only reserve sources whose commit is FIRING this
        # turn (so the chooser can't fire-now on top of the commit's
        # emit). Sources with surviving (in-flight) entries are NOT
        # reserved, leaving them free to opportunistically fire-now via
        # the chooser. The pending commit just needs `ships_planned`
        # ships still available when wait_remaining hits 0; if not
        # enough remain, the commit drops at emit time.
        due_moves: list[list] = []
        surviving_pending: list[dict] = []
        reserved_srcs: set[int] = set()
        reserved_for_new_commits: set[int] = set()
        if ledger_on:
            due_moves, surviving_pending = _tick_ledger(
                me, world, model, omega,
            )
            mode = os.environ.get("BASELINE_LEDGER_MODE",
                                  LEDGER_MODE).strip().lower()
            # Sources firing via the ledger this turn — chooser must not
            # fire-now on top of those (duplicate-src emit).
            firing_srcs = {int(m[0]) for m in due_moves}
            pending_srcs = {int(e["src_id"]) for e in surviving_pending}
            # Always block stacking a second wait-commit on a src that
            # already has a surviving commit (regardless of mode).
            reserved_for_new_commits = firing_srcs | pending_srcs
            # Hard mode: also block fire-now from pending srcs (preserve
            # the ship reserve for the future commit). Soft mode: leave
            # pending srcs free to fire-now (commit drops at emit time
            # if not enough ships remain).
            reserved_srcs = firing_srcs if mode == "soft" \
                else firing_srcs | pending_srcs

        moves, new_commits = choose_trajectory(
            snap_base, prerank, None,
            me, num_seats, wallclock_ms,
            MIN_HORIZON, MAX_HORIZON, gamma,
            world, model,
            reserved_srcs=reserved_srcs,
            reserved_for_new_commits=reserved_for_new_commits,
        )

        # 2. Persist updated ledger (surviving + new commits) when on.
        if ledger_on:
            _PENDING_LAUNCHES[me] = surviving_pending + new_commits

        moves = due_moves + moves
        moves = emit_threat_reinforcements(moves, planets, me, world, model, omega)
        moves = drain_idle_rear(moves, planets, me, world, model)
        moves = drain_stagnant_rear(moves, planets, me, world, model)
        moves = drain_combat_stack(moves, planets, me, world, model)
        return emit_sniper_strikes(moves, planets, me, world, model)

    # ROI chooser opt-in (2026-05-19). Closed-form ROI prior + N-way
    # coalition + opp-modifier posterior; no fast_sim rollout. See
    # agents/baseline/chooser_roi.py and the plan at
    # /root/.claude/plans/okay-we-can-do-elegant-lampson.md.
    if os.environ.get("BASELINE_CHOOSER", "").strip().lower() == "roi":
        prerank = propose(
            my_planets, target_pool, world, model, me, omega,
            baseline_len=MAX_HORIZON + 1,
        )
        # from agents.baseline.chooser_roi import choose_roi  # inlined by bundle_agent.py
        step = int(obs_d.get("step", 0))
        moves = choose_roi(
            snap_base, prerank,
            me, num_seats, wallclock_ms,
            MIN_HORIZON, MAX_HORIZON, gamma,
            world, model, step,
        )
        moves = emit_threat_reinforcements(moves, planets, me, world, model, omega)
        moves = drain_idle_rear(moves, planets, me, world, model)
        moves = drain_stagnant_rear(moves, planets, me, world, model)
        moves = drain_combat_stack(moves, planets, me, world, model)
        return emit_sniper_strikes(moves, planets, me, world, model)

    baseline_favors = build_idle_baseline(
        snap_base, me, num_seats, MAX_HORIZON, gamma,
    )

    prerank = propose(
        my_planets, target_pool, world, model, me, omega,
        baseline_len=len(baseline_favors),
    )

    # Ledger lifecycle for the composite chooser path (parallel to the
    # trajectory branch above). Tick first; pass reservation sets;
    # merge with chooser output.
    composite_due: list[list] = []
    composite_surviving: list[dict] = []
    composite_reserved: set[int] = set()
    composite_reserved_new: set[int] = set()
    if ledger_on:
        composite_due, composite_surviving = _tick_ledger(
            me, world, model, omega,
        )
        mode = os.environ.get("BASELINE_LEDGER_MODE",
                              LEDGER_MODE).strip().lower()
        firing_srcs = {int(m[0]) for m in composite_due}
        pending_srcs = {int(e["src_id"]) for e in composite_surviving}
        composite_reserved_new = firing_srcs | pending_srcs
        composite_reserved = firing_srcs if mode == "soft" \
            else firing_srcs | pending_srcs

    moves, new_commits = choose(
        snap_base, prerank, baseline_favors,
        me, num_seats, wallclock_ms,
        MIN_HORIZON, MAX_HORIZON, gamma,
        world=world,
        reserved_srcs=composite_reserved,
        reserved_for_new_commits=composite_reserved_new,
    )

    if ledger_on:
        _PENDING_LAUNCHES[me] = composite_surviving + new_commits

    moves = composite_due + moves
    moves = emit_threat_reinforcements(moves, planets, me, world, model, omega)
    moves = drain_idle_rear(moves, planets, me, world, model)
    moves = drain_stagnant_rear(moves, planets, me, world, model)
    moves = drain_combat_stack(moves, planets, me, world, model)
    return emit_sniper_strikes(moves, planets, me, world, model)
