# Bundled by scripts/bundle_agent.py from agents/baseline + lib/{geometry,fleet,orbit,aim,combat,world_model,intent,trajectory,mechanism,mission,scoring,missions/snipe,missions/reinforce,missions/recapture,missions/opening,missions/drain,missions/gang_up,missions/opp_archetypes,planner,lookahead,lookahead_planner,game/interpreter,fast_sim,opp_model,v7_search,candidate_portfolios,value_heads,joint_solver/opening_planner,kinematic_table,value_features}.
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

# === inlined: lib/value_features.py ===


import math
from typing import Any

import numpy as np


FEATURE_DIM = 40
PER_SEAT_FEATURES = 16
GLOBAL_FEATURES = 8
DANGER_RADIUS = 25.0
HIGH_VALUE_PROD = 3.0
EPISODE_STEPS = 500


def _read(obs: Any, attr: str, default: Any) -> Any:
    if hasattr(obs, attr):
        return getattr(obs, attr)
    if isinstance(obs, dict):
        return obs.get(attr, default)
    return default


def _seat_block(
    me: int,
    planets: list,
    fleets: list,
    other_planet_xy: list[tuple[float, float]],
) -> np.ndarray:
    """16-feature block for seat `me`.

    `other_planet_xy` is the list of (x,y) for planets NOT owned by `me`
    (passed in to avoid recomputation across the two seat-blocks).
    """
    out = np.zeros(PER_SEAT_FEATURES, dtype=np.float32)

    my_planets = [p for p in planets if int(p[1]) == me]
    my_fleets = [f for f in fleets if int(f[1]) == me]

    planet_ships = sum(float(p[5]) for p in my_planets)
    planet_prods = [float(p[6]) for p in my_planets]
    garrisons = [float(p[5]) for p in my_planets]
    in_flight_ships = sum(float(f[6]) for f in my_fleets)
    n_planets = len(my_planets)

    out[0] = planet_ships + in_flight_ships
    out[1] = sum(planet_prods)
    out[2] = float(n_planets)
    out[3] = float(len(my_fleets))
    out[4] = in_flight_ships
    out[5] = (planet_ships / n_planets) if n_planets else 0.0
    out[6] = max(garrisons) if garrisons else 0.0
    out[7] = planet_ships
    out[8] = (sum(planet_prods) / n_planets) if n_planets else 0.0
    out[9] = max(planet_prods) if planet_prods else 0.0

    # Distance features: mean / min over my planets of dist-to-nearest-
    # non-mine-planet. If no other planets exist or we have no planets,
    # leave at 0.
    if my_planets and other_planet_xy:
        d_per_planet: list[float] = []
        for p in my_planets:
            px, py = float(p[2]), float(p[3])
            d = min(math.hypot(px - ox, py - oy) for ox, oy in other_planet_xy)
            d_per_planet.append(d)
        out[10] = float(np.mean(d_per_planet))
        out[11] = float(min(d_per_planet))

    out[12] = float(sum(1 for prod in planet_prods if prod > HIGH_VALUE_PROD))

    # Incoming threat: enemy fleets within DANGER_RADIUS of any of MY planets.
    if my_planets:
        my_planet_xy = [(float(p[2]), float(p[3])) for p in my_planets]
        threat = 0.0
        for f in fleets:
            if int(f[1]) == me or int(f[1]) < 0:
                continue
            fx, fy = float(f[2]), float(f[3])
            for px, py in my_planet_xy:
                if math.hypot(fx - px, fy - py) <= DANGER_RADIUS:
                    threat += float(f[6])
                    break
        out[13] = threat

        out[14] = float(np.mean([xy[0] for xy in my_planet_xy])) / 100.0
        out[15] = float(np.mean([xy[1] for xy in my_planet_xy])) / 100.0

    return out


def _opp_aggregated_block(
    me: int,
    num_seats: int,
    planets: list,
    fleets: list,
) -> np.ndarray:
    """16-feature block aggregated over all non-`me` seats.

    For 2P this equals the opp's seat_block. For 4P it's a roll-up:
    cumulative features (ship_total, prod_total, planet_count, in-flight)
    are summed; per-planet derived features (mean/max/dist) are computed
    on the union of all opp planets.
    """
    opp_planets = [p for p in planets if int(p[1]) != me and int(p[1]) >= 0]
    opp_fleets = [f for f in fleets if int(f[1]) != me and int(f[1]) >= 0]
    # "other planets" for distance features = MY planets (for symmetry with
    # how the me-block computes dist-to-non-mine).
    my_planet_xy = [
        (float(p[2]), float(p[3])) for p in planets if int(p[1]) == me
    ]

    out = np.zeros(PER_SEAT_FEATURES, dtype=np.float32)

    planet_ships = sum(float(p[5]) for p in opp_planets)
    planet_prods = [float(p[6]) for p in opp_planets]
    garrisons = [float(p[5]) for p in opp_planets]
    in_flight_ships = sum(float(f[6]) for f in opp_fleets)
    n_planets = len(opp_planets)

    out[0] = planet_ships + in_flight_ships
    out[1] = sum(planet_prods)
    out[2] = float(n_planets)
    out[3] = float(len(opp_fleets))
    out[4] = in_flight_ships
    out[5] = (planet_ships / n_planets) if n_planets else 0.0
    out[6] = max(garrisons) if garrisons else 0.0
    out[7] = planet_ships
    out[8] = (sum(planet_prods) / n_planets) if n_planets else 0.0
    out[9] = max(planet_prods) if planet_prods else 0.0

    if opp_planets and my_planet_xy:
        d_per_planet: list[float] = []
        for p in opp_planets:
            px, py = float(p[2]), float(p[3])
            d = min(math.hypot(px - ox, py - oy) for ox, oy in my_planet_xy)
            d_per_planet.append(d)
        out[10] = float(np.mean(d_per_planet))
        out[11] = float(min(d_per_planet))

    out[12] = float(sum(1 for prod in planet_prods if prod > HIGH_VALUE_PROD))

    # Incoming threat for opp aggregate = MY fleets within DANGER_RADIUS of
    # any opp planet.
    if opp_planets:
        opp_planet_xy = [(float(p[2]), float(p[3])) for p in opp_planets]
        threat = 0.0
        for f in fleets:
            if int(f[1]) != me:
                continue
            fx, fy = float(f[2]), float(f[3])
            for px, py in opp_planet_xy:
                if math.hypot(fx - px, fy - py) <= DANGER_RADIUS:
                    threat += float(f[6])
                    break
        out[13] = threat

        out[14] = float(np.mean([xy[0] for xy in opp_planet_xy])) / 100.0
        out[15] = float(np.mean([xy[1] for xy in opp_planet_xy])) / 100.0

    return out


def extract_features(
    obs: Any, me: int, num_seats: int = 2
) -> np.ndarray:
    """Build the 40-dim feature vector for value-head inference / training.

    `obs` is whatever the env hands the agent: a dict with `planets`,
    `fleets`, `step` keys, OR a kaggle_environments observation object
    with same attributes. Schema mirrors `agents/baseline/value.py:_read`.

    Returns a `float32` ndarray of shape (40,). Order is fixed and
    documented in this module's header — DO NOT permute without retraining
    the value head.
    """
    planets = list(_read(obs, "planets", []) or [])
    fleets = list(_read(obs, "fleets", []) or [])
    step = int(_read(obs, "step", 0))

    other_planet_xy_for_me = [
        (float(p[2]), float(p[3])) for p in planets if int(p[1]) != me
    ]

    out = np.zeros(FEATURE_DIM, dtype=np.float32)
    out[0:PER_SEAT_FEATURES] = _seat_block(
        me, planets, fleets, other_planet_xy_for_me
    )
    out[PER_SEAT_FEATURES:2 * PER_SEAT_FEATURES] = _opp_aggregated_block(
        me, num_seats, planets, fleets
    )

    # Global block.
    total_ships = sum(float(p[5]) for p in planets) + sum(
        float(f[6]) for f in fleets
    )
    total_prod = sum(float(p[6]) for p in planets if int(p[1]) >= 0)
    neutral = sum(1 for p in planets if int(p[1]) < 0)
    total_planets = len(planets)

    g = 2 * PER_SEAT_FEATURES
    out[g + 0] = step / float(EPISODE_STEPS)
    out[g + 1] = num_seats / 4.0
    out[g + 2] = total_planets / 16.0
    out[g + 3] = neutral / 16.0
    out[g + 4] = total_ships / 200.0
    out[g + 5] = total_prod / 50.0
    out[g + 6] = float(pv_horizon(step, 0))
    out[g + 7] = len(fleets) / 20.0

    return out

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


def _source_survives_launch(
    src, ships: int, wait_N: int, world, model, me: int,
) -> bool:
    """Bug #4 fix (2026-05-18 PM): would `src` still defend itself
    against the earliest known inbound threat after launching `ships`
    at `wait_N`?

    Returns True when:
    - no enemy threat is inbound to `src` (`time_to_enemy_threat` is
      None), OR
    - the threat is "potential" only (no fleet in the ledger; opp
      would need to launch + travel — the chooser's rollout can
      score that case better), OR
    - the launch lands strictly BEFORE the threat AND the residue
      plus production accrual up to `threat_eta` covers the threat
      force (with a +1 margin for combat resolution).

    Returns False when the launch leaves `src` vulnerable: the
    chooser's leaf rollout (horizon 25) doesn't see threats landing
    30+ ticks later, so the chooser drains exposed sources. This
    filter is a proposer-side pre-cut that doesn't depend on the
    rollout horizon. Anchored on the asdf-game (76947663) P15 pattern
    (25 ships → launched 18 → opp threat at ~10 ticks → P15 falls).
    """
    threat_eta = model.time_to_enemy_threat(int(src.id), me, world)
    if threat_eta is None:
        return True
    threat_force = sum(
        sh
        for (eta_arr, owner, sh) in model.ledger.get(int(src.id), [])
        if owner != me and eta_arr <= int(threat_eta) + WAVE_LOOKAHEAD
    )
    if threat_force <= 0:
        # Potential-launch threats only; let the chooser's rollout
        # handle the assessment. The pre-cut is for in-flight cases
        # where the trajectory is already committed.
        return True
    if int(wait_N) >= int(threat_eta):
        # Launch would happen AT or AFTER the threat lands — the
        # source has already fallen by the time we'd fire. Drop.
        return False
    growth_during_wait = int(src.production) * int(wait_N)
    residue_after_launch = int(src.ships) + growth_during_wait - int(ships)
    if residue_after_launch < 0:
        return False  # nonsensical sizing; guard
    growth_after_launch_to_threat = (
        int(src.production) * (int(threat_eta) - int(wait_N))
    )
    garrison_at_threat = residue_after_launch + growth_after_launch_to_threat
    return garrison_at_threat >= int(threat_force) + 1


def _target_holdable_after_capture(
    src, tgt, ships: int, wait_N: int, eta: int, world, model, me: int,
) -> bool:
    """Tier 2 hold-feasibility filter (PI direction 2026-05-18 PM).

    Sibling to `_source_survives_launch`: that filter protects the
    SOURCE from being drained against inbound threats; this one protects
    the TARGET from being lost back to opp on counter-recapture.

    Pattern: we launch `ships` from `src` to capture `tgt` at arrival
    step `wait_N + eta`. Could the cheapest counter-launch from a
    nearby strong opp planet recapture `tgt` before our garrison +
    production can defend? If yes, the capture is unholdable — drop.

    The chooser's rollout (horizon 25) often misses this case: long-
    distance captures land near or past horizon, so the leaf state
    never reflects the opp counter. `lite_greedy_policy` (the rollout
    opp model) doesn't specifically counter our newly-captured
    targets, so even within horizon the rollout under-credits opp
    response.

    Returns True (hold-feasible) for: reinforcing our own planets,
    captures with no opp planet within plausible counter-range,
    captures where our delivered force + production accrual beats
    every opp's counter-force.
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

    # B1 (PI 2026-05-21 / completed 2026-05-22) — when BASELINE_ORBITAL_SAFETY=1,
    # the target and each opp/ally rotate to a different position by our
    # arrival. Without this, an orbiting target far from opp NOW but close
    # at arrival_step gets a falsely-HOLDABLE verdict and we capture into
    # a recapture. Sibling fix to f1774a7 in `time_to_enemy_threat`.
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
        opp_speed = fleet_speed(int(opp.ships))
        if opp_speed <= 0:
            continue
        opp_eta_after_landing = int(math.ceil(flight / opp_speed))
        # Garrison opp must overcome = our delivered residue plus our
        # production accruing on tgt during opp's transit.
        garrison_at_recapture = (
            delivered + int(tgt.production) * opp_eta_after_landing
        )
        # Opp's budget at their chosen launch moment (just after our
        # landing). Production accrues from now until then.
        opp_launch_budget = (
            int(opp.ships) + int(opp.production) * arrival_step
        )
        opp_needed = int(math.ceil(garrison_at_recapture)) + 1
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
                entry[1],  # src
                int(entry[3]),  # ships
                int(entry[7]),  # wait_N
                world, model, me,
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

# === inlined: agents/baseline/value_learned.py ===
"""Learned value head — pure-NumPy forward pass, weights embedded inline.

The MLP weights (40 -> 128 -> 128 -> 1) live as a base64-encoded
NumPy `.npz` blob in this module so the bundler's plain-source inliner
(see `scripts/bundle_agent.py`) ships them straight into the submission
.py without needing PyTorch on the agent side.

Public surface mirrors `agents/baseline/value.py::favor`:

  favor_learned(obs, me, num_seats=2, gamma=DEFAULT_GAMMA) -> float

so the existing `select_favor_fn` dispatcher can wire it via
`BASELINE_VALUE_HEAD=learned`.

Inference budget: target median < 200 µs per call (the chooser invokes
the leaf eval ~N_candidates * K_horizon times per turn — see
`agents/baseline/chooser.py`). Hot path uses only `np.dot`, `np.maximum`,
and a single-row `(40,)` vector — no allocations beyond the small
intermediates.

Weight provenance: trained by `scripts/kaggle_value_head_kernel/train.py`
on the dataset at `data/value_head/`. After training, re-embed via
`scripts/embed_value_head_weights.py` (regenerates the WEIGHTS_B64
literal in this file).
"""


import base64
import io
import os
from typing import Any

import numpy as np

# from lib.value_features import extract_features  # inlined by bundle_agent.py

DEFAULT_GAMMA = 0.99  # mirrors agents.baseline.value.DEFAULT_GAMMA

# ---------------------------------------------------------------------------
# Embedded weights
# ---------------------------------------------------------------------------
#
# The base64 string below is regenerated by
# `scripts/embed_value_head_weights.py`. While unset (initial state) the
# inference path falls back to a zero-weights identity that returns 0.0
# for every observation — this is intentional: it lets the bundler smoke
# work end-to-end before the first training run completes.

WEIGHTS_B64 = "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"


def _load_weights() -> dict[str, np.ndarray] | None:
    """Decode the embedded base64 .npz blob into a weight dict.

    Returns None when WEIGHTS_B64 is empty (pre-training stub). Caller
    must handle the None case — `_get_weights()` does so by lazily
    falling back to the zero-net.
    """
    if not WEIGHTS_B64:
        return None
    blob = base64.b64decode(WEIGHTS_B64)
    with np.load(io.BytesIO(blob)) as d:
        return {
            "W0": d["W0"].astype(np.float32),
            "b0": d["b0"].astype(np.float32),
            "W1": d["W1"].astype(np.float32),
            "b1": d["b1"].astype(np.float32),
            "W2": d["W2"].astype(np.float32),
            "b2": d["b2"].astype(np.float32),
        }


def _zero_weights() -> dict[str, np.ndarray]:
    """Identity-ish fallback when no weights have been embedded yet.

    Returns weights producing constant-0 output. Used so the bundler
    smoke (Rule 46) can run before the first training pass completes.
    """
    return {
        "W0": np.zeros((40, 128), dtype=np.float32),
        "b0": np.zeros(128, dtype=np.float32),
        "W1": np.zeros((128, 128), dtype=np.float32),
        "b1": np.zeros(128, dtype=np.float32),
        "W2": np.zeros((128, 1), dtype=np.float32),
        "b2": np.zeros(1, dtype=np.float32),
    }


_W: dict[str, np.ndarray] | None = None


def _get_weights() -> dict[str, np.ndarray]:
    global _W
    if _W is None:
        _W = _load_weights() or _zero_weights()
    return _W


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def _forward(features: np.ndarray) -> float:
    """Pure-NumPy MLP forward: 40 -> 128 -> 128 -> 1.

    `features` must be shape `(40,)` float32. Returns a Python float so
    the caller can use it interchangeably with the hand-coded `favor()`.
    """
    w = _get_weights()
    h0 = np.maximum(0.0, features @ w["W0"] + w["b0"])
    h1 = np.maximum(0.0, h0 @ w["W1"] + w["b1"])
    out = h1 @ w["W2"] + w["b2"]
    return float(out[0])


def forward_batch(features: np.ndarray) -> np.ndarray:
    """Batched forward for offline benches / training-time eval. Not
    used on the agent hot path. `features` shape `(N, 40)` float32 ->
    `(N,)` float32.
    """
    w = _get_weights()
    h0 = np.maximum(0.0, features @ w["W0"] + w["b0"])
    h1 = np.maximum(0.0, h0 @ w["W1"] + w["b1"])
    return (h1 @ w["W2"] + w["b2"])[:, 0]


# ---------------------------------------------------------------------------
# Public surface — matches agents.baseline.value.favor
# ---------------------------------------------------------------------------


def favor_learned(
    obs: Any, me: int, num_seats: int = 2,
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """Drop-in replacement for `agents.baseline.value.favor`.

    `gamma` is intentionally unused — the network learned its own time-
    discount via the `pv_horizon` feature in the global block. Argument
    kept for signature compat with the dispatcher.
    """
    feats = extract_features(obs, me=me, num_seats=num_seats)
    return _forward(feats)


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------


def weights_loaded() -> bool:
    """True iff the embedded base64 blob is populated (post-training)."""
    return bool(WEIGHTS_B64)


def weights_summary() -> str:
    """One-liner for the agent-startup log."""
    w = _get_weights()
    n_params = sum(int(np.prod(a.shape)) for a in w.values())
    state = "trained" if weights_loaded() else "ZERO_FALLBACK"
    return (
        f"value_learned: {state}, {n_params} params, "
        f"BASELINE_VALUE_HEAD={os.environ.get('BASELINE_VALUE_HEAD', '')!r}"
    )

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
      - "learned"             -> `favor_learned` (MVP learned-value-head
                                  wedge; weights embedded in
                                  `agents/baseline/value_learned.py`).

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
    if choice == "learned":
        # from agents.baseline.value_learned import favor_learned  # inlined by bundle_agent.py
        return favor_learned
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


def score_candidate(src, tgt, ships: int, angle: float, eta_hint: int,
                    me: int, world, ledger: dict,
                    ) -> tuple[float, str, int | None]:
    """Score a single candidate launch.

    Returns `(score, status, fate_step)`:
        `status` ∈ {'captured', 'reinforced', 'bounced', 'sun', 'oob',
                    'timeout', 'comet_collision', 'comet_expired',
                    'path_blocked'}
        `fate_step` = the tick of the resolving event (None if dropped pre-flight).
    """
    fate = predict_fleet_fate(src, tgt, angle, ships, world)

    if fate.outcome == "sun":
        return (float("-inf"), "sun", fate.step)
    if fate.outcome == "oob":
        return (float("-inf"), "oob", fate.step)
    if fate.outcome == "timeout":
        return (float("-inf"), "timeout", fate.step)
    if fate.outcome == "planet":
        # Hit a non-target planet first. Could be a comet collision
        # (engine treats comets as planets) — distinguish via comet_ids.
        if fate.hit_planet_id in world.comet_ids:
            return (float("-inf"), "comet_collision", fate.step)
        return (float("-inf"), "path_blocked", fate.step)

    # outcome == "target": fleet reaches the intended planet at fate.step.
    eta = int(fate.step)

    # Comet-expired guard: if the target IS a comet and runs out of
    # path at/before our arrival, the planet won't exist for capture.
    if int(tgt.id) in world.comet_ids:
        life = comet_remaining_lifetime(int(tgt.id), world)
        if life is None or life <= eta:
            return (float("-inf"), "comet_expired", eta)

    # Sparse single-tick combat prediction. Include our hypothetical
    # arrival in the ledger so resolve_arrivals handles same-tick
    # combat correctly with any other fleets due that tick.
    base_arrivals = list(ledger.get(int(tgt.id), []))
    our_arrival = (eta, int(me), int(ships))
    pred_owner, _pred_garrison = predict_garrison_at(
        tgt, eta, base_arrivals + [our_arrival],
    )

    if pred_owner != me:
        # We didn't end up holding the planet — bounce / under-sized.
        return (-WASTE_WEIGHT * ships, "bounced", eta)

    # Leader-focus bonus: 4P-only (in 2P _leader_owner_from_world returns None).
    bonus = 1.0
    if LEADER_FOCUS_WEIGHT != 1.0:
        leader = _leader_owner_from_world(world, me)
        if leader is not None and int(tgt.owner) == int(leader):
            bonus = LEADER_FOCUS_WEIGHT

    # Neutral-capture bonus: applies when the target is currently neutral
    # (tgt.owner == -1). Optional opening-phase extra multiplier for the
    # first NEUTRAL_EARLY_HORIZON steps to accelerate territorial grab.
    if NEUTRAL_BONUS_WEIGHT != 1.0 and int(tgt.owner) == -1:
        bonus *= NEUTRAL_BONUS_WEIGHT
        if int(world.step) < NEUTRAL_EARLY_HORIZON:
            bonus *= NEUTRAL_EARLY_EXTRA

    # We hold the planet at eta. Was it ours before our arrival?
    # If the planet was already me (with no enemy interference), this
    # is reinforcement — no extra credit. Otherwise it's a capture.
    if int(tgt.owner) == me:
        # Check whether anything would flip it away from us between now
        # and eta-1 (in which case our arrival is a recapture).
        pred_owner_without_us, _ = predict_garrison_at(
            tgt, eta, base_arrivals,
        )
        if pred_owner_without_us == me:
            # Still ours without us — pure reinforcement.
            return (0.0, "reinforced", eta)
        # We recaptured a planet that would otherwise have been lost.
        # Credit the recapture like a fresh capture (production × time).
        time_remaining = max(0, EPISODE_STEPS_TOTAL - int(world.step) - eta)
        held = time_remaining
        if int(tgt.id) in world.comet_ids:
            life = comet_remaining_lifetime(int(tgt.id), world)
            if life is not None:
                held = min(held, max(0, life - eta))
        return (CAPTURE_REWARD_WEIGHT * float(tgt.production) * float(held) * bonus,
                "captured", eta)

    # Fresh capture (planet was not ours).
    time_remaining = max(0, EPISODE_STEPS_TOTAL - int(world.step) - eta)
    held = time_remaining
    if int(tgt.id) in world.comet_ids:
        life = comet_remaining_lifetime(int(tgt.id), world)
        if life is not None:
            held = min(held, max(0, life - eta))
    return (CAPTURE_REWARD_WEIGHT * float(tgt.production) * float(held) * bonus,
            "captured", eta)


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
    for src, tgt, ships, angle, wait_N in launches:
        if skip_admissibility or int(wait_N) != 0:
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
    return (leaf - baseline_favors[horizon], "scored")


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
            )
            if status == "scored" and score > 0.0:
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
                    if j_status == "scored" and j_score > 0.0:
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
