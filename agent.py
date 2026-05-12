"""Orbit Wars agent — single-file source-of-truth.

Pipeline (one turn):

    obs -> World.from_obs -> WorldModel.from_world
        -> propose_snipe_missions + propose_reinforce_missions
        -> settle_plan
        -> realize(DEFAULT_MECHANISMS)

This is the v3.5.1 agent collapsed to one file: top-10-fingerprint-aligned
aggressive snipe sizing, 4P-spoiler leader bonus, snipe + reinforce mission
classes, per-source-greedy planner with same-turn arrival ledger, and a
6-stage mechanism pipeline (validate → arrival_size → lead_aim → sun_avoid
→ path_clears_other_planets → oob_guard).

Submit this file directly (`kaggle competitions submit -c orbit-wars -f
agent.py`) — no bundling step.
"""

from __future__ import annotations

import copy
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet


# ---------------------------------------------------------------------------
# Geometry — board, sun, point/segment primitives.
# ---------------------------------------------------------------------------

BOARD_SIZE: float = 100.0
CENTER: float = 50.0
SUN_RADIUS: float = 10.0
ROTATION_RADIUS_LIMIT: float = 50.0

Point = tuple[float, float]


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def sym_hypot(dx: float, dy: float) -> float:
    """Order-independent hypot — bit-equal for (dx, dy) and (dy, dx).

    `math.hypot(a, b)` is mathematically symmetric in its arguments but
    not bit-exact under FP rounding (a² + b² and b² + a² can differ by
    1 ULP, since the addition is non-associative). For σ-paired (src,
    target) pairs in 2P self-play, the (dx, dy) arguments are permutations
    of each other, hitting this exact case. Without canonicalisation,
    a 1-ULP score difference defeats the σ-equivariant tie-break in
    settle_plan. See submission #52565034 (μ=1063.2).
    """
    ax = abs(dx)
    ay = abs(dy)
    if ax > ay:
        ax, ay = ay, ax
    return math.hypot(ax, ay)


def point_to_segment_distance(p: Point, a: Point, b: Point) -> float:
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
    return point_to_segment_distance((CENTER, CENTER), src, dst) > SUN_RADIUS + safety


# ---------------------------------------------------------------------------
# Fleet speed — log-curve, capped at maxSpeed=6.0 (Configuration table).
# ---------------------------------------------------------------------------

DEFAULT_MAX_SPEED: float = 6.0
LOG_1000: float = math.log(1000.0)


def fleet_speed(ships: int | float, max_speed: float = DEFAULT_MAX_SPEED) -> float:
    if ships <= 1:
        return 1.0
    if ships >= 1000:
        return float(max_speed)
    ratio = math.log(ships) / LOG_1000
    return 1.0 + (max_speed - 1.0) * (ratio ** 1.5)


# ---------------------------------------------------------------------------
# Orbit prediction — relative (preferred for agents; no step counter needed).
# ---------------------------------------------------------------------------


def is_orbiting(planet) -> bool:
    """planet = [id, owner, x, y, radius, ships, production]."""
    px, py, pr = planet[2], planet[3], planet[4]
    orb_r = math.hypot(px - CENTER, py - CENTER)
    return (orb_r + pr) < ROTATION_RADIUS_LIMIT


def predict_relative(current_planet, angular_velocity: float, lead_turns: float) -> Point:
    """Predicted (x, y) `lead_turns` after the obs that yielded `current_planet`."""
    px, py = current_planet[2], current_planet[3]
    dx, dy = px - CENTER, py - CENTER
    orb_r = math.hypot(dx, dy)
    cur_angle = math.atan2(dy, dx)
    new_angle = cur_angle + angular_velocity * lead_turns
    return (
        CENTER + orb_r * math.cos(new_angle),
        CENTER + orb_r * math.sin(new_angle),
    )


# ---------------------------------------------------------------------------
# Lead-aim — 5-iter fixed-point + safe-intercept fallback.
# ---------------------------------------------------------------------------

INTERCEPT_TOLERANCE = 1
SEARCH_HORIZON = 60
CONVERGENCE_XY_TOL = 0.3
MAX_ITERATIONS = 5


def flight_distance(src_xy, src_radius, target_xy, target_radius):
    """Center-to-center minus launch offset (r_src + 0.1) minus capture radius."""
    d = math.hypot(target_xy[0] - src_xy[0], target_xy[1] - src_xy[1])
    return max(0.0, d - src_radius - target_radius - 0.1)


def estimate_eta(src_xy, src_radius, target_xy, target_radius, ships):
    flight = flight_distance(src_xy, src_radius, target_xy, target_radius)
    v = fleet_speed(ships)
    if v <= 0:
        return None
    return flight / v


def search_safe_intercept(
    src_xy, src_radius, target_tuple, target_radius, ships, omega,
    horizon=SEARCH_HORIZON,
):
    """Self-consistent intercept search — fallback when fixed-point oscillates."""
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
    """Returns (aim_angle, arrival_xy, eta) or None — 5-iter fixed-point with fallback."""
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
    fb = search_safe_intercept(
        src_xy, src_radius, target_tuple, target_radius, ships, omega,
    )
    if fb is not None:
        return fb
    # Last resort — fleet still launches, physics decides.
    angle = math.atan2(ty - src_xy[1], tx - src_xy[0])
    return angle, (tx, ty), last_eta or 0.0


def swept_pair_hit(A, B, P0, P1, r):
    """Mirror of env's swept-pair collision check (orbit_wars.py:46-67)."""
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


# ---------------------------------------------------------------------------
# Combat resolver — same-step arrivals per data/README.md §combat rules.
# ---------------------------------------------------------------------------


def resolve_arrivals(
    garrison_owner: int,
    garrison_ships: float,
    arrivals: list[tuple[int, int]],
) -> tuple[int, float]:
    by_owner: dict[int, int] = {}
    for owner, ships in arrivals:
        if ships <= 0:
            continue
        by_owner[owner] = by_owner.get(owner, 0) + int(ships)

    if not by_owner:
        return garrison_owner, max(0.0, garrison_ships)

    ranked = sorted(by_owner.items(), key=lambda kv: kv[1], reverse=True)
    top_owner, top_ships = ranked[0]

    if len(ranked) > 1:
        second_ships = ranked[1][1]
        if top_ships == second_ships:
            # Two-way tie — all destroyed (rule 4).
            survivor_owner = -1
            survivor_ships = 0
        else:
            survivor_owner = top_owner
            survivor_ships = top_ships - second_ships
    else:
        survivor_owner = top_owner
        survivor_ships = top_ships

    if survivor_ships <= 0:
        return garrison_owner, max(0.0, garrison_ships)

    if garrison_owner == survivor_owner:
        return garrison_owner, garrison_ships + survivor_ships

    garrison_ships -= survivor_ships
    if garrison_ships < 0:
        return survivor_owner, -garrison_ships
    return garrison_owner, garrison_ships


# ---------------------------------------------------------------------------
# Trajectory ray-cast — full-flight collision prediction.
# ---------------------------------------------------------------------------

DEFAULT_MAX_STEPS = 200
SUN_SAFETY = 0.5  # cushion against float drift on tangent paths


@dataclass(frozen=True)
class FleetFate:
    outcome: str               # "target" | "planet" | "sun" | "oob" | "timeout"
    hit_planet_id: int | None
    step: int


def predict_fleet_fate(
    src, target, aim_angle: float, ships: int,
    world, max_steps: int = DEFAULT_MAX_STEPS,
) -> FleetFate:
    """Walk fleet forward 1 step at a time, return first collision."""
    omega = world.omega
    cos_a = math.cos(aim_angle)
    sin_a = math.sin(aim_angle)
    spawn_x = src.x + cos_a * (src.radius + 0.1)
    spawn_y = src.y + sin_a * (src.radius + 0.1)
    speed_val = fleet_speed(ships)
    if speed_val <= 0:
        return FleetFate("oob", None, 0)

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

        sun_d = point_to_segment_distance((CENTER, CENTER), fleet_old, fleet_new)
        if sun_d < SUN_RADIUS + SUN_SAFETY:
            return FleetFate("sun", None, step + 1)

        if (
            fleet_new[0] < 0.0 or fleet_new[0] > BOARD_SIZE
            or fleet_new[1] < 0.0 or fleet_new[1] > BOARD_SIZE
        ):
            return FleetFate("oob", None, step + 1)

        for pid, positions in planet_positions.items():
            # Env explicitly does not collide a fresh fleet with its source
            # on the first move.
            if pid == src_id and step == 0:
                continue
            p_old = positions[step]
            p_new = positions[step + 1]
            prad = world.planets_by_id[pid].radius
            if swept_pair_hit(fleet_old, fleet_new, p_old, p_new, prad):
                outcome = "target" if pid == target_id else "planet"
                return FleetFate(outcome, pid, step + 1)

    return FleetFate("timeout", None, max_steps)


# ---------------------------------------------------------------------------
# Intent + World + Mission — substrate dataclasses.
# ---------------------------------------------------------------------------


@dataclass
class Intent:
    """A strategy's request for a single fleet launch."""
    src_id: int
    target_id: int
    ships: int
    aim_angle: float | None = None
    arrival_xy: tuple[float, float] | None = None
    note: str = ""


@dataclass
class World:
    """Frozen-once-per-turn view over an obs."""
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
        comet_ids = (
            frozenset(int(c) for c in raw_comet_ids)
            if raw_comet_ids else frozenset()
        )
        return cls(
            my_id=my_id,
            planets_by_id=planets_by_id,
            omega=omega,
            comet_ids=comet_ids,
            step=step,
            obs_raw=obs,
        )


@dataclass
class Mission:
    """A typed fleet-launch candidate."""
    mission_class: str
    src_id: int
    target_id: int
    ships: int
    score: float
    eta: int
    note: str = ""

    def to_intent(self) -> Intent:
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


# ---------------------------------------------------------------------------
# WorldModel — arrival ledger + per-planet timeline simulator.
# ---------------------------------------------------------------------------

DEFAULT_HORIZON = 250


def fleet_target_planet(fleet, planets, max_horizon: int = DEFAULT_HORIZON):
    """Ray-cast `fleet` along its angle; return (first-hit planet, eta) or (None, None)."""
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
    """{planet_id: [(eta, owner, ships), ...]} for in-flight fleets."""
    ledger: dict[int, list[tuple[int, int, int]]] = {p.id: [] for p in planets}
    for fleet in fleets:
        target, eta = fleet_target_planet(fleet, planets, horizon)
        if target is None:
            continue
        ledger[target.id].append((eta, int(fleet.owner), int(fleet.ships)))
    return ledger


def simulate_planet_timeline(planet, arrivals, horizon: int = DEFAULT_HORIZON):
    """Step-by-step ownership/garrison sim for one planet."""
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
    t = min(max(0, int(math.ceil(arrival_turn))), timeline["horizon"])
    return timeline["owner_at"][t], timeline["ships_at"][t]


@dataclass
class WorldModel:
    """Per-turn arrival-ledger snapshot."""
    ledger: dict
    timelines: dict
    horizon: int = DEFAULT_HORIZON

    @classmethod
    def from_world(cls, world, horizon: int = DEFAULT_HORIZON):
        raw = world.obs_raw
        fleets_raw = (
            raw.get("fleets", []) if isinstance(raw, dict)
            else getattr(raw, "fleets", [])
        )
        fleets = [Fleet(*f) for f in fleets_raw]
        planets = list(world.planets_by_id.values())
        ledger = build_arrival_ledger(fleets, planets, horizon)
        timelines = {
            p.id: simulate_planet_timeline(p, ledger[p.id], horizon)
            for p in planets
        }
        return cls(ledger=ledger, timelines=timelines, horizon=horizon)

    def owner_at(self, planet_id: int, step) -> int | None:
        tl = self.timelines.get(planet_id)
        if tl is None:
            return None
        return state_at_timeline(tl, step)[0]

    def ships_at(self, planet_id: int, step) -> float | None:
        tl = self.timelines.get(planet_id)
        if tl is None:
            return None
        return state_at_timeline(tl, step)[1]

    def incoming_enemy_eta(self, planet_id: int, my_id: int) -> int | None:
        arrivals = self.ledger.get(planet_id)
        if not arrivals:
            return None
        enemy_etas = [eta for (eta, owner, ships) in arrivals if owner != my_id and ships > 0]
        if not enemy_etas:
            return None
        return min(enemy_etas)


def _comet_paths_by_id(world) -> dict[int, tuple[list, int]]:
    raw = world.obs_raw
    if raw is None:
        return {}
    comets = (
        raw.get("comets", []) if isinstance(raw, dict)
        else getattr(raw, "comets", [])
    )
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
    """Steps until `planet_id` leaves the board, or None for non-comets."""
    paths_by_id = _comet_paths_by_id(world)
    entry = paths_by_id.get(int(planet_id))
    if entry is None:
        return None
    path, path_index = entry
    return max(0, len(path) - path_index)


# ---------------------------------------------------------------------------
# Snipe mission builder — capture enemy/neutral via cost-aware ROI with
# top-10-fingerprint-aligned aggressive ship sizing.
# ---------------------------------------------------------------------------

EPISODE_STEPS = 500

# 4P spoiler: when we're ranked >=2 (below 2nd) in a 3+ player game,
# preferentially attack the current leader's planets.
LEADER_MULTIPLIER = 1.5

# Aggressive sizing — top-10 fingerprint says mean fleet 38 vs midpack 29 (+33%)
# and garrison-at-launch 11 vs midpack 22 (half). Source garrison fraction:
# 0.7 dominates 0.6/0.8/0.9 in 32-seed sweep (v3.5.1 calibration).
AGGRESSIVE_FRACTION = 0.7
AGGRESSIVE_RESERVE = 5
AGGRESSIVE_MIN_GARRISON = 12


def _player_totals(world: World) -> dict[int, float]:
    """Total ships per player across planets + in-flight fleets (for 4P leader detection)."""
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
        owner = f[1]
        ships = f[6]
        if owner == -1:
            continue
        totals[owner] = totals.get(owner, 0) + ships
    return totals


def _leader_pid(world: World) -> tuple[int | None, int | None]:
    """Returns (leader_pid, our_rank). 0-indexed. (None, None) for 2P/solo."""
    totals = _player_totals(world)
    if len(totals) < 3:
        return None, None
    ordered = sorted(totals.items(), key=lambda kv: -kv[1])
    leader_pid = ordered[0][0]
    our_rank = None
    for i, (pid, _ships) in enumerate(ordered):
        if pid == world.my_id:
            our_rank = i
            break
    return leader_pid, our_rank


def propose_snipe_missions(world: World, model: WorldModel) -> list[Mission]:
    """One snipe Mission per (our source, non-our target) pair.

    Aggressive ship sizing: when src has > AGGRESSIVE_MIN_GARRISON ships, send
    min(src.ships * 0.7, src.ships - 5), floor target_min. Otherwise send
    the minimum-viable target.ships + 1.
    """
    if not world.planets_by_id:
        return []
    my_planets = [p for p in world.planets_by_id.values() if p.owner == world.my_id]
    if not my_planets:
        return []
    targets = [p for p in world.planets_by_id.values() if p.owner != world.my_id]
    if not targets:
        return []

    step_now = int(world.step)
    leader_pid, our_rank = _leader_pid(world)
    spoiler_on = leader_pid is not None and our_rank is not None and our_rank >= 2

    missions: list[Mission] = []
    for src in my_planets:
        for t in targets:
            d = sym_hypot(t.x - src.x, t.y - src.y)
            target_min = max(1, int(t.ships) + 1)
            if src.ships > AGGRESSIVE_MIN_GARRISON:
                fraction_size = max(1, int(src.ships * AGGRESSIVE_FRACTION))
                cap = max(1, int(src.ships) - AGGRESSIVE_RESERVE)
                base_ships = max(target_min, min(fraction_size, cap))
            else:
                base_ships = target_min
            v = fleet_speed(base_ships)
            eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
            pred_owner = model.owner_at(t.id, eta)
            pred_ships = model.ships_at(t.id, eta) or 0.0
            if pred_owner == world.my_id and pred_ships >= base_ships:
                # Target will already be ours with surplus garrison; skip.
                continue
            # Comets leave the board on a known schedule; cap hold time at
            # remaining lifetime so we don't price long-run yield on an
            # about-to-depart comet.
            is_comet = t.id in world.comet_ids
            if is_comet:
                rem = comet_remaining_lifetime(t.id, world)
                time_to_hold = max(0, (rem or 0) - eta)
            else:
                time_to_hold = max(1, EPISODE_STEPS - step_now - eta)
            value = t.production * time_to_hold

            priority = 1.0
            if spoiler_on and t.owner == leader_pid:
                priority *= LEADER_MULTIPLIER
            score = priority * value / (base_ships + d + 1.0)

            missions.append(Mission(
                mission_class="snipe",
                src_id=src.id,
                target_id=t.id,
                ships=base_ships,
                score=score,
                eta=eta,
            ))
    return missions


# ---------------------------------------------------------------------------
# Reinforce mission builder — defend our planets predicted to fall.
# ---------------------------------------------------------------------------


def propose_reinforce_missions(world: World, model: WorldModel) -> list[Mission]:
    """Build reinforce candidates for (our source, our threatened planet) pairs
    where we can arrive before the predicted loss step."""
    if not world.planets_by_id:
        return []
    my_planets = [p for p in world.planets_by_id.values() if p.owner == world.my_id]
    if len(my_planets) < 2:
        return []

    horizon = model.horizon
    step_now = int(world.step)

    threatened: list[tuple] = []
    for d in my_planets:
        t_loss: int | None = None
        for t in range(1, horizon + 1):
            owner = model.owner_at(d.id, t)
            if owner is not None and owner != world.my_id:
                t_loss = t
                break
        if t_loss is None:
            continue
        post_flip_ships = model.ships_at(d.id, t_loss) or 0.0
        threatened.append((d, t_loss, post_flip_ships))

    if not threatened:
        return []

    missions: list[Mission] = []
    for dp, t_loss, attacker_strength in threatened:
        for s in my_planets:
            if s.id == dp.id:
                continue
            cost = max(1, int(attacker_strength) + 1)
            v = fleet_speed(cost)
            d_dist = sym_hypot(dp.x - s.x, dp.y - s.y)
            eta = int(math.ceil(d_dist / max(v, 1e-6))) if v > 0 else horizon + 1
            if eta >= t_loss:
                continue
            time_to_hold = max(1, EPISODE_STEPS - step_now - eta)
            value = dp.production * time_to_hold
            score = value / (cost + d_dist + 1.0)
            missions.append(Mission(
                mission_class="reinforce",
                src_id=s.id,
                target_id=dp.id,
                ships=cost,
                score=score,
                eta=eta,
            ))
    return missions


# ---------------------------------------------------------------------------
# Planner — per-source greedy with same-turn arrival ledger.
# ---------------------------------------------------------------------------


def settle_plan(
    missions: list[Mission],
    world: World,
    model: WorldModel,
) -> list[Intent]:
    """Pick at most one mission per source under a same-turn arrival ledger.

    Two invariants: (1) multiple sources MAY pick the same target when one
    source's contribution is insufficient (gang-up by accident); (2) skip
    a mission when prior this-turn picks already cover the predicted
    garrison + 1 buffer (don't waste surplus).
    """
    if not missions:
        return []

    by_src: dict[int, list[Mission]] = defaultdict(list)
    for m in missions:
        by_src[m.src_id].append(m)

    # σ-equivariant tie-break + score rounding (submission #52565034, μ=1063.2).
    # Without these two, tied or near-tied mission scores defaulted to
    # insertion-order ascending-target.id, which made σ-paired sources pick
    # the SAME target instead of σ-paired targets. That single-turn asymmetry
    # cascades to elimination over 500 steps and shows up as ~20% non-draws
    # in self-play. The geometric key -(src.x-CENTER)*(tgt.x-CENTER) negates
    # under σ (mirror through the sun), so σ-paired (src, tgt) pairs get
    # opposite-sign keys and consistent σ-equivariant choices. Score is
    # rounded to 6 decimal places so 1-ULP env-coord asymmetries (e.g. seed
    # 1's planet 12.y vs 100-planet 15.y differ by 1 ULP) don't defeat the
    # tie-break with a "false" non-tie at the primary key.
    SCORE_ROUND = 6

    def _tb(m: "Mission") -> tuple[float, float, int]:
        src = world.planets_by_id.get(m.src_id)
        tgt = world.planets_by_id.get(m.target_id)
        if src is None or tgt is None:
            return (0.0, 0.0, m.target_id)
        kx = (src.x - CENTER) * (tgt.x - CENTER)
        ky = (src.y - CENTER) * (tgt.y - CENTER)
        return (-kx, -ky, m.target_id)

    for src_id in by_src:
        by_src[src_id].sort(key=lambda m: (-round(m.score, SCORE_ROUND), _tb(m)))

    source_order = sorted(
        by_src.keys(),
        key=lambda s: (-round(by_src[s][0].score, SCORE_ROUND), _tb(by_src[s][0])),
    )

    pending: dict[int, list[tuple[int, int]]] = defaultdict(list)
    chosen: list[Mission] = []
    for src_id in source_order:
        for m in by_src[src_id]:
            already = sum(s for (e, s) in pending[m.target_id] if e <= m.eta)
            pred_enemy = model.ships_at(m.target_id, m.eta)
            if pred_enemy is None:
                pred_enemy = 0.0
            if already >= pred_enemy + 1:
                continue
            chosen.append(m)
            pending[m.target_id].append((m.eta, m.ships))
            break

    return [m.to_intent() for m in chosen]


# ---------------------------------------------------------------------------
# Mechanism pipeline — validate, arrival_size, lead_aim, sun_avoid,
# path_clears_other_planets, oob_guard.
# ---------------------------------------------------------------------------


def validate(intents: list[Intent], world: World) -> list[Intent]:
    """Drop intents that violate ownership / garrison constraints."""
    out: list[Intent] = []
    for intent in intents:
        src = world.planets_by_id.get(intent.src_id)
        if src is None or src.owner != world.my_id:
            continue
        if intent.target_id == intent.src_id:
            continue
        if intent.ships <= 0 or intent.ships > src.ships:
            continue
        out.append(intent)
    return out


def arrival_size(intents: list[Intent], world: World, model: WorldModel | None = None) -> list[Intent]:
    """Bump `ships` to cover predicted garrison at arrival (production + stacking).

    Two sources: (1) static estimate `target.ships + production * eta_prod + 1`;
    (2) WorldModel `ships_at(target, eta) + 1` for in-flight adversary stacking.
    Take max(static, model). For dynamic targets (orbiting + comets) add one
    extra production tick to match the env's entry-turn combat resolution.

    Drops the intent if our full garrison can't fund the needed size, or if
    the model predicts the target flips to us en route.
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
        target_tuple = [
            target.id, target.owner, target.x, target.y,
            target.radius, target.ships, target.production,
        ]
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
                continue
            pred_ships = model.ships_at(target.id, eta)
            if pred_ships is not None:
                needed = max(static_needed, int(math.ceil(pred_ships)) + 1)
        intent.ships = max(intent.ships, needed)
        if intent.ships > src.ships:
            continue
        out.append(intent)
    return out


def lead_aim(intents: list[Intent], world: World) -> list[Intent]:
    """Populate `aim_angle` AND `arrival_xy` for each intent.

    Orbiting non-comet targets: 5-iter fixed-point with safe-intercept
    fallback. Static targets and comets: atan2 of current position. Skips
    intents that already have aim_angle set.
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
                continue
            intent.aim_angle, intent.arrival_xy, _eta = result
        else:
            intent.aim_angle = math.atan2(target.y - src.y, target.x - src.x)
            intent.arrival_xy = (target.x, target.y)
    return intents


def sun_avoid(intents: list[Intent], world: World) -> list[Intent]:
    """Drop intents whose full-trajectory ray-cast intersects the sun."""
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


def path_clears_other_planets(intents: list[Intent], world: World) -> list[Intent]:
    """Drop intents whose flight path collides with a non-target planet."""
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


def oob_guard(intents: list[Intent], world: World) -> list[Intent]:
    """Drop intents whose fleet path exits the board (or times out) before collision."""
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
        if fate.outcome in ("oob", "timeout"):
            continue
        out.append(intent)
    return out


DEFAULT_MECHANISMS = [
    validate,
    arrival_size,
    lead_aim,
    sun_avoid,
    path_clears_other_planets,
    oob_guard,
]


# ---------------------------------------------------------------------------
# realize — apply mechanism pipeline and emit env-format actions.
# ---------------------------------------------------------------------------


def realize(intents, obs, *, mechanisms=DEFAULT_MECHANISMS, model=None) -> list[list]:
    """Apply mechanism pipeline; emit `[src_id, aim_angle, ships]` triples.

    Mechanisms with a 3-arg signature receive `model` (used by arrival_size
    for in-flight adversary stacking). Other mechanisms get `(intents, world)`.
    Routing is by `__code__.co_argcount` so the episode_postmortem wrappers
    can preserve the signature when they monkey-patch.
    """
    world = World.from_obs(obs)
    for m in mechanisms:
        code = getattr(m, "__code__", None)
        if code is not None and code.co_argcount >= 3:
            intents = m(intents, world, model)
        else:
            intents = m(intents, world)
    return [
        [i.src_id, i.aim_angle, i.ships]
        for i in intents
        if i.ships > 0 and i.aim_angle is not None
    ]


# ---------------------------------------------------------------------------
# v3-class baseline agent — used as the rollout policy by the v7 maximin
# layer below, and as the 4P fallback (maximin doesn't extend to n>=3).
# ---------------------------------------------------------------------------


def _v3_agent_impl(obs):
    """v3-class agent: aggressive-snipe sizing + 4P spoiler + σ-equiv tie-break.

    This is the policy used inside the v7 Sim<K> rollouts. Identical to
    the pre-merge top-level `agent(obs)` before the v7_minimax layer was
    bolted on.
    """
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    model = WorldModel.from_world(world)
    missions = (
        propose_snipe_missions(world, model)
        + propose_reinforce_missions(world, model)
    )
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)


def _v3() -> Callable:
    """Accessor — the callable v7's `policy` argument expects."""
    return _v3_agent_impl


# ---------------------------------------------------------------------------
# Sim<K> forward-simulation scorer.
#
# env.clone() + env.step() repeated for K turns under a fixed rollout
# policy is statistically indistinguishable from the perfect oracle at
# step 50 (AUC matched O50 to 0.000 in the Phase-2 lookahead probe).
# Used by v7_minimax to fill an N×M payoff matrix.
# ---------------------------------------------------------------------------


def env_from_obs(obs, configuration: dict | None = None):
    """Rebuild a steppable env mirroring the agent-visible state.

    Both players, viewing the same obs, derive the same `cfg["seed"]`
    from `obs.step` (when configuration doesn't already set one). This
    means their Sim<K> rollouts use identical comet-spawn RNG → σ-mirrored
    payoff matrices in 2P self-play → maximin picks agree → draws. Without
    deterministic seeding the env's RNG diverges per-process and self-play
    falls from 16/16 draws to ~50% (verified locally, 2026-05-12).
    """
    cfg = dict(configuration or {})
    if "seed" not in cfg:
        obs_step = obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)
        cfg["seed"] = int(obs_step or 0)
    env = make("orbit_wars", configuration=cfg, debug=False)
    env.reset(num_agents=2)
    snapshot_keys = (
        "planets", "fleets", "comets", "comet_planet_ids",
        "initial_planets", "angular_velocity", "step", "next_fleet_id",
    )
    obs_dict = obs if isinstance(obs, dict) else {
        k: getattr(obs, k, None) for k in snapshot_keys + ("remainingOverageTime",)
    }
    public = {k: copy.deepcopy(obs_dict[k]) for k in snapshot_keys if obs_dict.get(k) is not None}
    for i in range(2):
        env.state[i].observation.update(public)
        env.state[i].observation["player"] = i
        env.state[i].observation["remainingOverageTime"] = obs_dict.get(
            "remainingOverageTime", 60.0
        )
        env.state[i].status = "ACTIVE"
        env.state[i].reward = 0
        env.state[i].action = None
    return env


def _ship_total_by_owner(observation) -> dict[int, float]:
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


def score_joint_action(
    env,
    our_action: list,
    opp_action: list,
    K: int,
    my_id: int,
    policy: Callable,
) -> float:
    """Sim<K> with BOTH first-turn actions injected; (our - opp) at horizon."""
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
    """Average over both seat assignments to cancel env's P1-favoring tie-break.

    The Orbit Wars env has a documented seat asymmetry (P1 favored ~4:1 in
    identical self-play, audit/2026-05-10-day-1-data-inventory.md:98).
    Without this averaging, P0 and P1's maximin payoff matrices diverge
    and self-play games stop drawing.
    """
    a = score_joint_action(env, our_action, opp_action, K, my_id=0, policy=policy)
    b = score_joint_action(env, our_action, opp_action, K, my_id=1, policy=policy)
    return (a + b) / 2.0


# ---------------------------------------------------------------------------
# v7_minimax — K-step maximin agent (real game theory at action level).
#
# Per-turn algorithm:
#   1. Generate N=2 candidates: [v3 incumbent, drop-smallest-launch]
#   2. Generate M=2 opp models: [v3 from opp POV, drop-smallest of that]
#   3. Fill N×M payoff matrix via score_joint_action_symmetric.
#   4. Pick i* = argmax_i (min_j P[i,j]) — maximin, tie-break lower index.
# 4P fallback to v3 (no Nash guarantee for n>=3).
#
# Local A/B (game-theory branch commit 59ffd85):
#   v7 vs v3.4:         6W/0D/2L = 75% (8 games both sides)
#   v7 vs precision_v3: 6W/0D/2L = 75% (8 games both sides)
#   144 ms/turn avg, K=3→2 adaptive downshift, 700 ms hard deadline.
# ---------------------------------------------------------------------------

N_CANDS = 2
M_OPPS = 2
K_INIT = 3
K_FALLBACK = 2
DOWNSHIFT_MS = 300.0
HARD_DEADLINE_MS = 750.0


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _detect_num_players(planets) -> int:
    return len({p[1] for p in planets if p[1] != -1})


def _drop_smallest(action: list) -> list:
    """Return `action` with its smallest-ship launch removed.

    Ties broken σ-deterministically by removing the EARLIEST among smallest.
    Empty or single-launch input returns the empty list (maximum drop).
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


def _swap_obs_player(obs, opp_id: int):
    """Shallow-copy obs with `player` set to opp_id (for opp-POV invocation)."""
    if isinstance(obs, dict):
        obs2 = dict(obs)
        obs2["player"] = opp_id
        return obs2
    keys = (
        "player", "planets", "fleets", "angular_velocity",
        "initial_planets", "comet_planet_ids", "comets",
        "step", "next_fleet_id", "remainingOverageTime",
    )
    obs2 = {}
    for k in keys:
        v = getattr(obs, k, None)
        if v is not None:
            obs2[k] = v
    obs2["player"] = opp_id
    return obs2


def _our_candidates(obs) -> list[list]:
    """N=2 candidates: v3 incumbent + drop-smallest variant. Dedup by repr."""
    c1 = _v3()(obs)
    c2 = _drop_smallest(c1)
    seen = set()
    out: list[list] = []
    for c in (c1, c2):
        k = repr(c)
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
        if len(out) >= N_CANDS:
            break
    return out


def _opp_candidates(obs, opp_id: int) -> list[list]:
    """M=2 opp models: v3-from-opp-POV + drop-smallest. Dedup by repr."""
    swapped = _swap_obs_player(obs, opp_id)
    try:
        o1 = _v3()(swapped)
    except Exception:
        return [[]]
    o2 = _drop_smallest(o1)
    seen = set()
    out: list[list] = []
    for o in (o1, o2):
        k = repr(o)
        if k in seen:
            continue
        seen.add(k)
        out.append(o)
        if len(out) >= M_OPPS:
            break
    return out


def _maximin_pick(matrix: list[list[float]], unfilled: list[list[bool]]) -> int:
    """argmax_i (min_j P[i,j]). Ties → lower row index. Rows with all
    columns unfilled get worst = -inf (won't win unless they're the only row).
    """
    best_i = 0
    best_worst = float("-inf")
    n = len(matrix)
    if n == 0:
        return 0
    m = len(matrix[0]) if matrix[0] else 0
    for i in range(n):
        evaluated = [matrix[i][j] for j in range(m) if not unfilled[i][j]]
        worst = min(evaluated) if evaluated else float("-inf")
        if worst > best_worst:
            best_worst = worst
            best_i = i
    return best_i


def agent(obs):
    """v7_minimax: 2-step lookahead, maximin over a 2-element opp class.

    4P (or n>=3) falls back to the v3 baseline (no Nash guarantee).
    """
    my_id = int(_obs_get(obs, "player", 0))
    planets = _obs_get(obs, "planets", []) or []

    if _detect_num_players(planets) != 2:
        return _v3()(obs)

    opp_id = 1 - my_id

    C = _our_candidates(obs)
    if len(C) <= 1:
        return C[0] if C else []
    O = _opp_candidates(obs, opp_id)

    try:
        env = env_from_obs(obs)
    except Exception:
        return C[0]

    N = len(C)
    M = len(O)
    P: list[list[float]] = [[0.0] * M for _ in range(N)]
    unfilled: list[list[bool]] = [[True] * M for _ in range(N)]

    t0 = time.monotonic()
    K = K_INIT

    # Row 0 (v3 incumbent) evaluated first and in full — its worst-case
    # has to be honest because it's the conservative tie-break fallback.
    # Row 1+ runs O0 → O1 (most aggressive opp first) so a mid-row bail
    # still leaves the row's worst-against-aggression honest.
    for i in range(N):
        for j in range(M):
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            if i > 0 and elapsed_ms > HARD_DEADLINE_MS:
                break
            if i > 0 and elapsed_ms > DOWNSHIFT_MS and K == K_INIT:
                K = K_FALLBACK
            try:
                P[i][j] = score_joint_action_symmetric(
                    env, C[i], O[j], K=K, policy=_v3(),
                )
                unfilled[i][j] = False
            except Exception:
                P[i][j] = float("-inf")
                unfilled[i][j] = False
        else:
            continue
        break  # exited inner loop via budget bail

    i_star = _maximin_pick(P, unfilled)
    return C[i_star]
