"""Precision intercept solver.

For each (source planet, target, ship_count) triple, compute the launch angle
and ETA that lands the fleet on the (possibly moving) target while passing every
safety check (sun, OOB, premature planet collision).

All shots emitted are GUARANTEED to land — they pass the same swept-pair check
the engine uses, and have been verified to clear all obstacles on every flight tick.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from agents.precision import sim


# --- Data ----------------------------------------------------------------

@dataclass(frozen=True)
class PlanetView:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int
    is_comet: bool = False
    comet_group: dict | None = None  # if is_comet, the engine's group dict
    comet_index: int | None = None   # index within the group


@dataclass(frozen=True)
class Shot:
    src_id: int
    tgt_id: int
    eta: int            # number of flight ticks (k); arrival on tick eta-1 (0-indexed)
    ship_count: int     # S
    angle: float        # launch direction
    arrival_xy: tuple[float, float]
    arrival_ships: int  # ship count at arrival (== launch count; speed is fixed)


# --- World view ----------------------------------------------------------

class SweepCache:
    """Per-turn cache for planet/comet swept segments.

    Each planet's swept segment for tick j depends only on (planet, omega, obs_step, j).
    Caching avoids recomputing the same positions across many find_shot calls.
    """
    __slots__ = ("omega", "obs_step", "_cache")

    def __init__(self, omega: float, obs_step: int):
        self.omega = omega
        self.obs_step = obs_step
        self._cache: dict[tuple[int, int], tuple] = {}

    def sweep(self, p: "PlanetView", tick_start: int):
        key = (p.id, tick_start)
        s = self._cache.get(key)
        if s is not None:
            return s
        if p.is_comet:
            s = sim.comet_sweep_segment(p.comet_group, tick_start, p.comet_index)
        else:
            s = sim.planet_sweep_segment(p.x, p.y, p.radius, self.omega, tick_start, self.obs_step)
        self._cache[key] = s
        return s

    def pos(self, p: "PlanetView", tick: int):
        if p.is_comet:
            return sim.comet_pos_at(p.comet_group, tick, p.comet_index)
        return sim.predict_planet_pos(p.x, p.y, p.radius, self.omega, tick, self.obs_step)


def parse_world(obs) -> dict:
    """Parse a kaggle_environments obs into a flat dict for the planner.

    Handles dict-style or namespace-style obs (engine uses both via `get`).
    """
    def g(key, default=None):
        if isinstance(obs, dict):
            return obs.get(key, default)
        return getattr(obs, key, default)

    planets_raw = g("planets", []) or []
    comet_ids = set(g("comet_planet_ids", []) or [])
    comets_raw = g("comets", []) or []

    # Map (group_idx, comet_idx_in_group) for each comet pid
    comet_lookup = {}
    for grp in comets_raw:
        for i, pid in enumerate(grp["planet_ids"]):
            comet_lookup[pid] = (grp, i)

    planets = []
    for p in planets_raw:
        is_comet = p[0] in comet_ids
        grp, idx = comet_lookup.get(p[0], (None, None))
        planets.append(PlanetView(
            id=int(p[0]),
            owner=int(p[1]),
            x=float(p[2]),
            y=float(p[3]),
            radius=float(p[4]),
            ships=int(p[5]),
            production=int(p[6]),
            is_comet=is_comet,
            comet_group=grp,
            comet_index=idx,
        ))

    return {
        "player": int(g("player", 0)),
        "step": int(g("step", 0)),
        "omega": float(g("angular_velocity", 0.0)),
        "planets": planets,
        "planet_by_id": {p.id: p for p in planets},
        "fleets": [list(f) for f in (g("fleets", []) or [])],
        "comets": comets_raw,
        "remaining_overage": float(g("remainingOverageTime", 0.0)),
    }


# --- Target prediction --------------------------------------------------

def target_pos_at(p: PlanetView, omega: float, ticks_ahead: int, obs_step: int) -> tuple[float, float] | None:
    """Predict target position `ticks_ahead` ticks from now (0=current obs)."""
    if p.is_comet:
        return sim.comet_pos_at(p.comet_group, ticks_ahead, p.comet_index)
    return sim.predict_planet_pos(p.x, p.y, p.radius, omega, ticks_ahead, obs_step)


def target_sweep(
    p: PlanetView, omega: float, tick_start: int, obs_step: int
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Swept segment for target during tick `tick_start` (0=launch tick)."""
    if p.is_comet:
        return sim.comet_sweep_segment(p.comet_group, tick_start, p.comet_index)
    return sim.planet_sweep_segment(p.x, p.y, p.radius, omega, tick_start, obs_step)


# --- Intercept core ------------------------------------------------------

MAX_ETA = 60
THETA_ITERS = 4
KCONV_ITERS = 6


def _converge_theta_and_spawn(
    src: PlanetView, tgt_arrival: tuple[float, float]
) -> tuple[float, tuple[float, float]]:
    """Fixed-point on theta accounting for spawn offset depending on theta."""
    theta = math.atan2(tgt_arrival[1] - src.y, tgt_arrival[0] - src.x)
    for _ in range(THETA_ITERS):
        spawn = sim.spawn_position(src.x, src.y, src.radius, theta)
        theta = math.atan2(tgt_arrival[1] - spawn[1], tgt_arrival[0] - spawn[0])
    spawn = sim.spawn_position(src.x, src.y, src.radius, theta)
    return theta, spawn


def find_shot(
    src: PlanetView,
    tgt: PlanetView,
    ship_count: int,
    world: dict,
    sun_margin: float = sim.SUN_SAFETY_MARGIN,
    cache: SweepCache | None = None,
) -> Shot | None:
    """Find the smallest-ETA clean intercept for `ship_count` ships from src to tgt."""
    if src.id == tgt.id or ship_count <= 0:
        return None
    if ship_count > src.ships:
        return None
    omega = world["omega"]
    obs_step = world["step"]
    if cache is None:
        cache = SweepCache(omega, obs_step)
    v = sim.fleet_speed(ship_count)

    # First-cut ETA from straight-line distance.
    dx = tgt.x - src.x
    dy = tgt.y - src.y
    D0 = math.hypot(dx, dy)
    k = max(1, int(round((D0 - src.radius - sim.SPAWN_OFFSET) / max(v, 1e-6))))
    k = max(k, 1)

    # Converge k by repeatedly predicting target position and updating ETA.
    last_k = -1
    for _ in range(KCONV_ITERS):
        if k > MAX_ETA or k < 1:
            return None
        tgt_arrival = cache.pos(tgt, k)
        if tgt_arrival is None:
            return None  # comet expired
        theta, spawn = _converge_theta_and_spawn(src, tgt_arrival)
        D = math.hypot(tgt_arrival[0] - spawn[0], tgt_arrival[1] - spawn[1])
        k_new = max(1, int(round(D / v)))
        if k_new == k:
            break
        if k_new == last_k:
            # 2-cycle oscillation: try both.
            break
        last_k = k
        k = k_new

    # Try candidate ETAs around the converged value.
    best = None
    best_k = None
    for k_try in sorted({k, k - 1, k + 1, k - 2, k + 2}):
        if k_try < 1 or k_try > MAX_ETA:
            continue
        shot = _verify_intercept(src, tgt, ship_count, k_try, world, sun_margin, cache)
        if shot is not None:
            if best is None or k_try < best_k:
                best = shot
                best_k = k_try
    return best


def find_shot_for_arrival(
    src: PlanetView,
    tgt: PlanetView,
    target_step: int,
    world: dict,
    sun_margin: float = sim.SUN_SAFETY_MARGIN,
    cache: SweepCache | None = None,
) -> Shot | None:
    """Inverse intercept: find the ship count that makes the fleet arrive at `target_step`.

    Used for multi-source wave coordination — pick S so that v(S) puts the fleet
    on the target on the exact engine step requested.
    """
    if src.id == tgt.id:
        return None
    omega = world["omega"]
    obs_step = world["step"]
    if cache is None:
        cache = SweepCache(omega, obs_step)

    k = target_step - world["step"]
    if k < 1 or k > MAX_ETA:
        return None

    # Predict target position at arrival; iterate theta + spawn to fixed point.
    tgt_arrival = cache.pos(tgt, k)
    if tgt_arrival is None:
        return None  # comet expired by then
    theta, spawn = _converge_theta_and_spawn(src, tgt_arrival)
    D = math.hypot(tgt_arrival[0] - spawn[0], tgt_arrival[1] - spawn[1])
    v_req = D / k

    # Find ship count that achieves v_req (or the closest feasible v).
    if v_req > sim.MAX_SHIP_SPEED + 1e-9:
        return None  # unreachable by target_step even at max speed
    S = sim.ships_for_speed(max(1.0, v_req))
    if S > src.ships:
        return None  # would exceed available garrison

    # _verify_intercept also tries the exact k. Try ±1 to absorb integer rounding.
    for k_try in (k, k - 1, k + 1):
        if k_try < 1 or k_try > MAX_ETA:
            continue
        shot = _verify_intercept(src, tgt, S, k_try, world, sun_margin, cache)
        if shot is not None and shot.eta == k:
            return shot
    # Fallback: accept ±1 step from requested if exact-k doesn't verify.
    for k_try in (k, k - 1, k + 1):
        if k_try < 1 or k_try > MAX_ETA:
            continue
        shot = _verify_intercept(src, tgt, S, k_try, world, sun_margin, cache)
        if shot is not None:
            return shot
    return None


def _verify_intercept(
    src: PlanetView,
    tgt: PlanetView,
    ship_count: int,
    k: int,
    world: dict,
    sun_margin: float,
    cache: SweepCache,
) -> Shot | None:
    """Given a target ETA k, compute theta and verify the shot lands and is safe."""
    omega = world["omega"]
    obs_step = world["step"]
    v = sim.fleet_speed(ship_count)
    tgt_arrival = cache.pos(tgt, k)
    if tgt_arrival is None:
        return None
    theta, spawn = _converge_theta_and_spawn(src, tgt_arrival)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dir_step = (v * cos_t, v * sin_t)

    # Fleet swept segment on tick j (0-indexed; j=0 is the launch tick).
    # Tick j: from (spawn + j*v*dir) to (spawn + (j+1)*v*dir).
    # Planet/comet swept on tick j: from predict(j) to predict(j+1).

    # Check the arrival tick (j = k-1) actually hits the target.
    arr_j = k - 1
    A = (spawn[0] + arr_j * dir_step[0], spawn[1] + arr_j * dir_step[1])
    B = (A[0] + dir_step[0], A[1] + dir_step[1])
    tgt_sweep = cache.sweep(tgt, arr_j)
    if tgt_sweep is None:
        return None
    if not sim.swept_pair_hit(A, B, tgt_sweep[0], tgt_sweep[1], tgt.radius):
        return None

    # On every tick j = 0..k-1, verify safety:
    # - No premature collision with ANY non-target planet/comet
    # - No sun crossing
    # - No OOB
    # Critical: on the arrival tick, the target HAS to be the first hit in engine
    # iteration order. To be safe, we reject if any other planet would hit on tick k-1.
    all_planets = world["planets"]
    for j in range(k):
        A_j = (spawn[0] + j * dir_step[0], spawn[1] + j * dir_step[1])
        B_j = (A_j[0] + dir_step[0], A_j[1] + dir_step[1])

        if sim.segment_oob(A_j, B_j):
            return None
        if sim.segment_crosses_sun(A_j, B_j, margin=sun_margin):
            return None

        # Quick orbital-radius prune (rotation-invariant): a planet's distance
        # from sun is constant. A fleet point at distance d_from_sun can only
        # collide with a planet at orbital radius r if |d - r| <= radius+v/2.
        mid_x = (A_j[0] + B_j[0]) * 0.5
        mid_y = (A_j[1] + B_j[1]) * 0.5
        mid_from_sun = math.hypot(mid_x - sim.CENTER, mid_y - sim.CENTER)
        half_len = v * 0.5
        for p in all_planets:
            if p.id == tgt.id and j == arr_j:
                continue  # intended hit
            if not p.is_comet:
                p_orb = math.hypot(p.x - sim.CENTER, p.y - sim.CENTER)
                # Need |mid_from_sun - p_orb| <= p.radius + half_len + small slack
                slack = p.radius + half_len + 1.0
                if abs(mid_from_sun - p_orb) > slack:
                    continue
            sweep = cache.sweep(p, j)
            if sweep is None:
                continue
            if p.is_comet and sweep[0] == sweep[1] and sweep[0][0] < 0:
                continue  # comet not yet placed
            if sim.swept_pair_hit(A_j, B_j, sweep[0], sweep[1], p.radius):
                return None

    return Shot(
        src_id=src.id,
        tgt_id=tgt.id,
        eta=k,
        ship_count=ship_count,
        angle=theta,
        arrival_xy=B,  # fleet endpoint on arrival tick
        arrival_ships=ship_count,
    )


# --- Smart ship-count axis ------------------------------------------------

def candidate_ship_counts(
    src: PlanetView,
    tgt: PlanetView,
    predicted_garrison_at_arrival: int,
    defense_reserve: int = 0,
) -> list[int]:
    """Smart-discretized set per the plan: small set of meaningful operating points."""
    avail = src.ships - defense_reserve
    if avail < 1:
        return []
    cap = max(1, predicted_garrison_at_arrival + 1)  # strict > to flip ownership
    raw = {
        cap,                     # exact minimum-to-capture
        cap + 5,                 # buffer for production growth
        max(1, avail // 2),      # half-stack
        avail,                   # all-in
    }
    # Cap to available; filter to [1, avail].
    return sorted({min(max(1, s), avail) for s in raw})


def build_shot_menu(
    world: dict,
    defense_reserve_fn=None,
    deadline: float | None = None,
    max_targets_per_src: int = 6,
) -> dict[tuple[int, int], list[Shot]]:
    """Enumerate every clean intercept from each of OUR planets to every target.

    Returns dict keyed by (src_id, tgt_id) -> list of Shots (sorted by ETA).

    Limits the search to the closest `max_targets_per_src` targets per source
    to bound per-turn compute. Distant targets are unlikely to be valuable
    decisions; we can revisit them next turn after the closer plays settle.
    """
    import time as _t
    me = world["player"]
    my_planets = [p for p in world["planets"] if p.owner == me and p.ships > 0]
    all_planets = world["planets"]
    omega = world["omega"]
    obs_step = world["step"]

    cache = SweepCache(omega, obs_step)
    menu: dict[tuple[int, int], list[Shot]] = {}

    for src in my_planets:
        if deadline is not None and _t.perf_counter() >= deadline:
            break
        reserve = 0 if defense_reserve_fn is None else defense_reserve_fn(src)
        # Sort targets by raw distance — try nearest first.
        targets = sorted(
            (p for p in all_planets if p.id != src.id),
            key=lambda p: (p.x - src.x) ** 2 + (p.y - src.y) ** 2,
        )[:max_targets_per_src]

        for tgt in targets:
            if deadline is not None and _t.perf_counter() >= deadline:
                break
            rough_eta = max(1, int(math.hypot(tgt.x - src.x, tgt.y - src.y) / sim.MAX_SHIP_SPEED))
            if tgt.owner == -1:
                pred_garrison = tgt.ships
            else:
                pred_garrison = tgt.ships + tgt.production * rough_eta
            counts = candidate_ship_counts(src, tgt, pred_garrison, defense_reserve=reserve)
            shots = []
            seen_keys = set()
            for s in counts:
                shot = find_shot(src, tgt, s, world, cache=cache)
                if shot is None:
                    continue
                key = (shot.eta, shot.ship_count)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                shots.append(shot)
            if shots:
                shots.sort(key=lambda s: (s.eta, s.ship_count))
                menu[(src.id, tgt.id)] = shots

    return menu
