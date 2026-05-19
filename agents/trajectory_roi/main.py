"""trajectory_roi v3 — blast-radius-anchored joint forward projector.

Three layers per turn, all closed-form / analytical:

  Layer 1 — Blast-radius enumeration: only consider targets reachable
    within K_HORIZON turns from each of our planets. Per source, keep
    the top TOP_TARGETS_PER_SOURCE by production / (reach + 1).

  Layer 2 — Central-control value: each captured planet's value
    includes a centrality bonus (inner rotating planets are worth
    extra production-equivalent per turn beyond their raw production).
    Encodes the strategic objective "maximize blast radius."

  Layer 3 — Forward-projection joint solve: replace v2's 2-opt with
    incremental joint optimization. For each candidate, project the
    K=50-turn outcome with `lite_greedy_policy` for BOTH sides after
    turn 0 — that captures opp's reactive counter-launches naturally.
    Each candidate is scored by its MARGINAL contribution to the
    projected outcome of the current plan, not by standalone ROI.

Compute budget per turn (mid-game): ~600ms target. K=50 projection
under lite_greedy is ~12 ms/plan per the benchmark.

v2 stays at commit e006b91 as the depth-2 reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from lib.aim import aim_orbiting, flight_distance
from lib.fleet import speed as fleet_speed
from lib import fast_sim
from lib.opp_model import lite_greedy_policy
from lib.trajectory_layer import World


# ---- tuneables ------------------------------------------------------------

K_HORIZON = 50                    # forward-projection depth
TOP_TARGETS_PER_SOURCE = 5        # Layer-1 prune cap
MAX_CANDIDATES_TOTAL = 30         # final candidate cap
MAX_ITERATIONS = 2                # incremental joint solve iterations
MIN_LAUNCH_SHIPS = 5              # env minimum
SAFETY_MARGIN = 1                 # ships above net defenders
MAX_ETA = 50                      # don't consider launches that take > MAX_ETA turns

# Layer-2 centrality thresholds
CENTER_X = 50.0
CENTER_Y = 50.0
INNER_RADIUS = 25.0               # inner zone (high blast-radius value)
MID_RADIUS = 35.0                 # mid zone
CENTRAL_STATIC_DIST = 25.0        # static planet centrality cutoff
CENTRALITY_BONUS_TURNS = 30       # future turns of credit for centrality


# ---- core types -----------------------------------------------------------


@dataclass(frozen=True)
class Allocation:
    src_id: int
    ships: int
    aim_angle: float


@dataclass(frozen=True)
class Candidate:
    flavor: Literal["capture", "defense"]
    target_id: int
    arrival_turn: int
    allocations: tuple[Allocation, ...]
    raw_value: float
    total_ships: int

    @property
    def roi(self) -> float:
        return self.raw_value / (self.total_ships + 1.0)


# ---- centrality (Layer 2) -------------------------------------------------


def _build_centrality_cache(world: World) -> dict[int, float]:
    """Per-planet centrality score for blast-radius weighting."""
    cache: dict[int, float] = {}
    for p in world.planets:
        if getattr(p, "is_rotating", False):
            if p.orbital_radius < INNER_RADIUS:
                cache[p.id] = 1.0
            elif p.orbital_radius < MID_RADIUS:
                cache[p.id] = 0.5
            else:
                cache[p.id] = 0.0
        else:
            d = math.hypot(p.current_x - CENTER_X, p.current_y - CENTER_Y)
            if d < CENTRAL_STATIC_DIST:
                cache[p.id] = 0.5
            else:
                cache[p.id] = 0.0
    return cache


def _centrality_from_raw_planet(raw_p) -> float:
    """Used by terminal-value scoring inside projection (raw obs tuple)."""
    x, y = raw_p[2], raw_p[3]
    d = math.hypot(x - CENTER_X, y - CENTER_Y)
    if d < INNER_RADIUS:
        return 1.0
    if d < MID_RADIUS:
        return 0.5
    return 0.0


# ---- aim / ETA primitives -------------------------------------------------


def _aim_and_eta(src, target, ships: int, omega: float):
    if ships < MIN_LAUNCH_SHIPS:
        return None
    v = fleet_speed(ships)
    if v <= 0:
        return None
    src_xy = (src.current_x, src.current_y)
    tgt_xy = (target.current_x, target.current_y)
    flight = flight_distance(src_xy, src.radius, tgt_xy, target.radius)
    if flight <= 0:
        return None
    if not getattr(target, "is_rotating", False) or omega == 0.0:
        eta = int(math.ceil(flight / v))
        if eta <= 0 or eta > MAX_ETA:
            return None
        ang = math.atan2(target.current_y - src.current_y,
                         target.current_x - src.current_x)
        return (ang, eta)
    target_tuple = (
        target.id, target.owner,
        target.current_x, target.current_y,
        target.radius, target.ships, target.production,
    )
    res = aim_orbiting(src_xy, src.radius, target_tuple, target.radius,
                       ships, omega)
    if res is None:
        return None
    angle, _arrival_xy, eta_f = res
    eta = int(math.ceil(eta_f))
    if eta <= 0 or eta > MAX_ETA:
        return None
    return (angle, eta)


# ---- defender math (Layer 1 building block) ------------------------------


def _fleet_eta_to_planet(fleet, target_planet) -> int | None:
    fx, fy = fleet.current_x, fleet.current_y
    spd = fleet.speed
    if spd <= 0:
        return None
    tx, ty, tr = target_planet.current_x, target_planet.current_y, target_planet.radius
    dx, dy = tx - fx, ty - fy
    dir_x, dir_y = math.cos(fleet.angle), math.sin(fleet.angle)
    proj = dx * dir_x + dy * dir_y
    if proj < 0:
        return None
    perp_sq = dx * dx + dy * dy - proj * proj
    r_sq = tr * tr
    if perp_sq >= r_sq:
        return None
    hit_d = max(0.0, proj - math.sqrt(max(0.0, r_sq - perp_sq)))
    return int(math.ceil(hit_d / spd))


def _net_defenders(world: World, target, arrival_turn: int, my_id: int,
                   target_is_ours: bool) -> float:
    """Closed-form defenders at arrival, accounting for in-flight fleets."""
    base = (float(target.ships) if target.owner == -1
            else float(target.ships) + float(target.production) * float(arrival_turn))
    ours_in = 0.0
    theirs_in = 0.0
    for fleet in world.fleets:
        eta = _fleet_eta_to_planet(fleet, target)
        if eta is None or eta > arrival_turn:
            continue
        if fleet.owner == my_id:
            ours_in += float(fleet.ships)
        else:
            theirs_in += float(fleet.ships)
    if target_is_ours:
        return float(theirs_in) - (base + float(ours_in))
    return base + float(theirs_in) - float(ours_in)


# ---- Layer 1: single-source capture solver -------------------------------


def _solve_single_source(src, target, world: World, my_id: int,
                         centrality_cache: dict[int, float],
                         target_is_ours: bool) -> Candidate | None:
    omega = world.omega
    src_budget = int(src.ships)
    if src_budget < MIN_LAUNCH_SHIPS:
        return None

    K = max(MIN_LAUNCH_SHIPS, int(target.ships) + SAFETY_MARGIN)
    for _ in range(4):
        ae = _aim_and_eta(src, target, K, omega)
        if ae is None:
            return None
        angle, eta = ae
        net = _net_defenders(world, target, eta, my_id, target_is_ours)
        if not target_is_ours and net < 0:
            return None
        if target_is_ours and net <= 0:
            return None
        K_needed = max(MIN_LAUNCH_SHIPS, int(math.ceil(net)) + SAFETY_MARGIN)
        if K_needed <= K:
            K = K_needed
            break
        K = K_needed

    if K > src_budget:
        return None
    ae = _aim_and_eta(src, target, K, omega)
    if ae is None:
        return None
    angle, eta = ae
    net_final = _net_defenders(world, target, eta, my_id, target_is_ours)
    if K < max(MIN_LAUNCH_SHIPS, net_final + SAFETY_MARGIN):
        return None

    held = max(0, K_HORIZON - eta)
    centrality_v = centrality_cache.get(target.id, 0.0)
    base_value = float(target.production) * float(held)
    central_value = centrality_v * float(target.production) * CENTRALITY_BONUS_TURNS
    raw_value = base_value + central_value
    if target_is_ours:
        raw_value += float(target.ships)  # avoided ship loss
    if raw_value <= 0:
        return None

    return Candidate(
        flavor=("defense" if target_is_ours else "capture"),
        target_id=target.id,
        arrival_turn=eta,
        allocations=(Allocation(src.id, K, angle),),
        raw_value=raw_value,
        total_ships=K,
    )


def _solve_multi_source(target, world: World, my_id: int,
                        centrality_cache: dict[int, float],
                        target_is_ours: bool) -> Candidate | None:
    """Multi-source bundle for high-priority targets (centrality > 0)."""
    my_planets = [p for p in world.planets if p.owner == my_id and p.id != target.id]
    if len(my_planets) < 2:
        return None
    omega = world.omega
    feasible: list[tuple[int, int, float, int]] = []
    for src in my_planets:
        if src.ships < MIN_LAUNCH_SHIPS:
            continue
        ae = _aim_and_eta(src, target, max(MIN_LAUNCH_SHIPS, int(src.ships)), omega)
        if ae is None:
            continue
        angle, eta = ae
        if eta > MAX_ETA:
            continue
        feasible.append((eta, int(src.ships), angle, src.id))
    if len(feasible) < 2:
        return None
    feasible.sort()  # by eta

    best: Candidate | None = None
    for k_sources in range(2, min(len(feasible), 4) + 1):
        chosen = feasible[:k_sources]
        arrival_turn = chosen[-1][0]
        net = _net_defenders(world, target, arrival_turn, my_id, target_is_ours)
        if not target_is_ours and net < 0:
            return None
        if target_is_ours and net <= 0:
            return None
        need = max(MIN_LAUNCH_SHIPS, int(math.ceil(net)) + SAFETY_MARGIN)
        remaining = need
        allocations: list[Allocation] = []
        for (eta, budget, ang, src_id) in chosen:
            if remaining <= 0:
                break
            take = min(remaining, budget)
            if take < MIN_LAUNCH_SHIPS:
                continue
            allocations.append(Allocation(src_id, take, ang))
            remaining -= take
        if remaining > 0 or len(allocations) < 2:
            continue
        total = sum(a.ships for a in allocations)
        held = max(0, K_HORIZON - arrival_turn)
        centrality_v = centrality_cache.get(target.id, 0.0)
        raw_value = (float(target.production) * float(held)
                     + centrality_v * float(target.production) * CENTRALITY_BONUS_TURNS)
        if target_is_ours:
            raw_value += float(target.ships)
        if raw_value <= 0:
            continue
        c = Candidate(
            flavor="defense" if target_is_ours else "capture",
            target_id=target.id,
            arrival_turn=arrival_turn,
            allocations=tuple(allocations),
            raw_value=raw_value,
            total_ships=total,
        )
        if best is None or c.roi > best.roi:
            best = c
    return best


# ---- Layer 1: enumeration ------------------------------------------------


def _threatened_planets(world: World, my_id: int):
    """Our planets with projected opp arrivals exceeding garrison + ours-in."""
    out = []
    for p in world.planets:
        if p.owner != my_id:
            continue
        ours_in = 0.0
        theirs_in = 0.0
        for fleet in world.fleets:
            eta = _fleet_eta_to_planet(fleet, p)
            if eta is None or eta > MAX_ETA:
                continue
            if fleet.owner == my_id:
                ours_in += float(fleet.ships)
            else:
                theirs_in += float(fleet.ships)
        if theirs_in <= 0:
            continue
        if theirs_in > float(p.ships) + float(ours_in):
            out.append(p)
    return out


def enumerate_candidates(world: World, my_id: int,
                         centrality_cache: dict[int, float]) -> list[Candidate]:
    """Layer 1: build all candidates within blast radius, top-N per source."""
    per_target_top: dict[int, list[Candidate]] = {}
    my_planets = [p for p in world.planets if p.owner == my_id]
    targets_capture = [p for p in world.planets if p.owner != my_id]

    def _add(c: Candidate):
        bucket = per_target_top.setdefault(c.target_id, [])
        bucket.append(c)
        bucket.sort(key=lambda x: -x.roi)
        del bucket[3:]  # keep top 3 per target (different sources / single vs multi)

    # Per-source: rank reachable targets by closed-form ROI, take top-N.
    for src in my_planets:
        if src.ships < MIN_LAUNCH_SHIPS:
            continue
        scored: list[Candidate] = []
        for tgt in targets_capture:
            c = _solve_single_source(src, tgt, world, my_id,
                                     centrality_cache, target_is_ours=False)
            if c is not None:
                scored.append(c)
        scored.sort(key=lambda x: -x.roi)
        for c in scored[:TOP_TARGETS_PER_SOURCE]:
            _add(c)

    # Multi-source bundles for central targets (centrality > 0).
    for tgt in targets_capture:
        if centrality_cache.get(tgt.id, 0.0) <= 0.0:
            continue
        mc = _solve_multi_source(tgt, world, my_id, centrality_cache,
                                 target_is_ours=False)
        if mc is not None:
            _add(mc)

    # Defense candidates.
    threatened = _threatened_planets(world, my_id)
    for tgt in threatened:
        for src in my_planets:
            if src.id == tgt.id:
                continue
            c = _solve_single_source(src, tgt, world, my_id,
                                     centrality_cache, target_is_ours=True)
            if c is not None:
                _add(c)
        mc = _solve_multi_source(tgt, world, my_id, centrality_cache,
                                 target_is_ours=True)
        if mc is not None:
            _add(mc)

    flat: list[Candidate] = []
    for bucket in per_target_top.values():
        flat.extend(bucket)
    flat.sort(key=lambda c: -c.roi)
    return flat[:MAX_CANDIDATES_TOTAL]


# ---- Layer 3: forward projection ----------------------------------------


def _obs_from_snap(snap, seat: int) -> dict:
    s_obs = snap.state[seat].observation
    return {
        "player": seat,
        "step": int(getattr(s_obs, "step", 0)),
        "planets": [list(p) for p in s_obs.planets],
        "fleets": [list(f) for f in (s_obs.fleets or [])],
        "comets": list(getattr(s_obs, "comets", [])),
        "comet_planet_ids": list(getattr(s_obs, "comet_planet_ids", [])),
        "angular_velocity": float(getattr(s_obs, "angular_velocity", 0.0)),
        "initial_planets": [list(p) for p in getattr(s_obs, "initial_planets", s_obs.planets)],
    }


def _terminal_value(snap, my_id: int) -> float:
    """Augmented terminal score: ship diff + centrality bonus."""
    base = fast_sim.delta_us_minus_them(snap, my_id)
    obs0 = snap.state[0].observation
    central_bonus = 0.0
    for p in obs0.planets:
        if int(p[1]) != my_id:
            continue
        c = _centrality_from_raw_planet(p)
        if c <= 0:
            continue
        central_bonus += c * float(p[6]) * CENTRALITY_BONUS_TURNS
    return base + central_bonus


def project(initial_obs: dict, my_id: int, opp_id: int,
            my_turn0_emits: list, K: int = K_HORIZON) -> float:
    """Forward-project K turns. Turn 0: our planned emits + opp's
    `lite_greedy`. Turns 1..K-1: both sides run `lite_greedy`.
    Returns augmented terminal value."""
    snap = fast_sim.from_obs(initial_obs, configuration=None)
    actions = [None, None]
    actions[my_id] = my_turn0_emits
    actions[opp_id] = lite_greedy_policy(_obs_from_snap(snap, opp_id))
    snap = fast_sim.step(snap, actions)
    for _ in range(K - 1):
        if snap.fake_env.done:
            break
        actions[my_id] = lite_greedy_policy(_obs_from_snap(snap, my_id))
        actions[opp_id] = lite_greedy_policy(_obs_from_snap(snap, opp_id))
        snap = fast_sim.step(snap, actions)
    return _terminal_value(snap, my_id)


def _emit_for_candidate(c: Candidate) -> list[list]:
    return [[a.src_id, a.aim_angle, a.ships] for a in c.allocations]


def _fits(c: Candidate, remaining: dict[int, int]) -> bool:
    for a in c.allocations:
        if remaining.get(a.src_id, 0) < a.ships:
            return False
    return True


def joint_solve_forward(candidates: list[Candidate], initial_obs: dict,
                        my_id: int, opp_id: int, my_planets) -> list[list]:
    """Incremental joint via K-turn forward projection. Returns the
    list of turn-0 emit actions to execute."""
    remaining_budget = {p.id: int(p.ships) for p in my_planets}
    plan_emits: list[list] = []
    chosen: list[Candidate] = []
    used_targets: set[int] = set()

    current_value = project(initial_obs, my_id, opp_id, plan_emits)

    for _ in range(MAX_ITERATIONS):
        best_marginal = 0.0
        best_candidate: Candidate | None = None
        best_with_value: float | None = None
        for c in candidates:
            if c.target_id in used_targets:
                continue
            if not _fits(c, remaining_budget):
                continue
            test_emits = plan_emits + _emit_for_candidate(c)
            with_value = project(initial_obs, my_id, opp_id, test_emits)
            marginal = with_value - current_value
            if marginal > best_marginal:
                best_marginal = marginal
                best_candidate = c
                best_with_value = with_value
        if best_candidate is None or best_with_value is None:
            break
        chosen.append(best_candidate)
        used_targets.add(best_candidate.target_id)
        for a in best_candidate.allocations:
            remaining_budget[a.src_id] -= a.ships
            plan_emits.append([a.src_id, a.aim_angle, a.ships])
        current_value = best_with_value
    return plan_emits


# ---- helpers (must be defined BEFORE `agent` so that
#       kaggle_environments.agent.get_last_callable picks `agent`
#       as the entry point, not a helper) -------------------------------


def _obs_from_snap_like(obs) -> dict:
    """Best-effort dict obs from struct-form obs."""
    return {
        "player": int(getattr(obs, "player", 0)),
        "step": int(getattr(obs, "step", 0)),
        "planets": [list(p) for p in obs.planets],
        "fleets": [list(f) for f in (obs.fleets or [])],
        "comets": list(getattr(obs, "comets", [])),
        "comet_planet_ids": list(getattr(obs, "comet_planet_ids", [])),
        "angular_velocity": float(getattr(obs, "angular_velocity", 0.0)),
        "initial_planets": [list(p) for p in getattr(obs, "initial_planets", obs.planets)],
    }


# ---- main entry — MUST BE LAST callable in this module ------------------


def agent(obs, configuration=None):
    world = World.from_obs(obs, configuration)
    my_id = world.my_id
    other_owners = [p.owner for p in world.planets
                    if p.owner not in (-1, my_id)]
    if not other_owners:
        return []
    opp_id = max(set(other_owners), key=other_owners.count)
    if my_id == opp_id:
        return []

    my_planets = [p for p in world.planets if p.owner == my_id]
    if not my_planets:
        return []

    centrality_cache = _build_centrality_cache(world)
    candidates = enumerate_candidates(world, my_id, centrality_cache)
    if not candidates:
        return []

    initial_obs = obs if isinstance(obs, dict) else _obs_from_snap_like(obs)
    return joint_solve_forward(candidates, initial_obs, my_id, opp_id, my_planets)
