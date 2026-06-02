"""Marco-v3-3 opening planner port — used as a Tier-3 opponent model.

Background: ~7-9 of the top-30 public agents share the marco-v3-3
lineage's EAM opening planner (deterministic depth-5 / beam-width-8
search over capture-time minimising plans, gated to step < 50 with
≤ 6 owned planets). When we play one of those forks, predicting their
exact first 5 launches lets the chooser run a one-ply adversarial
re-rank of its own top candidates against the predicted opponent reply
— see plan at audit/2026-06-02-marco-lineage-reference/PLAN.md.

This module ports the planner faithfully from
`audit/2026-06-02-marco-lineage-reference/kernels/marco-dg-v3-3.py`
lines 2199-2484. Constants and beam structure match the source verbatim;
the orbital primitives are substituted from `lib/` (mathematically
equivalent — orbital radius and current-angle conventions are the same).

Public API:

    predict_marco_plan(obs, opp_seat, time_budget_ms=30) -> list[Commit] | None

`obs` may be any of: kaggle obs-dict, `Struct`, or a Snapshot per-seat
state (.observation). `opp_seat` is the seat whose moves we're predicting
(so when modelling opp from our own chooser, this is the OPP id, not us).
Returns `None` when the planner's own gate falls through — opening window
closed, too many planets, 4P game, deadline expired, or no feasible plan.

Default OFF in the chooser: the wiring in `lib/opp_model.py` Tier 3 +
`agents/baseline/chooser_trajectory.py` is env-var gated. With both
gates off the live champion bundle is byte-identical to today's.
"""

from __future__ import annotations

import math
import time
from collections import namedtuple
from dataclasses import dataclass
from typing import Any

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from lib.geometry import CENTER, ROTATION_RADIUS_LIMIT
from lib.fleet import speed as fleet_speed
from lib.orbit import predict_relative
from lib.world_model import build_arrival_ledger, simulate_planet_timeline


# Marco's EAM constants — source kernel lines 2202-2208.
PLAN_DEPTH: int = 5
PLAN_BEAM_WIDTH: int = 8
PLAN_MAX_EXTRA_WAIT: int = 15
EAM_OPENING_LIMIT: int = 50
EAM_MAX_MY_PLANETS: int = 6
EAM_DEFENSE_LOOKAHEAD: int = 15

EPISODE_STEPS_TOTAL: int = 500
ARRIVAL_HORIZON: int = 250  # matches lib.world_model.DEFAULT_HORIZON


Commit = namedtuple("Commit", ["src_id", "tgt_id", "t_launch", "fleet", "eta"])


@dataclass
class _MarcoView:
    """View of the world from `player`'s seat — the shape marco's planner
    reads (subset of fields actually used by `_plan_best_launch`,
    `_plan_evaluate`, `_plan_beam_search`, `eam_choose_moves`).

    Built once per `predict_marco_plan` call from an obs + opp_seat.
    """
    player: int
    step: int
    remaining_steps: int
    omega: float  # angular velocity; matches marco's `ang_vel`
    is_four_player: bool
    planets: list  # list[Planet]
    planet_by_id: dict[int, Any]
    my_planets: list  # list[Planet]
    arrivals_by_planet: dict[int, list[tuple[int, int, int]]]
    fall_turn_map: dict[int, int | None]


def _is_static(planet) -> bool:
    """Match marco's `is_static_planet`: orbital_radius + planet.radius
    >= ROTATION_LIMIT.

    Our `lib.orbit.is_orbiting` returns the inverse; this helper exists so
    the port reads the same way as the source kernel.
    """
    dx = float(planet.x) - CENTER
    dy = float(planet.y) - CENTER
    orb_r = math.hypot(dx, dy)
    return (orb_r + float(planet.radius)) >= ROTATION_RADIUS_LIMIT


def _predict_position(planet, omega: float, turns: float) -> tuple[float, float]:
    """Substitute for marco's `predict_planet_position`. Bit-equivalent
    for orbital planets because orbital radius is conserved during play
    (so `r` from initial == `r` from current); static planets short-circuit
    to current (x, y).
    """
    if _is_static(planet):
        return float(planet.x), float(planet.y)
    p_tuple = [planet.id, planet.owner, planet.x, planet.y, planet.radius,
               planet.ships, planet.production]
    return predict_relative(p_tuple, omega, turns)


def _marco_dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


# ---------------------------------------------------------------------------
# Beam planner — port of marco-dg-v3-3.py lines 2210-2437
# ---------------------------------------------------------------------------


def _plan_best_launch(src_id, src_planet, ref_ships, ref_prod, ref_time,
                      target, view: _MarcoView, R):
    """Port of marco's `_plan_best_launch` (line 2210). Find optimal launch
    time minimising capture_time for this (source, target) pair.

    Source planet has `ref_ships` at `ref_time` and produces `ref_prod`/turn.
    Returns dict(t_launch, fleet, eta, cap_t) or None if infeasible within R.
    """
    G = int(target.ships)
    if ref_prod <= 0 and ref_ships < G + 1:
        return None
    if ref_ships >= G + 1:
        t_min = ref_time
    else:
        need = G + 1 - ref_ships
        t_min = ref_time + int(math.ceil(need / max(1, ref_prod)))
    best = None
    src_static = _is_static(src_planet)
    tgt_static = _is_static(target)

    for extra in range(0, PLAN_MAX_EXTRA_WAIT + 1):
        t = t_min + extra
        if t >= R:
            break
        fleet = ref_ships + ref_prod * (t - ref_time)
        if fleet < G + 1:
            continue
        speed = fleet_speed(fleet)
        if src_static or t == 0:
            sx, sy = float(src_planet.x), float(src_planet.y)
        else:
            sx, sy = _predict_position(src_planet, view.omega, t)
        if tgt_static:
            eta = _marco_dist(sx, sy, float(target.x), float(target.y)) / speed
        else:
            eta = _marco_dist(sx, sy, float(target.x), float(target.y)) / speed
            for _ in range(8):
                px, py = _predict_position(target, view.omega, t + eta)
                new_eta = _marco_dist(sx, sy, px, py) / speed
                if abs(new_eta - eta) < 0.05:
                    eta = new_eta
                    break
                eta = new_eta
        cap_t = t + eta
        if cap_t >= R:
            continue
        if best is None or cap_t < best["cap_t"]:
            best = {"t_launch": t, "fleet": int(fleet), "eta": eta, "cap_t": cap_t}
        if extra > 5 and cap_t > best["cap_t"] + 1.0:
            break
    return best


def _enemy_earliest_capture(target, view: _MarcoView):
    """Smallest time (in turns from now) for any enemy of `view.player` to
    capture target. Port of marco's `_enemy_earliest_capture` (line 2263)."""
    best = float("inf")
    G = int(target.ships)
    for src in view.planets:
        if src.owner == view.player or src.owner == -1:
            continue
        S = int(src.ships)
        p_rate = int(src.production)
        for W in range(0, 40):
            fleet = S + p_rate * W
            if fleet < G + 1:
                continue
            speed = fleet_speed(fleet)
            if _is_static(target):
                eta = _marco_dist(float(src.x), float(src.y),
                            float(target.x), float(target.y)) / speed
            else:
                tx, ty = float(target.x), float(target.y)
                eta = _marco_dist(float(src.x), float(src.y), tx, ty) / speed
                for _ in range(2):
                    px, py = _predict_position(target, view.omega, W + eta)
                    eta = _marco_dist(float(src.x), float(src.y), px, py) / speed
            t = W + eta
            if t < best:
                best = t
            if W > 5 and t > best:
                break
    return best


def _plan_evaluate(plan, view: _MarcoView, enemy_earliest=None):
    """Simulate a plan of (src_id, tgt_id) commitments.

    Returns dict(V, moves) or None if plan is infeasible.
    Port of marco-dg-v3-3.py line 2296.
    """
    R = view.remaining_steps
    sources = {}
    for planet in view.my_planets:
        sources[planet.id] = (int(planet.ships), int(planet.production), 0)

    # Pre-populate planets being captured by in-flight friendly fleets
    # (marco line 2310-2331). The same logic transfers from the seat:
    # arrivals_by_planet is global, but we only credit friendlies (owner
    # == view.player) and only on planets the seat doesn't already own.
    in_flight_captures: set = set()
    for pid, arrivals in view.arrivals_by_planet.items():
        planet = view.planet_by_id.get(pid)
        if planet is None or planet.owner == view.player:
            continue
        friendly = sorted(
            [(eta, ships) for eta, owner, ships in arrivals if owner == view.player],
            key=lambda x: x[0],
        )
        if not friendly:
            continue
        garrison = int(planet.ships)
        cumulative = 0
        for eta, ships in friendly:
            cumulative += ships
            if cumulative > garrison:
                residual = cumulative - garrison
                sources[pid] = (residual, int(planet.production), eta)
                in_flight_captures.add(pid)
                break

    V = 0.0
    moves = []
    for src_id, tgt_id in plan:
        if src_id not in sources:
            return None
        if tgt_id == src_id:
            return None
        ref_ships, ref_prod, ref_t = sources[src_id]
        src_planet = view.planet_by_id[src_id]
        target = view.planet_by_id[tgt_id]
        already_captured_in_plan = {t for _, t in plan[:len(moves)]}
        if target.owner == view.player and tgt_id not in already_captured_in_plan:
            return None
        if tgt_id in in_flight_captures and tgt_id not in already_captured_in_plan:
            return None
        launch = _plan_best_launch(src_id, src_planet, ref_ships, ref_prod,
                                   ref_t, target, view, R)
        if launch is None:
            return None

        if enemy_earliest is not None and tgt_id in enemy_earliest:
            if enemy_earliest[tgt_id] < launch["cap_t"] - 0.5:
                return None

        V += int(target.production) * (R - launch["cap_t"])
        moves.append({
            "src_id": src_id,
            "tgt_id": tgt_id,
            "t_launch": launch["t_launch"],
            "fleet": launch["fleet"],
            "eta": launch["eta"],
            "cap_t": launch["cap_t"],
            "production": int(target.production),
        })
        sources[src_id] = (0, ref_prod, launch["t_launch"])
        residual = max(0, launch["fleet"] - int(target.ships))
        sources[tgt_id] = (residual, int(target.production), launch["cap_t"])
    return {"V": V, "moves": moves}


def _plan_beam_search(view: _MarcoView, depth: int, beam_width: int,
                      deadline: float | None) -> dict | None:
    """Beam search over plans. Port of marco line 2375."""
    player = view.player
    all_targets = [p for p in view.planets if p.owner != player]
    if not all_targets:
        return None

    enemy_earliest = {t.id: _enemy_earliest_capture(t, view) for t in all_targets}

    initial_sources = {p.id for p in view.my_planets}
    plans = [{"plan": [], "V": 0.0, "moves": []}]

    for _ in range(depth):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        new_plans = []
        for entry in plans:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            prev_plan = entry["plan"]
            used_tgts = {tid for _, tid in prev_plan}
            avail_sources = set(initial_sources) | used_tgts
            for src_id in avail_sources:
                for tgt in all_targets:
                    if tgt.id in used_tgts:
                        continue
                    if tgt.id == src_id:
                        continue
                    new_plan = prev_plan + [(src_id, tgt.id)]
                    res = _plan_evaluate(new_plan, view, enemy_earliest=enemy_earliest)
                    if res is None:
                        continue
                    new_plans.append({
                        "plan": new_plan,
                        "V": res["V"],
                        "moves": res["moves"],
                    })
        if not new_plans:
            break
        seen = {}
        for p in new_plans:
            key = tuple(p["plan"])
            if key not in seen or p["V"] > seen[key]["V"]:
                seen[key] = p
        candidates = sorted(seen.values(), key=lambda x: -x["V"])
        plans = candidates[:beam_width]

    if not plans:
        return None
    return max(plans, key=lambda x: x["V"])


# ---------------------------------------------------------------------------
# View construction
# ---------------------------------------------------------------------------


def _obs_field(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _build_view(obs: Any, opp_seat: int) -> _MarcoView | None:
    """Construct a `_MarcoView` from any observation, viewed from
    `opp_seat`. Returns None if the obs is empty (no planets).
    """
    raw_planets = _obs_field(obs, "planets", []) or []
    raw_fleets = _obs_field(obs, "fleets", []) or []
    if not raw_planets:
        return None
    omega = float(_obs_field(obs, "angular_velocity", 0.0) or 0.0)
    step = int(_obs_field(obs, "step", 0) or 0)

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    planet_by_id = {int(p.id): p for p in planets}
    my_planets = [p for p in planets if int(p.owner) == int(opp_seat)]

    arrivals_by_planet = build_arrival_ledger(fleets, planets, omega,
                                              horizon=ARRIVAL_HORIZON)

    # fall_turn_map: walk each planet's timeline (predict_garrison_at-style)
    # and record the first turn its owner flips off opp_seat. None if it
    # holds for the full ARRIVAL_HORIZON. Used by eam_choose_moves to skip
    # the EAM gate when any of opp's planets is about to fall.
    fall_turn_map: dict[int, int | None] = {}
    for p in planets:
        if int(p.owner) != int(opp_seat):
            fall_turn_map[int(p.id)] = None
            continue
        tl = simulate_planet_timeline(p, arrivals_by_planet[int(p.id)],
                                      horizon=ARRIVAL_HORIZON)
        owners = tl["owner_at"]
        fall = None
        for t in range(1, int(tl["horizon"]) + 1):
            if owners.get(t, int(opp_seat)) != int(opp_seat):
                fall = t
                break
        fall_turn_map[int(p.id)] = fall

    # num_players: count distinct seat ids that own any planet or fleet
    # (matches marco's count_players). Excludes neutral (-1).
    seats: set[int] = set()
    for p in planets:
        if int(p.owner) >= 0:
            seats.add(int(p.owner))
    for f in fleets:
        if int(f.owner) >= 0:
            seats.add(int(f.owner))
    num_players = len(seats)

    return _MarcoView(
        player=int(opp_seat),
        step=step,
        remaining_steps=max(1, EPISODE_STEPS_TOTAL - step),
        omega=omega,
        is_four_player=(num_players >= 4),
        planets=planets,
        planet_by_id=planet_by_id,
        my_planets=my_planets,
        arrivals_by_planet=arrivals_by_planet,
        fall_turn_map=fall_turn_map,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_marco_plan(obs: Any, opp_seat: int,
                       time_budget_ms: float = 30.0,
                       ) -> list[Commit] | None:
    """Predict the first ≤5 captures a marco-v3-3 fork would commit to
    from this observation, viewed from `opp_seat`.

    Returns a list of `Commit` records (one per planned capture), or
    `None` when the EAM gate falls through:
      - `obs` empty;
      - step >= EAM_OPENING_LIMIT (50);
      - opp owns > EAM_MAX_MY_PLANETS (6) planets;
      - 4P game (marco's EAM is 2P-only);
      - one of opp's planets is about to fall within EAM_DEFENSE_LOOKAHEAD;
      - `time_budget_ms` expired;
      - no feasible plan found.

    The caller treats `None` as "no marco-specific opp prediction; fall
    through to Tier 0/1 mirror".

    `time_budget_ms` defaults to 30 ms. Empirically the planner takes a
    few ms in the opening (small planet sets); the deadline guards against
    pathological inputs and the late-window 5-depth × 8-width blow-up.
    """
    view = _build_view(obs, opp_seat)
    if view is None:
        return None
    if not view.my_planets:
        return None
    if view.step >= EAM_OPENING_LIMIT:
        return None
    if view.is_four_player:
        return None
    if len(view.my_planets) > EAM_MAX_MY_PLANETS:
        return None
    for p in view.my_planets:
        fall = view.fall_turn_map.get(int(p.id))
        if fall is not None and fall < EAM_DEFENSE_LOOKAHEAD:
            return None

    # Marco's adaptive depth (line 2456-2465).
    n = len(view.my_planets)
    if n == 1:
        depth = 5
    elif n == 2:
        depth = 4
    elif n <= 4:
        depth = 3
    else:
        depth = 2

    deadline = time.perf_counter() + time_budget_ms / 1000.0
    best = _plan_beam_search(view, depth=depth, beam_width=PLAN_BEAM_WIDTH,
                             deadline=deadline)
    if best is None or not best.get("moves"):
        return None

    out: list[Commit] = []
    for commit in best["moves"]:
        out.append(Commit(
            src_id=int(commit["src_id"]),
            tgt_id=int(commit["tgt_id"]),
            t_launch=int(commit["t_launch"]),
            fleet=int(commit["fleet"]),
            eta=float(commit["eta"]),
        ))
    return out
