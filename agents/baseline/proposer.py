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

from __future__ import annotations

import math
import os

from lib.aim import aim_orbiting
from lib.fleet import speed as fleet_speed
from lib.orbit import is_orbiting, predict_relative
from lib.scoring import pv_horizon

NUM_TARGETS_PER_SOURCE = 8
MIN_FLEET_SIZE = 2
SIM_SETTLE_TURNS = 2
MIN_HORIZON = 25
MAX_HORIZON = 40
WAIT_EXTRA_SURPLUS = (0, 5, 12)
CHEAP_REJECT_THRESHOLD = -10.0
EPISODE_STEPS = 500
GAMMA = 0.99


def aim_and_eta(src, tgt, ships: int, omega: float, wait_N: int = 0):
    """Return (aim_angle_radians, ceil_eta_turns) for one (src, tgt, ships).

    For orbiting targets, jointly solves aim + eta via lib.aim.aim_orbiting.
    For wait_N>0 candidates, pre-rotates BOTH src and tgt by omega*wait_N
    so aim is computed at the geometry that will hold at fire time (co-
    rotating planets preserve relative geometry).
    """
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
        enemy_inflight = sum(
            ships
            for (eta_arr, owner, ships) in model.ledger.get(int(tgt.id), [])
            if owner != me and eta_arr <= enemy_eta + 1
        )
        enemy_potential = 0.0
        if enemy_inflight <= 0:
            best_enemy_ships = 0.0
            for p in world.planets_by_id.values():
                if int(p.owner) < 0 or int(p.owner) == me:
                    continue
                if int(p.ships) > best_enemy_ships:
                    best_enemy_ships = float(p.ships)
            enemy_potential = best_enemy_ships
        enemy_strength = max(enemy_inflight, enemy_potential)
        my_garrison = float(tgt.ships) + float(tgt.production) * enemy_eta
        shortfall = enemy_strength - my_garrison + 1
        return max(0, int(math.ceil(shortfall)))

    initial = max(MIN_FLEET_SIZE, int(tgt.ships) + 1)
    _angle, eta = aim_and_eta(src, tgt, initial, omega)
    pred = float(model.ships_at(int(tgt.id), eta) or 0.0)
    return max(MIN_FLEET_SIZE, int(math.ceil(pred)) + 1)


def max_safe_launch_now(src, world, model, me: int, *,
                        horizon: int = 20, safety: int = 2) -> int:
    """Max ships we can launch from `src` this turn without `src` falling
    within `horizon` turns.

    Walks every enemy arrival at `src` with eta <= horizon. For each eta_e,
    the binding constraint is

        L <= src.ships + prod*eta_e + my_reinforce_by(eta_e)
                 - enemy_arrived_by(eta_e) - safety.

    Returns max(0, min(src.ships, min over eta_e)) — i.e. the tightest
    floor across all incoming waves. If no enemy is inbound within
    `horizon`, the full garrison is launchable.

    NOTE on calibration (2026-05-20 Phase 2 A/B): in 2P self-play vs the
    pre-Phase-1 baseline, applying this floor cumulatively (with Phase 1
    on) hit 3/8 wins. Symmetric matchups penalise the more defensive
    side — both bleed similarly without this floor, but with it the
    floored side captures less ground while still losing to the opp
    attack that arrives anyway. Live calibration via the ladder is the
    truer test (asymmetric opponents); see BASELINE_GARRISON_FLOOR env
    var gating in enumerate_ship_counts.
    """
    pid = int(src.id)
    arrivals = model.ledger.get(pid, [])
    if not arrivals:
        return int(src.ships)

    enemy = sorted(
        ((eta, ships) for (eta, owner, ships) in arrivals
         if owner != me and ships > 0 and eta <= horizon),
        key=lambda x: x[0],
    )
    if not enemy:
        return int(src.ships)

    mine = sorted(
        ((eta, ships) for (eta, owner, ships) in arrivals
         if owner == me and eta <= horizon),
        key=lambda x: x[0],
    )

    prod = int(src.production)
    src_ships = int(src.ships)
    enemy_cum = 0
    binding = src_ships  # start permissive
    for eta_e, ships_e in enemy:
        enemy_cum += int(ships_e)
        my_reinforce = sum(int(s) for (e, s) in mine if e <= eta_e)
        avail = src_ships + prod * int(eta_e) + my_reinforce - enemy_cum - safety
        if avail < binding:
            binding = avail
    return max(0, min(src_ships, binding))


def enumerate_ship_counts(src, tgt, model, omega: float, me: int, world) -> list[int]:
    """Fire-now ship-count set: capture-size, 2x capture-size, full budget.

    Gated by BASELINE_GARRISON_FLOOR=1: when set, each candidate size is
    clamped by `max_safe_launch_now(src)` so a src under imminent threat
    cannot bleed below its garrison floor. Off by default after a 2P
    self-play A/B (n=8) showed Phase-1+floor cumulative at 3/8 wins
    against the pre-Phase-1 baseline — the floor is too conservative in
    symmetric play. Kept for live-ladder A/B as a separate knob.
    """
    cap = capture_size(src, tgt, model, omega, me, world)
    budget = int(src.ships)
    if os.environ.get("BASELINE_GARRISON_FLOOR", "0").strip() == "1":
        budget = min(budget, max_safe_launch_now(src, world, model, me))
    if cap == 0:
        return []  # reinforce-targets with no threat
    sizes = set()
    if MIN_FLEET_SIZE <= cap <= budget:
        sizes.add(cap)
    if 2 * cap <= budget:
        sizes.add(2 * cap)
    if budget >= MIN_FLEET_SIZE and budget > cap:
        sizes.add(budget)
    return sorted(sizes)


def wait_then_fire_variants(src, tgt, model, omega: float, me: int, world=None):
    """Multi-wait-grid candidates for one (src, tgt). Returns list of
    (ships, wait_N, angle, eta). Generates one variant per extra_surplus
    in WAIT_EXTRA_SURPLUS, deduped by (wait_N, ships).

    Wait variants are dropped when `model.owner_at(src.id, wait_N)` is no
    longer me — src is predicted to fall before the launch time, so the
    plan is infeasible. (R40: model-correctness, not a brittle cap.)
    """
    if int(tgt.owner) == me:
        return []
    prod = int(src.production)
    if prod <= 0:
        return []

    initial = max(MIN_FLEET_SIZE, int(tgt.ships) + 1)
    _a0, eta0 = aim_and_eta(src, tgt, initial, omega)
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

        # Drop wait variants whose src is predicted to fall before fire time.
        pred_owner = model.owner_at(int(src.id), wait_N)
        if pred_owner is not None and pred_owner != me:
            continue

        angle, eta = aim_and_eta(src, tgt, target_fleet, omega, wait_N=wait_N)
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
                angle, eta = aim_and_eta(src, tgt, ships, omega)
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
                src, tgt, model, omega, me,
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

    best_per_band: dict[tuple[int, int, int], tuple] = {}
    for entry in prerank:
        cheap, src, tgt, _ships, _angle, _eta, _horizon, w = entry
        key = (int(src.id), int(tgt.id), wait_band(int(w)))
        prev = best_per_band.get(key)
        if prev is None or cheap > prev[0]:
            best_per_band[key] = entry

    deduped = list(best_per_band.values())
    deduped.sort(key=lambda e: -e[0])
    return deduped
