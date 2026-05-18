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
from lib.trajectory import predict_fleet_fate
from lib.world_model import comet_remaining_lifetime

NUM_TARGETS_PER_SOURCE = 8
MIN_FLEET_SIZE = 2
SIM_SETTLE_TURNS = 2
MIN_HORIZON = 25
MAX_HORIZON = 40
WAIT_EXTRA_SURPLUS = (0, 5, 12)  # legacy forward grid (kept for rollback)
CHEAP_REJECT_THRESHOLD = -10.0
EPISODE_STEPS = 500
GAMMA = 0.99

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

# Bug #12 fix (2026-05-18): widen the in-flight-enemy summation window
# so a staggered multi-wave attack (e.g. f1 at eta=2 + f2 at eta=4) is
# accounted for as a single coordinated threat. Pre-fix the window was
# `enemy_eta + 1` of the EARLIEST inbound, which silently excluded
# later waves and zeroed the shortfall. Anchored on the asdf-game
# (76947663) step 37 trace. The principled v2 of this fix is a full
# timeline simulation to find the max shortfall over time; this
# constant is the cheap version.
WAVE_LOOKAHEAD = 12


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
        return max(0, int(math.ceil(shortfall)))

    initial = max(MIN_FLEET_SIZE, int(tgt.ships) + 1)
    _angle, eta = aim_and_eta(src, tgt, initial, omega)
    pred = float(model.ships_at(int(tgt.id), eta) or 0.0)
    return max(MIN_FLEET_SIZE, int(math.ceil(pred)) + 1)


def enumerate_ship_counts(src, tgt, model, omega: float, me: int, world) -> list[int]:
    """Fire-now ship-count set: capture-size, 2x capture-size, full budget."""
    cap = capture_size(src, tgt, model, omega, me, world)
    budget = int(src.ships)
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


def wait_then_fire_variants_forward(src, tgt, model, omega: float, me: int):
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


def min_wait_affordable(src, tgt, model, omega: float, me: int) -> int | None:
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
    _a0, eta0 = aim_and_eta(src, tgt, initial, omega, wait_N=0)
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
        _angle, eta = aim_and_eta(src, tgt, target_fleet, omega, wait_N=wait_N)
        pred_at_arrival = float(model.ships_at(int(tgt.id), wait_N + eta) or 0.0)
        cap_final = max(MIN_FLEET_SIZE, int(math.ceil(pred_at_arrival)) + 1)
        if cap_final <= budget and wait_N + eta + SIM_SETTLE_TURNS <= MAX_HORIZON:
            return wait_N
    return None  # hopeless within MAX_HORIZON


def wait_then_fire_variants(src, tgt, model, omega: float, me: int):
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
        return wait_then_fire_variants_forward(src, tgt, model, omega, me)
    if int(tgt.owner) == me:
        return []
    min_w = min_wait_affordable(src, tgt, model, omega, me)
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
        angle, eta = aim_and_eta(src, tgt, target_fleet, omega, wait_N=wait_N)
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

    deduped.sort(key=lambda e: -e[0])
    return deduped
