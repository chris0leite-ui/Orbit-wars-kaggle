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
from lib.mirror import detect_num_players
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
# Fix (strategic defense, 2026-05-21): for high-prod own planets,
# floor the reinforce target to a preemptive stockpile so the LP can
# see strategic defense before shortfall materialises. Without this,
# `capture_size` returns 0 whenever current garrison covers current
# threat — blinding the LP to opp's build-up on a planet we won't
# defend until it's too late. Prefixed STRATEGIC_* to avoid bundler
# namespace collision (cf. OPENING_* rename, 2026-05-21 AM).
STRATEGIC_DEFENSE_PROD = 4    # production threshold for "strategic"
STRATEGIC_STOCKPILE_TICKS = 5 # buffer = N ticks × planet's production

# Fix (confidence-aware capture buffer, 2026-05-21): the offensive
# `capture_size` is spec-minimum — `ceil(predicted_defender)+1`. Median
# solo capture margin in 2P is +5 ships (per launch_introspect); 14%
# of solo attempts bounce because opp launched reinforcements during
# our flight that the model couldn't see at depart time. Buffer ε
# scales with flight time × target production (the two best proxies
# for opp reinforcement rate over the flight window). Capped by
# CONFIDENCE_BUFFER_MAX; discounted in multi-player formats where
# over-commitment to a single capture drains defense against the
# other opps (per AGGR sibling-branch 4P caveat).
CONFIDENCE_BUFFER_BASE = 1
CONFIDENCE_ETA_SCALE = 0.5
CONFIDENCE_PROD_NORM = 3.0
CONFIDENCE_BUFFER_MAX = 12
CONFIDENCE_4P_DISCOUNT = 0.4

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
from lib.world_model import WAVE_LOOKAHEAD  # noqa: E402


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
    _angle, eta = aim_and_eta(src, tgt, initial, omega)
    pred = float(model.ships_at(int(tgt.id), eta) or 0.0)
    return max(MIN_FLEET_SIZE, int(math.ceil(pred)) + 1)


def confidence_buffered_size(src, tgt, model, omega: float, me: int, world) -> int:
    """Offensive capture size with confidence buffer ε added.

    The model predicts opp strength at arrival using current trajectory;
    new opp launches during our flight are unmodeled. ε scales with
    flight time × target production (the two best proxies for opp
    reinforcement rate). Returns the buffered size, capped by
    CONFIDENCE_BUFFER_MAX; discounted in 4P+ where over-commitment
    drains defense.

    Returns 0 for own targets (buffer is offense-only — own-target
    reinforce is sized by capture_size's shortfall logic).
    """
    if int(tgt.owner) == me:
        return 0
    initial = max(MIN_FLEET_SIZE, int(tgt.ships) + 1)
    _angle, eta = aim_and_eta(src, tgt, initial, omega)
    pred = float(model.ships_at(int(tgt.id), eta) or 0.0)
    prod_factor = float(tgt.production) / CONFIDENCE_PROD_NORM
    epsilon = (
        CONFIDENCE_BUFFER_BASE
        + CONFIDENCE_ETA_SCALE * float(eta) * prod_factor
    )
    epsilon = min(epsilon, float(CONFIDENCE_BUFFER_MAX))
    n_players = detect_num_players(list(world.planets_by_id.values()))
    if n_players >= 3:
        epsilon *= CONFIDENCE_4P_DISCOUNT
    return max(MIN_FLEET_SIZE, int(math.ceil(pred + epsilon)) + 1)


def enumerate_ship_counts(src, tgt, model, omega: float, me: int, world,
                          peer_sources_in_reach: int = 1) -> list[int]:
    """Fire-now ship-count set: capture-size, 2x capture-size, full budget.

    Emits `budget` as a candidate when `budget >= MIN_FLEET_SIZE`,
    GATED on bundle feasibility when `budget < cap`: a partial-budget
    candidate fires solo and bounces unless a peer source can also
    contribute. Gate (2026-05-21) added after the 2026-05-21 bundle
    fix surfaced 6-12 ship partials that fired solo (introspect on
    2P seeds 384458460/42/7: 6 confirmed regressions, ~10 ships each).
    `peer_sources_in_reach` defaults to 1 so direct callers (tests,
    introspect tools) preserve the prior emit-always behavior; `propose`
    computes the actual count and passes it through.
    """
    cap = capture_size(src, tgt, model, omega, me, world)
    budget = int(src.ships)
    if cap == 0:
        return []  # reinforce-targets with no threat
    sizes = set()
    if MIN_FLEET_SIZE <= cap <= budget:
        sizes.add(cap)
    # Confidence-buffered variant: insurance against opp reinforcement
    # during flight. Emitted alongside spec-min so the LP can choose
    # between ship-efficient (cap) and robust (cap + ε) sizing per
    # outcome value. The bundler must inline `lib/mirror` for this to
    # work — see scripts/bundle_agent.py:DEFAULT_LIB_ORDER (added 2026-
    # 05-21 after a NameError-in-bundle bug silently swallowed by
    # kaggle_environments debug=False made it look like a strategic
    # regression).
    buffered = confidence_buffered_size(src, tgt, model, omega, me, world)
    if MIN_FLEET_SIZE <= buffered <= budget and buffered != cap:
        sizes.add(buffered)
    if 2 * cap <= budget:
        sizes.add(2 * cap)
    if budget >= MIN_FLEET_SIZE:
        # Gate (CAPTURE only): a sub-cap partial-budget candidate against
        # an opp/neutral target fires solo and bounces unless a peer can
        # also fire (bundle). Own-target reinforces are NOT gated — the
        # ships arrive at our planet and add to garrison; partial defense
        # is strictly better than no defense.
        is_reinforce = int(tgt.owner) == me
        if is_reinforce or budget >= cap or peer_sources_in_reach >= 1:
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

    nearest_opp = None
    nearest_opp_dist = float("inf")
    for opp in world.planets_by_id.values():
        if int(opp.owner) == me or int(opp.owner) == -1:
            continue
        if int(opp.id) == int(tgt.id):
            continue
        if int(opp.ships) < MIN_COUNTER_SHIPS:
            continue
        d = math.hypot(
            float(opp.x) - float(tgt.x), float(opp.y) - float(tgt.y),
        )
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
        d = math.hypot(
            float(ally.x) - float(tgt.x), float(ally.y) - float(tgt.y),
        )
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


def propose(my_planets, target_pool, world, model, me: int,
            omega: float, baseline_len: int):
    """Build the pre-rank list of candidates, then dedup by
    (src_id, tgt_id, wait_band) keeping the top cheap-Δ per bucket.

    Returns: list of tuples
        (cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N)
    sorted by cheap_delta descending.
    """
    # Pre-compute per-peer reach sets: a peer's reach is its
    # NUM_TARGETS_PER_SOURCE nearest targets. Used to gate sub-cap
    # partial-budget candidates (Fix 1, 2026-05-21) — only emit when
    # a bundle is physically possible.
    peer_reach = {
        int(p.id): {int(t.id) for t in nearest_k(target_pool, p, NUM_TARGETS_PER_SOURCE)}
        for p in my_planets
        if int(p.ships) >= MIN_FLEET_SIZE
    }

    prerank = []
    for src in my_planets:
        if int(src.ships) < MIN_FLEET_SIZE:
            continue
        for tgt in nearest_k(target_pool, src, NUM_TARGETS_PER_SOURCE):
            if int(tgt.id) == int(src.id):
                continue

            peer_count = sum(
                1 for pid, reach_set in peer_reach.items()
                if pid != int(src.id) and int(tgt.id) in reach_set
            )

            for ships in enumerate_ship_counts(
                src, tgt, model, omega, me, world,
                peer_sources_in_reach=peer_count,
            ):
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
            # Wait-then-fire: predict fate against fire-time geometry
            # (planet positions advanced by wait_N orbital ticks).
            fate = predict_fleet_fate(
                src, tgt, float(angle), int(ships), world, wait_N=int(w),
            )
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

    deduped.sort(key=lambda e: -e[0])
    return deduped
