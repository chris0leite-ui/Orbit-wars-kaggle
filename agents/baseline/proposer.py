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

from lib.aim import aim_comet, aim_orbiting
from lib.fleet import speed as fleet_speed
from lib.orbit import is_orbiting, predict_relative
from lib.scoring import pv_horizon
from lib.trajectory import predict_fleet_fate
from lib.world_model import _comet_paths_by_id, _position_at, comet_remaining_lifetime

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
    tgt=None,
) -> bool:
    """Source-drain protection. Returns True if `src` survives sending
    `ships` ships at step `wait_N`.

    Base predicate (pre-2026-05-27): True when no inbound threat, OR
    threat is potential-only (let the rollout handle), OR the residue
    + production accrual covers the in-flight threat force + 1.

    Larger→smaller hardening (2026-05-27, PI direction). When
    `tgt is not None` AND `src.production > tgt.production` — i.e.
    we'd be sending a fleet from a higher-prod planet to a lower-prod
    one — three extra clauses fire to prevent losing the more
    valuable source:

      Clause A (stockpile floor). Regardless of threat, residue must
        cover `STOCKPILE_PROD_MULT × src.production` ships. Catches
        the "drain home to capture a tiny neutral" case where there's
        no inbound threat in the ledger today.

      Clause B (stricter margin). Under threat, require
        `SAFETY_MARGIN_DRAIN × threat_force` residue, not just
        `threat_force + 1`.

      Clause C (potential-launch coverage). When the ledger has no
        in-flight threat but `time_to_enemy_threat` (which considers
        potential launches) returned a non-None eta, fold in the
        biggest single opp's ship count as a conservative
        potential-threat estimate.

    Anchored on the PI directive 2026-05-27: "we never lose a large
    important planet because we send ships somewhere when there is
    an opponent that could capture us after we lower our garrison."
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
            # Original behaviour: let the rollout score potential-only.
            return True
        potential = _largest_opp_potential_force(src, world, me)
        if potential <= 0:
            return True
        # 0.5× because the biggest single opp launching everything is a
        # worst-case bound; the realistic threat is roughly half of that.
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


def _target_holdable_after_capture(
    src, tgt, ships: int, wait_N: int, eta: int, world, model, me: int,
) -> bool:
    """Tier 2 hold-feasibility filter.

    "Will the target stay ours after the cheapest opp recapture?"
    Iterates EVERY opp with `ships >= MIN_COUNTER_SHIPS` (not just the
    nearest), computes each opp's true recapture cost via a small
    fixed-point on `(opp_needed, opp_speed, t_op)`, and rejects the
    candidate if ANY opp can both afford and overwhelm our
    garrison-at-recapture.

    Three correctness fixes vs the pre-2026-05-27 nearest-only
    version:

    (1) ALL opps iterated. Previously picked `nearest_opp` purely by
        distance — a stronger but slightly-further opp was silently
        ignored.

    (2) `opp_speed` computed from `opp_needed` (the ships opp would
        actually launch), not from `opp.ships` (full garrison).
        `fleet_speed` is monotone increasing in ships; using the
        garrison overestimated launch speed → underestimated `t_op`
        → underestimated `garrison_at_recapture` → mis-modeled the
        opp.

    (3) B7-style fixed-point on `t_op` for orbiting targets. The
        target rotates during opp's transit; the rendezvous point
        shifts. Mirrors `lib/world_model.time_to_enemy_threat`
        :464-474.
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
        # Fixed-point on opp_needed (matches `_target_holdable_after_capture`):
        # speed estimated from the ships opp actually launches, not from
        # the full garrison. Otherwise `fleet_speed(opp.ships)` (monotone
        # increasing) inflated launch speed and underestimated cost.
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
