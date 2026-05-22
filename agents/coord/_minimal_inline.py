"""_minimal_inline — bundle-safe copy of helpers from agents/minimal/main.py.

This file exists ONLY to make `agents/coord/` self-contained for the
`scripts/bundle_agent.py` bundler. The bundler inlines submodules
within a single agent directory but doesn't follow cross-agent imports
(`from agents.minimal.main import ...`). By keeping a local copy here,
`agents/coord/main.py` can import `from agents.coord._minimal_inline import X`
which the bundler natively handles.

Contents (identical bodies to agents/minimal/main.py — refresh if minimal
evolves):
- Constants from minimal (baked from orbitfix env config).
- Helper functions: favor_hybrid, aim_and_eta, nearest_k,
  enumerate_ship_counts, wait_then_fire_variants, cheap_marginal_value,
  _source_survives_launch, _target_holdable_after_capture,
  _target_cost_parity_ok, opp_actions_for_snap, affordable_validate_cap,
  build_trajectory_baseline, score_candidate_v4, score_candidate_v4_joint,
  _as_dict, _num_seats, and all their transitive dependencies.

Excluded (orchestration that coord replaces with its own pipeline):
- agent(), propose(), choose_trajectory(),
  emit_threat_reinforcements(), _emit_anticipated_reinforces().

Note on multi-line imports: the bundler's per-line regex strips line 1
but leaves continuation lines as orphans (IndentationError at runtime).
The `from lib.world_model import (...)` was flattened to one-name-per-line
imports for bundle safety.
"""
from __future__ import annotations

import math
import time

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from lib.aim import aim_comet, aim_orbiting
from lib.fast_sim import clone as fs_clone
from lib.fast_sim import from_obs as fs_from_obs
from lib.fast_sim import step as fs_step
from lib.fleet import speed as fleet_speed
from lib.intent import World
from lib.opp_model import lite_greedy_policy
from lib.orbit import is_orbiting, predict_relative
from lib.scoring import pv_horizon
from lib.trajectory import predict_fleet_fate
from lib.value_heads import composite_capture_value
from lib.world_model import WAVE_LOOKAHEAD
from lib.world_model import WorldModel
from lib.world_model import _comet_paths_by_id
from lib.world_model import _position_at
from lib.world_model import comet_remaining_lifetime
from lib.world_model import predict_garrison_at


# ---------------------------------------------------------------------------
# Constants (baked from orbitfix env config; no runtime overrides).
# ---------------------------------------------------------------------------

GAMMA = 0.99
EPISODE_STEPS = 500
WALLCLOCK_BUDGET_MS = 600.0

# Proposer
NUM_TARGETS_PER_SOURCE = 8
MIN_FLEET_SIZE = 2
SIM_SETTLE_TURNS = 2
MIN_HORIZON = 25
MAX_HORIZON = 40
CHEAP_REJECT_THRESHOLD = -10.0
WAIT_BUFFER_OFFSET = 3
STRATEGIC_DEFENSE_PROD = 4
STRATEGIC_STOCKPILE_TICKS = 5

# Reactor-aware (cost-parity filter + dual-mover candidates)
COST_PARITY_MARGIN = 0.7
MIN_REACTOR_SHIPS = 8
MAX_REACTOR_CANDIDATES_PER_TURN = 12
REACTOR_TOP_K_SOURCES_PER_TARGET = 3

# Chooser
SETTLE_TURNS = 3
N_VALIDATE = 200
PER_CANDIDATE_SAFETY = 1.5
RESERVED_OVERHEAD_MS = 50.0
WASTE_WEIGHT = 0.5
CAPTURE_REWARD_WEIGHT = 0.05

# Joint scoring (BASELINE_JOINT_AGGR=1 — 2-source pairing in both 2P and 4P,
# stacking allowed via used_tgts lift)
JOINT_TOP_K_PER_TARGET = 5
JOINT_MAX_PAIRS = 60

# Reinforce post-pass
REINFORCE_MAX_LAUNCHES = 3
REINFORCE_MIN_PROD = 2
ANTICIPATE_MIN_PROD = 3
ANTICIPATE_MARGIN = 1.3

# Value head leaf
ELIMINATION_BONUS = 55.0
WEAK_ENEMY_THRESHOLD = 110.0
WEAKEST_ENEMY_MULT_4P = 1.5
ELIMINATION_GATE_RATIO = 0.9
STRENGTH_PROD_WEIGHT = 15.0


# ---------------------------------------------------------------------------
# Value head — `favor_hybrid` (composite in 2P, A2-favor in 4P).
# ---------------------------------------------------------------------------

def _read(obs, attr, default):
    if hasattr(obs, attr):
        return getattr(obs, attr)
    return obs.get(attr, default) if isinstance(obs, dict) else default


def favor_4p(obs, me: int, num_seats: int) -> float:
    """A2 4P leaf: F1 (ship balance) + F2 (PV-weighted prod balance) +
    elimination bonus when we can finish the weakest opp.

    F1 = my_ships - opp_ships_weighted
    F2 = (my_prod - opp_prod_weighted) * pv_horizon
    weakest opp contribution gets a 1.5x multiplier (4P only).
    """
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
    opps = sorted(o for o in (set(ships_by_owner) | set(prod_by_owner))
                  if o != me and o >= 0)

    elim_bonus = 0.0
    if num_seats <= 2 or len(opps) < 2:
        opp_ships = max((ships_by_owner.get(o, 0.0) for o in opps), default=0.0)
        opp_prod = max((prod_by_owner.get(o, 0.0) for o in opps), default=0.0)
    else:
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

    pv = pv_horizon(step, 0, gamma=GAMMA, t_total=EPISODE_STEPS)
    return (my_ships - opp_ships) + (my_prod - opp_prod) * pv + elim_bonus


def favor_hybrid(obs, me: int, num_seats: int, gamma: float = GAMMA) -> float:
    """2P uses the composite waste-aware head; 4P uses A2-favor."""
    if num_seats <= 2:
        return composite_capture_value(obs, me)
    return favor_4p(obs, me, num_seats)


# ---------------------------------------------------------------------------
# Aim + ETA — comet path-lead, orbital pre-rotation, straight-line fallback.
# ---------------------------------------------------------------------------

def aim_and_eta(src, tgt, ships: int, omega: float, wait_N: int = 0, world=None):
    """Return (aim_angle, ceil_eta_turns) for one (src, tgt, ships).

    Comet targets use path-indexed lead (advances comet path_index by
    wait_N). Orbiting non-comet targets use joint aim+eta solver with
    BOTH src and tgt pre-rotated by omega*wait_N. Static targets fall
    back to straight-line atan2.
    """
    if world is not None:
        comet_entry = None
        if int(tgt.id) in getattr(world, "comet_ids", set()):
            comet_entry = _comet_paths_by_id(world).get(int(tgt.id))
        if comet_entry is not None:
            path, path_index = comet_entry
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


# ---------------------------------------------------------------------------
# Proposer — candidate enumeration, cheap-rank, dedup, filters.
# ---------------------------------------------------------------------------

def capture_size(src, tgt, model, omega: float, me: int, world) -> int:
    """WorldModel-aware minimum size to take or hold tgt from src."""
    if int(tgt.owner) == me:
        # Reinforce: cover predicted shortfall vs known + speculative threats.
        enemy_eta = model.time_to_enemy_threat(int(tgt.id), me, world)
        if enemy_eta is None:
            return 0
        enemy_inflight = sum(
            ships
            for (eta_arr, owner, ships) in model.ledger.get(int(tgt.id), [])
            if owner != me and eta_arr <= enemy_eta + WAVE_LOOKAHEAD
        )
        enemy_potential = 0.0
        if enemy_inflight <= 0:
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
        # Strategic stockpile: floor for high-prod planets even when
        # current shortfall ≤ 0 (otherwise LP can't see preemptive defense).
        if int(tgt.production) >= STRATEGIC_DEFENSE_PROD:
            base = max(base, STRATEGIC_STOCKPILE_TICKS * int(tgt.production))
        return base

    initial = max(MIN_FLEET_SIZE, int(tgt.ships) + 1)
    _angle, eta = aim_and_eta(src, tgt, initial, omega, world=world)
    pred = float(model.ships_at(int(tgt.id), eta) or 0.0)
    return max(MIN_FLEET_SIZE, int(math.ceil(pred)) + 1)


def enumerate_ship_counts(src, tgt, model, omega: float, me: int, world) -> list[int]:
    """Fire-now ship counts: capture-size, 2× capture-size, full budget.

    `budget` is always emitted when ≥ MIN_FLEET_SIZE so multi-source
    bundles always have a column in the LP's outcome-table enumeration.
    """
    cap = capture_size(src, tgt, model, omega, me, world)
    budget = int(src.ships)
    if cap == 0:
        return []
    sizes = set()
    if MIN_FLEET_SIZE <= cap <= budget:
        sizes.add(cap)
    if 2 * cap <= budget:
        sizes.add(2 * cap)
    if budget >= MIN_FLEET_SIZE:
        sizes.add(budget)
    return sorted(sizes)


def min_wait_affordable(src, tgt, model, omega: float, me: int, world=None) -> int | None:
    """Smallest wait_N at which src can affordably capture tgt.

    Returns 0 if fire-now is feasible, N>0 if accumulation needed,
    None if hopeless within MAX_HORIZON.
    """
    if int(tgt.owner) == me:
        return None
    if int(src.production) <= 0:
        return None
    prod = int(src.production)

    initial = max(MIN_FLEET_SIZE, int(tgt.ships) + 1)
    _a0, eta0 = aim_and_eta(src, tgt, initial, omega, wait_N=0, world=world)
    pred_now = float(model.ships_at(int(tgt.id), eta0) or 0.0)
    cap_now = max(MIN_FLEET_SIZE, int(math.ceil(pred_now)) + 1)
    if cap_now <= int(src.ships):
        return 0

    for wait_N in range(1, MAX_HORIZON):
        budget = int(src.ships) + prod * wait_N
        if max(MIN_FLEET_SIZE, int(tgt.ships) + 1) > budget:
            continue
        target_fleet = max(MIN_FLEET_SIZE, int(tgt.ships) + 1)
        _angle, eta = aim_and_eta(src, tgt, target_fleet, omega, wait_N=wait_N, world=world)
        pred_at_arrival = float(model.ships_at(int(tgt.id), wait_N + eta) or 0.0)
        cap_final = max(MIN_FLEET_SIZE, int(math.ceil(pred_at_arrival)) + 1)
        if cap_final <= budget and wait_N + eta + SIM_SETTLE_TURNS <= MAX_HORIZON:
            return wait_N
    return None


def wait_then_fire_variants(src, tgt, model, omega: float, me: int, world=None):
    """Backward wait grid: anchor on min_wait_affordable, emit at
    {min_wait, min_wait + WAIT_BUFFER_OFFSET} with full-budget ships.

    Already-armed pairs (min_wait==0) return [] so fire-now isn't
    out-bid by a speculative wait variant. Hopeless pairs return [].
    """
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
        # Use the accumulation — emit full budget. Banding-dedup collapses
        # to one column per (src, tgt, wait_band); residue defends against
        # opp recapture in 4P.
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
    """Analytic Δ for pre-rank. Capture: +0.05 × prod × pv. Bounce: -0.5 × ships.
    Reinforce (mine, threatened within eta+30): pv-weighted loss-prevention.
    Orbital safety baked in: enemy threat is evaluated AT arrival step.
    """
    arrival_step = wait_N + eta
    pred_owner = model.owner_at(int(tgt.id), arrival_step)
    pred_ships = float(model.ships_at(int(tgt.id), arrival_step) or 0.0)

    if pred_owner == me:
        t_to_threat = model.time_to_enemy_threat(
            int(tgt.id), me, world, arrival_eta=int(arrival_step),
        )
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
    if wait_N == 0:
        return 0
    return 1 if wait_N <= 7 else 2


def _source_survives_launch(
    src, ships: int, wait_N: int, world, model, me: int,
) -> bool:
    """Would src still defend itself against the earliest known inbound
    threat after launching `ships` at wait_N?

    Catches "drain a planet exposed to a threat the rollout's horizon
    can't see" — chooser's leaf rollout is 25 ticks, threats landing
    30+ ticks later are invisible.
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
        return True  # potential-only; chooser rollout assesses
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


def _target_holdable_after_capture(
    src, tgt, ships: int, wait_N: int, eta: int, world, model, me: int,
) -> bool:
    """Would the captured target hold against the cheapest opp recapture?

    Orbital safety: target / ally / opp positions are predicted at
    `arrival_step` so an orbiting target that's far from opp now but
    close at arrival doesn't get a falsely-HOLDABLE verdict.
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

    omega = float(getattr(world, "omega", 0.0))
    use_predict = omega != 0.0 and arrival_step > 0
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


def _target_cost_parity_ok(
    src, tgt, ships: int, wait_N: int, eta: int, world, model, me: int,
) -> bool:
    """Drop launches where the cheapest opp reactor pays materially fewer
    ships than we do — we'd be the wasteful first-mover.

    Orbital safety: tgt / ally / opp positions predicted at arrival_step.
    Ally-closer safety valve: if some ally is strictly closer to tgt than
    every threatening opp, accept the launch (we're the cheap second-mover).
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

    omega = float(getattr(world, "omega", 0.0))
    use_predict = omega != 0.0 and arrival_step > 0
    if use_predict:
        tgt_x, tgt_y = _position_at(tgt, omega, arrival_step)
    else:
        tgt_x, tgt_y = float(tgt.x), float(tgt.y)

    nearest_us_dist = float("inf")
    for ally in world.planets_by_id.values():
        if int(ally.owner) != me:
            continue
        if int(ally.id) == int(tgt.id):
            continue
        if int(ally.id) == int(src.id):
            continue
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
        if nearest_us_dist < d:
            return True  # ally-closer safety valve
        flight = d - float(opp.radius) - float(tgt.radius) - 0.1
        if flight <= 0:
            continue
        opp_speed = fleet_speed(int(opp.ships))
        if opp_speed <= 0:
            continue
        opp_eta_after_landing = int(math.ceil(flight / opp_speed))
        garrison_at_recapture = (
            delivered + int(tgt.production) * opp_eta_after_landing
        )
        opp_launch_budget = (
            int(opp.ships) + int(opp.production) * arrival_step
        )
        opp_needed = int(math.ceil(garrison_at_recapture)) + 1
        if opp_needed > opp_launch_budget:
            continue
        opp_needed = max(MIN_FLEET_SIZE, opp_needed)
        if min_opp_reactor_cost is None or opp_needed < min_opp_reactor_cost:
            min_opp_reactor_cost = opp_needed

    if min_opp_reactor_cost is None:
        return True
    if float(min_opp_reactor_cost) < float(ships) * COST_PARITY_MARGIN:
        return False
    return True


def _enumerate_reactor_candidates(
    my_planets, world, model, me: int, omega: float, baseline_len: int,
):
    """For each non-our target with an opp fleet inbound that will
    capture it, propose our own recapture launches sized to take it
    back after opp lands. Output extends the standard prerank list.
    """
    if not my_planets:
        return []

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
            continue

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
            continue
        if post_owner == me:
            continue
        if int(post_owner) == -1:
            continue

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
            angle, eta = aim_and_eta(src, tgt, needed, omega, wait_N=wait_N, world=world)
            if wait_N + int(eta) + SIM_SETTLE_TURNS > MAX_HORIZON:
                continue
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


# ---------------------------------------------------------------------------
# Chooser — trajectory-first with favor-leaf Δ scoring + joint pairing.
# ---------------------------------------------------------------------------

def opp_actions_for_snap(snap, me: int, num_seats: int) -> list[list]:
    """One reactive opp action set per non-me seat via lite_greedy_policy."""
    actions: list[list] = [[] for _ in range(num_seats)]
    for opp_id in range(num_seats):
        if opp_id == me:
            continue
        try:
            actions[opp_id] = lite_greedy_policy(
                snap.state[opp_id].observation,
            ) or []
        except Exception:
            actions[opp_id] = []
    return actions


def affordable_validate_cap(snap_base, me: int, num_seats: int,
                            max_horizon: int, wallclock_ms: float,
                            min_horizon: int) -> tuple[int, float]:
    """Probe per-step + per-leaf cost; size candidate cap and per-cand cost
    so safe_deadline can pre-bail before entering a long rollout. Per-leaf
    cost matters because the hybrid head's composite branch builds a World
    + ray-casts every fleet (~2-5ms vs favor's ~100µs).
    """
    t0 = time.perf_counter()
    probe = fs_clone(snap_base)
    probe = fs_step(probe, [[] for _ in range(num_seats)], in_place=True)
    per_step_ms = max(0.05, (time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    favor_hybrid(probe.state[me].observation, me, num_seats, gamma=GAMMA)
    per_leaf_ms = max(0.05, (time.perf_counter() - t0) * 1000.0)

    avg_K = (min_horizon + max_horizon) / 2.0
    per_cand_ms = (per_step_ms * avg_K + per_leaf_ms) * PER_CANDIDATE_SAFETY
    budget = wallclock_ms - RESERVED_OVERHEAD_MS
    cap = max(8, int(budget / per_cand_ms))
    return cap, per_cand_ms


def build_trajectory_baseline(snap_base, me: int, num_seats: int,
                              horizon: int) -> list[float]:
    """Idle-baseline favor at every tick in [0, horizon] under
    (me-idle, opp-reactive). Used to subtract the do-nothing alternative
    from each candidate's leaf favor.
    """
    snap = fs_clone(snap_base)
    out: list[float] = [
        favor_hybrid(snap.state[me].observation, me, num_seats, gamma=GAMMA),
    ]
    for _ in range(horizon):
        if snap.fake_env.done:
            out.append(out[-1])
            continue
        actions = opp_actions_for_snap(snap, me, num_seats)
        snap = fs_step(snap, actions, in_place=True)
        out.append(
            favor_hybrid(snap.state[me].observation, me, num_seats, gamma=GAMMA),
        )
    return out


def score_candidate_v4(snap_base, src, tgt, ships: int, angle: float,
                       me: int, num_seats: int, world,
                       baseline_favors: list[float],
                       horizon: int, wait_N: int = 0,
                       ) -> tuple[float, str, int | None]:
    """v4 leaf: Δ = favor(my_action @ wait_N + opp reactive) − baseline[horizon].

    Admissibility filter (predict_fleet_fate) runs only for wait_N==0 —
    wait variants' geometry depends on launch-time orbital state; fast_sim's
    collision resolution catches sun/oob/comet hits inside the rollout.

    Returns (delta, status, eta). Status ∈ {sun, oob, timeout, planet
    (path_blocked / comet_collision), comet_expired, scored}.
    """
    eta = 0
    if int(wait_N) == 0:
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

    if horizon >= len(baseline_favors):
        horizon = len(baseline_favors) - 1

    snap = fs_clone(snap_base)
    for t in range(horizon):
        if snap.fake_env.done:
            break
        actions = opp_actions_for_snap(snap, me, num_seats)
        if t == int(wait_N):
            actions[me] = [[int(src.id), float(angle), int(ships)]]
        snap = fs_step(snap, actions, in_place=True)

    leaf = favor_hybrid(snap.state[me].observation, me, num_seats, gamma=GAMMA)
    return (leaf - baseline_favors[horizon], "scored", eta)


def score_candidate_v4_joint(snap_base, launches, me: int, num_seats: int,
                              world, baseline_favors: list[float],
                              horizon: int) -> tuple[float, str]:
    """Score a JOINT candidate (multiple launches in one rollout).

    `launches` = list of (src, tgt, ships, angle, wait_N). All injected
    at their respective wait_N in the SAME rollout.

    H44 fix (Day 12, 2026-05-22): per-leg admissibility now runs for
    ALL legs including wait_N>0 via predict_fleet_fate's wait_N
    parameter. The previous "wait_N!=0 → continue" bypass left ~65% of
    live in-flight deaths uncaught (H44 audit, btjeK 2026-05-20; fix
    in claude/extract-physics-trajectory-Vjaz9 commit c6a0c80).
    """
    for src, tgt, ships, angle, wait_N in launches:
        fate = predict_fleet_fate(
            src, tgt, angle, ships, world, wait_N=int(wait_N),
        )
        if fate.outcome in ("sun", "oob", "timeout", "planet"):
            return (float("-inf"), "admissibility_fail")
        if int(tgt.id) in world.comet_ids:
            life = comet_remaining_lifetime(int(tgt.id), world)
            if life is None or life <= int(fate.step):
                return (float("-inf"), "comet_expired")

    if horizon >= len(baseline_favors):
        horizon = len(baseline_favors) - 1

    inject_at: dict[int, list] = {}
    for src, tgt, ships, angle, wait_N in launches:
        inject_at.setdefault(int(wait_N), []).append(
            [int(src.id), float(angle), int(ships)],
        )

    snap = fs_clone(snap_base)
    for t in range(horizon):
        if snap.fake_env.done:
            break
        actions = opp_actions_for_snap(snap, me, num_seats)
        if t in inject_at:
            actions[me] = list(inject_at[t])
        snap = fs_step(snap, actions, in_place=True)

    leaf = favor_hybrid(snap.state[me].observation, me, num_seats, gamma=GAMMA)
    return (leaf - baseline_favors[horizon], "scored")



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


