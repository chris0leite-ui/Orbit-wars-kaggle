"""v8_scavenge — fast-sim chooser with opp-trajectory baseline subtraction.

Pipeline:
  Δ = favor(leaf with my_action at wait_N + opp_traj replayed)
      − favor(leaf with me idle, same opp_traj replayed)

`opp_traj` is pre-computed once per turn via `_opp_policy` (lite_greedy
with bounce-check) on each non-me seat's observation. Both baseline and
every candidate replay this SAME trajectory — common random numbers —
so opp's expansion cancels in Δ and only my action's marginal value
remains. The wait-N candidate's value emerges from this evaluation:
long waits are correctly penalised when opp's expansion outpaces my
hoarding, and rewarded when waiting unlocks a high-value near target
that fire-now can't afford.

Why the prior strict-idle baseline failed: when opp_idle was assumed,
my "fire fast at far-prod-target" candidate scored +386 at horizon=30
because the baseline saw no opp captures. In reality (Felipe seed
1492346051), opp captures 4 planets in those 30 turns and the right
play is "wait 17 turns, accumulate 31 ships, fire at near prod-5
neutral" (Δ=+475 under the corrected baseline). With the opp_traj fix
the chooser picks the wait-then-fire candidate naturally.

For ORBITING wait-N candidates the aim must rotate BOTH src and tgt
forward by omega*wait_N (the planets co-rotate, so relative geometry
is preserved at fire time). Rotating only the target — the prior code —
gave wildly wrong angles and inflated eta, blocking the wait-N
candidate via the MAX_HORIZON check.
"""

from __future__ import annotations

import math
import os
import time

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from lib.aim import aim_orbiting
from lib.fast_sim import clone as fs_clone
from lib.fast_sim import from_obs as fs_from_obs
from lib.fast_sim import step as fs_step
from lib.fleet import speed as fleet_speed
from lib.intent import World
from lib.opp_model import lite_greedy_policy as _opp_policy
from lib.orbit import is_orbiting as _is_orbiting
from lib.orbit import predict_relative as _orbit_predict_relative
from lib.scoring import pv_horizon
from lib.world_model import WorldModel

# ---------------------------------------------------------------------------
# Tunable knobs
# ---------------------------------------------------------------------------

EPISODE_STEPS = 500
NUM_TARGETS_PER_SOURCE = 8       # K nearest non-owned planets per source
MIN_FLEET_SIZE = 2               # 1-ship fleets are slow + rarely useful

# Two-stage scoring (Iteration 1, 2026-05-16):
# Cheap pre-rank with WorldModel.owner_at/ships_at (~0.1ms each), then
# only run the expensive fast_sim K-step rollout on the top
# N_VALIDATE candidates. The pre-rank's known weakness is misattributing
# orbital captures (lib.world_model.fleet_target_planet uses straight
# ray-cast) — but we only need RANK to be approximately right; the
# fast_sim validation is ground truth for the actual outcome.
#
# N_VALIDATE bumped 30→60 after first try regressed to 65.6% (Wlo 0.534)
# vs the 75% (Wlo 0.579) single-stage baseline. The cheap-rank was
# dropping borderline candidates that fast_sim would have scored
# positive; widening the validate pool restores most of the lift.
# Pre-rank filter also relaxed: include cheap-zero candidates (potential
# reinforcement/scavenge fast_sim might value).
#
# Budget impact: pre-rank ~15ms + validate ~60×5ms = ~315ms per turn —
# still well under the 1000ms ceiling.
N_VALIDATE = 60

# Forward-sim horizon parameters
SIM_SETTLE_TURNS = 2             # extra idle turns after arrival to settle combat
MIN_HORIZON = 25                 # Every candidate's rollout runs at least
                                 # this many steps. Long enough for a typical
                                 # capture to be exposed to opp's reactive
                                 # counter-launch (which is the real test
                                 # of whether a captured planet "sticks").
MAX_HORIZON = 40                 # baseline cache depth.

# Wallclock safety. The env's actTimeout is 1000ms. Two-stage scoring
# brought max from 1494ms (single-stage) down to 1131ms (Iter 1 panel)
# — still occasional outliers because the deadline check fires BETWEEN
# candidates, and a single fast_sim K-step rollout in mid-late game with
# many in-flight fleets can take 200-300ms (per-step cost ~10ms instead
# of the docs-stated 0.12ms with few fleets). Worst case used to be:
# budget(600) + one-slow-candidate(~300) + overhead(~30) = ~930ms in
# theory, but panel showed 1131ms outliers — slower per-step cost in
# practice.
#
# Adaptive fix (Iteration 1.1, 2026-05-16): measure per-step cost ONCE
# at the start of agent(), use it to compute N_AFFORDABLE_VALIDATE.
# Effective cap = min(N_VALIDATE, N_AFFORDABLE). Bounds the worst
# case to ~(budget + 1 candidate worth) ≈ 700ms reliably.
WALLCLOCK_BUDGET_MS = 600.0

# Parity-test override: the bundle-parity gate sets this env var to
# effectively unbound the budget, so every candidate is scored and the
# agent becomes a pure function of `obs`. Otherwise mid-candidate-list
# deadline bails create source-vs-bundle action drift from CPU jitter
# alone — exactly what the parity gate is designed to catch as a real
# bundling defect, but timing isn't one. Pattern lifted from
# `lib/v7_search.py::_WALLCLOCK_ENV_VAR`.
_WALLCLOCK_ENV_VAR = "ORBIT_WARS_PARITY_WALLCLOCK_MS"


def _effective_wallclock_ms() -> float:
    override = os.environ.get(_WALLCLOCK_ENV_VAR)
    if not override:
        return WALLCLOCK_BUDGET_MS
    try:
        return float(override)
    except ValueError:
        return WALLCLOCK_BUDGET_MS

# Safety factor on the per-candidate cost estimate. fast_sim's per-step
# cost varies within a rollout (combat steps are slower than no-combat
# steps), so a one-shot measurement underestimates. 1.5× covers the
# variance.
_PER_CANDIDATE_SAFETY = 1.5
# Reserved for non-validate work (pre-rank, baseline build, emit).
_RESERVED_OVERHEAD_MS = 50.0

# v12: there is no hard MAX_WAIT cap. With the full opp_trajectory
# replayed in baseline + every candidate (common random numbers),
# wait-N's value emerges from evaluation; long waits are correctly
# penalised when opp's expansion outpaces my hoarding, and rewarded
# when waiting unlocks a high-prod near target that fire-now can't
# afford. The only remaining cap on wait_N is structural:
# `wait_N + eta + SIM_SETTLE_TURNS ≤ MAX_HORIZON` — a computational
# horizon bound, not a behavioural restriction.


# ---------------------------------------------------------------------------
# Obs helpers
# ---------------------------------------------------------------------------


def _as_dict(obs):
    """Coerce an obs (Struct or dict) into a dict for consistent access."""
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


def _num_seats(planets, fleets):
    """Detect 2P vs 4P from the obs."""
    max_owner = -1
    for p in planets:
        if int(p.owner) > max_owner:
            max_owner = int(p.owner)
    for f in fleets:
        if int(f.owner) > max_owner:
            max_owner = int(f.owner)
    return 4 if max_owner >= 2 else 2


# ---------------------------------------------------------------------------
# Geometry / timing primitives
# ---------------------------------------------------------------------------


def _aim_and_eta(src, tgt, ships, omega, wait_N=0):
    """Return (lead_aim_angle, integer_eta) for one candidate fleet.

    For ORBITING targets, `lib.aim.aim_orbiting` jointly solves the
    aim angle AND the arrival eta via fixed-point iteration.

    For wait-then-fire candidates (wait_N > 0): the fleet fires
    `wait_N` turns AFTER the current step. By then BOTH src and tgt
    have rotated by `omega * wait_N` around the center. We pre-rotate
    BOTH endpoints via `predict_relative` so aim_orbiting operates on
    the geometry that will hold at fire time (verified empirically on
    Felipe seed 1492346051: at step 4 with wait_N=17, shifting both
    matches the step-21 ground-truth aim to 0.001 rad). Rotating only
    the target — as the prior code did — gave a wildly wrong aim
    because it computed "from src at step 4 toward tgt at step 21,"
    which is not what the fleet would do when launched at step 21.

    Falls back to straight-aim + straight-eta for non-orbiting
    targets (static planets don't rotate; wait_N is a no-op for aim).
    """
    if _is_orbiting(list(tgt)):
        tgt_list = list(tgt)
        src_x, src_y = float(src.x), float(src.y)
        if wait_N > 0:
            # Rotate both endpoints to their position at fire time.
            # Co-rotating planets preserve their relative geometry,
            # so the angle returned by aim_orbiting from rotated_src
            # to rotated_tgt is the correct world-frame aim at step
            # (current_step + wait_N) — exactly what fast_sim will
            # use when it replays wait_N idle steps then fires.
            fx, fy = _orbit_predict_relative(tgt_list, omega, wait_N)
            tgt_list = list(tgt_list)
            tgt_list[2] = fx
            tgt_list[3] = fy
            src_x, src_y = _orbit_predict_relative(list(src), omega, wait_N)
        res = aim_orbiting(
            (src_x, src_y), src.radius, tgt_list, tgt.radius, ships, omega,
        )
        if res is not None:
            return float(res[0]), max(1, int(math.ceil(float(res[2]))))
    angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
    flight = max(
        0.0,
        math.hypot(src.x - tgt.x, src.y - tgt.y)
        - src.radius - tgt.radius - 0.1,
    )
    spd = fleet_speed(ships)
    if spd <= 0:
        return angle, 999
    return angle, int(math.ceil(flight / spd))


def _nearest_k(targets, src, k):
    return sorted(
        targets,
        key=lambda t: math.hypot(src.x - t.x, src.y - t.y),
    )[:k]


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def _capture_size(src, tgt, model, omega, me, world):
    """WorldModel-aware minimum capture size.

    For NON-MINE targets (capture): predicted defenders at eta + 1.
    For MINE targets (reinforce): the predicted SHORTFALL against the
      strongest incoming enemy fleet, i.e. enough ships to win the
      defense at the moment of conflict. If no incoming threat,
      returns 0 (no reinforce needed).

    One Newton-style iteration: initial size from current tgt garrison,
    use orbital-aware `_aim_and_eta` for arrival turn, query model for
    predicted defenders at that eta.
    """
    if int(tgt.owner) == me:
        # Reinforce: size to survive the predicted enemy threat. Uses
        # `time_to_enemy_threat` which covers both in-flight fleets AND
        # potential launches from stationary enemy planets at current
        # garrisons (Fix 2 of v9). Drop-in widening of the threat pool.
        enemy_eta = model.time_to_enemy_threat(int(tgt.id), me, world)
        if enemy_eta is None:
            return 0  # no near-term threat; reinforce unnecessary
        # Sum in-flight enemy ships landing at-or-before enemy_eta + 1.
        enemy_arrivals = model.ledger.get(int(tgt.id), [])
        enemy_ship_sum_inflight = sum(
            ships for (eta_arr, owner, ships) in enemy_arrivals
            if owner != me and eta_arr <= enemy_eta + 1
        )
        # If no in-flight threat (preemptive case), the threat is a
        # potential launch from a stationary enemy planet. Estimate the
        # threat magnitude as the nearest enemy planet's CURRENT garrison
        # (worst-case full send).
        enemy_potential = 0.0
        if enemy_ship_sum_inflight <= 0:
            tgt_x, tgt_y = float(tgt.x), float(tgt.y)
            best_enemy_ships = 0.0
            for p in world.planets_by_id.values():
                if int(p.owner) < 0 or int(p.owner) == me:
                    continue
                # Match the enemy's eta range — they could have launched
                # at most enemy_eta turns ago from any of their planets.
                if int(p.ships) > best_enemy_ships:
                    best_enemy_ships = float(p.ships)
            enemy_potential = best_enemy_ships
        enemy_strength = max(enemy_ship_sum_inflight, enemy_potential)
        # Predicted defender at enemy_eta (with production accrual).
        my_garrison_at_eta = float(tgt.ships) + float(tgt.production) * enemy_eta
        shortfall = enemy_strength - my_garrison_at_eta + 1
        return max(0, int(math.ceil(shortfall)))
    # Capture (non-mine target)
    initial = max(MIN_FLEET_SIZE, int(tgt.ships) + 1)
    _angle, eta = _aim_and_eta(src, tgt, initial, omega)
    pred = float(model.ships_at(int(tgt.id), eta) or 0.0)
    size = int(math.ceil(pred)) + 1
    return max(MIN_FLEET_SIZE, size)


def _enumerate_ship_counts_basic(src, tgt, model, omega, me, world):
    """Phase 1 ship-count set: capture/reinforce size, 2×, full budget.

    For reinforce (my own target), size 0 means no threat → skip.
    """
    cap = _capture_size(src, tgt, model, omega, me, world)
    budget = int(src.ships)
    if cap == 0:
        return []  # no threat; don't reinforce
    sizes = set()
    if MIN_FLEET_SIZE <= cap <= budget:
        sizes.add(cap)
    if 2 * cap <= budget:
        sizes.add(2 * cap)
    if budget >= MIN_FLEET_SIZE and budget > cap:
        sizes.add(budget)
    return sorted(sizes)


def _wait_then_fire_candidate(src, tgt, model, omega, me):
    """Generate a single "wait N turns then fire" candidate for one
    (src, tgt) pair, OR None if not applicable. v10 mechanism.

    Triggers ONLY when:
    - tgt is not mine (reinforces are deadline-bound; can't wait)
    - src.production > 0 (otherwise can't accumulate)
    - capture-size NOW exceeds src.ships (infeasible-now)
    - wait_N ≥ 1 (otherwise the fire-now path covers this pair)
    - wait_N + eta + SETTLE ≤ MAX_HORIZON (computational cap;
      beyond MAX_HORIZON we cannot evaluate the leaf state)

    Returns (ships, wait_N, angle, eta) for the validate stage.

    Newton-iteration like `_capture_size` non-mine branch: estimate
    cap at NOW eta, derive wait_N, then re-iterate eta/cap at the
    post-wait arrival time (a larger fleet arrives faster).

    Targets the Felipe-Ferreira 2P loss pattern (replay 76655989):
    the prod-4 near target was infeasible-now (cap=20 > 14 budget) so
    v8 dropped it. With this enumerator, the wait-6-then-fire variant
    enters the candidate pool and fast_sim picks it if Δ > fire-now.
    """
    if int(tgt.owner) == me:
        return None
    prod = int(src.production)
    if prod <= 0:
        return None

    initial = max(MIN_FLEET_SIZE, int(tgt.ships) + 1)
    _a0, eta0 = _aim_and_eta(src, tgt, initial, omega)
    pred_now = float(model.ships_at(int(tgt.id), eta0) or 0.0)
    cap_now = max(MIN_FLEET_SIZE, int(math.ceil(pred_now)) + 1)
    if cap_now <= int(src.ships):
        return None  # feasible-now; fire-now path covers this pair

    shortfall = cap_now - int(src.ships)
    wait_N = (shortfall + prod - 1) // prod  # ceil
    if wait_N < 1:
        return None

    # Newton step at post-wait arrival. Pass wait_N to _aim_and_eta so
    # the orbital lead-prediction accounts for the target's rotation
    # DURING the wait phase — otherwise the angle is correct for
    # "fire now, intercept at eta" but wrong for "fire at wait_N,
    # intercept at wait_N + eta". Bug root-caused on Felipe seed
    # 1492346051 turn 15: fleet fired but tgt stayed neutral at leaf.
    ships_attempt = cap_now + 1
    angle, eta = _aim_and_eta(src, tgt, ships_attempt, omega, wait_N=wait_N)
    pred_at_arr = float(model.ships_at(int(tgt.id), wait_N + eta) or 0.0)
    cap_final = max(MIN_FLEET_SIZE, int(math.ceil(pred_at_arr)) + 1)

    budget_at_wait = int(src.ships) + prod * wait_N
    if cap_final > budget_at_wait:
        return None

    if wait_N + eta + SIM_SETTLE_TURNS > MAX_HORIZON:
        return None

    return cap_final, wait_N, angle, eta


# ---------------------------------------------------------------------------
# Cheap pre-rank (Stage 1 of two-stage scoring)
# ---------------------------------------------------------------------------


def _cheap_marginal_value(src, tgt, ships, eta, world, model, me, wait_N=0):
    """Approximate Δ value for ranking only — NOT the final score.

    Reads the BASELINE WorldModel (built once per turn) to predict
    pred_owner + pred_ships at our arrival eta.

    Three cases:
    - **CAPTURE** (pred_owner != me, ships > pred_ships): credit by
      capture_weight × production × time_remaining.
    - **BOUNCE** (pred_owner != me, ships ≤ pred_ships): penalty
      = −waste_weight × ships.
    - **REINFORCE** (pred_owner == me): if WorldModel.time_to_enemy_threat
      predicts an enemy could attack this planet within a relevant
      horizon (eta + 30), score as "value of preventing loss" =
      capture_weight × production × pv_horizon(threat_eta).
      Otherwise return 0 (no near-term threat → reinforce unnecessary).

    The reinforce branch is Fix 1 of the v9 iteration: previously
    returned 0 for all reinforce, which caused them to be cut from
    the validate stage by the adaptive wallclock cap in late game.
    Now reinforce candidates RANK competitive with captures so they
    reach fast_sim validation.

    Known weakness: `fleet_target_planet` does a non-orbital ray-cast,
    so for orbital captures the model's predicted state at our eta is
    off by 1-2 turns of orbital drift. This is acceptable for RANKING:
    relative ordering is mostly preserved. The fast_sim downstream is
    the ground truth for the FINAL decision.
    """
    # v10: arrival_step = wait_N + eta. For fire-now wait_N=0 → unchanged.
    arrival_step = wait_N + eta
    pred_owner = model.owner_at(int(tgt.id), arrival_step)
    pred_ships = float(model.ships_at(int(tgt.id), arrival_step) or 0.0)

    if pred_owner == me:
        # REINFORCE: score value of preventing loss of this planet.
        # (wait-N is not generated for reinforce targets, so this branch
        # only sees fire-now reinforce candidates.)
        t_to_threat = model.time_to_enemy_threat(int(tgt.id), me, world)
        if t_to_threat is None or t_to_threat > eta + 30:
            return 0.0  # no near-term threat → reinforce truly unnecessary
        # Loss-prevention credit: planet's pv-discounted production
        # stream from threat onward, scaled by capture_weight (0.05) to
        # match the offensive capture-credit scale.
        pv = pv_horizon(int(world.step), int(t_to_threat),
                        gamma=0.99, t_total=EPISODE_STEPS)
        return 0.05 * float(tgt.production) * float(pv)

    if ships > pred_ships:
        # CAPTURE credit. PV-discounted production stream from arrival.
        # For wait-N: discount is γ^(wait_N+eta), matching the longer
        # delay before production begins.
        pv = pv_horizon(int(world.step), int(arrival_step),
                        gamma=0.99, t_total=EPISODE_STEPS)
        return 0.05 * float(tgt.production) * float(pv)
    # BOUNCE penalty (waste_weight=0.5).
    return -0.5 * float(ships)


# ---------------------------------------------------------------------------
# Favor (F1 + F2) — bootstrap's proven leaf scorer
# ---------------------------------------------------------------------------


def _favor(obs, me, num_seats=2):
    """F1 + F2 favor with PV-discount and 4P-aware opp aggregation.

    F1 = my_ships − opp_ships_agg (in-flight + planets).
    F2 = (my_prod − opp_prod_agg) × pv_horizon(step, 0, γ=0.99).

    PV-discount (Fix 3 of v9): linear `turns_remaining` over-weights
    far-future production. In late-game with opp prod-lead, F2 dominates
    F1 by 100× and the chooser stops valuing ship preservation. PV
    with γ=0.99 makes a unit production stream worth ~99 (vs 500), so
    F1 and F2 are on comparable scales.

    4P-aware opp aggregation (Fix 4 of v9): in 2P use max-of-opps
    (identical to "the only opp"); in 4P use SUM-of-opps so capturing
    from a weak opp gets 2× credit (my +prod AND their −prod),
    matching the credit for capturing from the leader. This corrects
    the systematic under-credit of non-leader captures that left v8
    passive in 4P.
    """
    planets = obs.planets if hasattr(obs, "planets") else obs.get("planets", [])
    fleets = obs.fleets if hasattr(obs, "fleets") else obs.get("fleets", [])
    step = obs.step if hasattr(obs, "step") else obs.get("step", 0)

    # Per-owner totals
    ships_by_owner = {}
    prod_by_owner = {}
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
    if num_seats <= 2:
        opp_ships = max(
            (v for k, v in ships_by_owner.items() if k != me),
            default=0.0,
        )
        opp_prod = max(
            (v for k, v in prod_by_owner.items() if k != me),
            default=0.0,
        )
    else:
        # 4P / 3P: sum across all opps. Capturing from any weakens the
        # collective; same credit as capturing from the leader.
        opp_ships = sum(v for k, v in ships_by_owner.items() if k != me)
        opp_prod = sum(v for k, v in prod_by_owner.items() if k != me)

    pv = pv_horizon(int(step), 0, gamma=0.99, t_total=EPISODE_STEPS)
    return (my_ships - opp_ships) + (my_prod - opp_prod) * pv


# ---------------------------------------------------------------------------
# Idle baseline + per-candidate score
# ---------------------------------------------------------------------------


def _opp_actions_for_snap(snap, me, num_seats):
    """Compute each non-me seat's lite_greedy action against the CURRENT
    snap. Used inline by both baseline and candidate rollouts so opp
    reacts to the evolving state (including my fleets and captures)
    rather than replaying a precomputed trajectory."""
    actions = [[] for _ in range(num_seats)]
    for opp_id in range(num_seats):
        if opp_id == me:
            continue
        try:
            actions[opp_id] = _opp_policy(snap.state[opp_id].observation) or []
        except Exception:
            actions[opp_id] = []
    return actions


def _build_idle_baseline(snap_base, me, num_seats, max_horizon):
    """Pre-compute favor at every horizon 0..max_horizon under me-idle.

    Opp acts REACTIVELY at each step via `_opp_policy` against the
    evolving snap. Lost CRN cancellation (vs precomputed opp_traj),
    gained: opp counter-attacks against my captures in candidate
    rollouts emerge naturally, so F2 over-credit on fragile captures
    is corrected at the leaf.
    """
    snap = fs_clone(snap_base)
    out = [_favor(snap.state[me].observation, me, num_seats)]
    for _step_i in range(max_horizon):
        if snap.fake_env.done:
            out.append(out[-1])
            continue
        actions = _opp_actions_for_snap(snap, me, num_seats)
        # me slot stays [] (idle baseline)
        snap = fs_step(snap, actions, in_place=True)
        out.append(_favor(snap.state[me].observation, me, num_seats))
    return out


def _score_action(snap_base, me, num_seats, src_id, angle, ships,
                  horizon, baseline_favors, wait_N=0):
    """Δ favor at horizon = leaf(me_action @ wait_N + reactive opp) − baseline.

    Opp acts reactively at each step (lite_greedy on opp's evolving obs),
    so my captured planets DO trigger opp counter-launches in the rollout
    — which collapses F2's over-credit on fragile (low-garrison) captures.
    """
    snap = fs_clone(snap_base)
    for step_i in range(horizon):
        if snap.fake_env.done:
            break
        actions = _opp_actions_for_snap(snap, me, num_seats)
        if step_i == int(wait_N):
            actions[me] = [[int(src_id), float(angle), int(ships)]]
        # else: actions[me] stays []
        snap = fs_step(snap, actions, in_place=True)
    leaf_favor = _favor(snap.state[me].observation, me, num_seats)
    return leaf_favor - baseline_favors[horizon]


# ---------------------------------------------------------------------------
# Public agent
# ---------------------------------------------------------------------------


def agent(obs, configuration=None):
    obs_d = _as_dict(obs)
    me = int(obs_d.get("player", 0))
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

    # Identify threatened MY planets via WorldModel.time_to_enemy_threat,
    # which considers BOTH (a) in-flight enemy fleets AND (b) potential
    # launches from stationary enemy planets at current garrison sizes.
    # The previous version used `incoming_enemy_eta` (only in-flight)
    # and missed preemptive threats from large enemy garrisons that
    # could launch any turn — the dominant failure mode in the Naoism
    # 2P loss (turn 70-95 attrition). Fix 2 of v9 iteration.
    threatened_mine = [
        p for p in my_planets
        if model.time_to_enemy_threat(int(p.id), me, world) is not None
    ]
    # Target pool = capture targets + defensive reinforce targets
    target_pool = other_planets + threatened_mine

    # Build the fast_sim snapshot once per turn (~1 ms).
    snap_base = fs_from_obs(obs, num_seats=num_seats)

    # Probe fast_sim per-step cost for THIS board state — used to bound
    # how many candidates we can afford to validate inside the wallclock
    # budget. Per-step cost varies with the number of in-flight fleets
    # (mid-late game can be 5-15× the empty-board cost). One step + a
    # clock measurement, ~1-3ms total.
    t_probe = time.perf_counter()
    probe_snap = fs_clone(snap_base)
    probe_snap = fs_step(probe_snap, [[] for _ in range(num_seats)],
                         in_place=True)
    per_step_ms = max(0.05, (time.perf_counter() - t_probe) * 1000.0)
    # Expected per-candidate cost: K steps × per_step_ms × safety
    # factor. Use the AVERAGE expected K (MIN_HORIZON + a few) as the
    # estimate; outliers get caught by the post-loop deadline guard.
    avg_K = (MIN_HORIZON + MAX_HORIZON) / 2.0
    per_cand_ms = per_step_ms * avg_K * _PER_CANDIDATE_SAFETY
    # How many candidates fit inside the budget AFTER reserving overhead?
    wallclock_ms = _effective_wallclock_ms()
    budget_for_validate = wallclock_ms - _RESERVED_OVERHEAD_MS
    n_affordable = max(8, int(budget_for_validate / per_cand_ms))

    # v13: reactive opp inside every rollout. lite_greedy is recomputed
    # against the evolving snap at each step, so my captured planets
    # trigger opp counter-launches in the rollout — collapsing F2's
    # over-credit on fragile (low-garrison) captures. Drops the
    # precomputed opp_traj and common-random-numbers cancellation;
    # accepts more Δ variance for realistic counter-attacks.
    baseline_favors = _build_idle_baseline(
        snap_base, me, num_seats, MAX_HORIZON,
    )

    # ---------------------------------------------------------------
    # Stage 1: cheap pre-rank via analytic marginal_value (~0.1ms each).
    # Enumerate every (src, tgt, ships) candidate, rank by approximate Δ.
    # Also append ONE "wait-then-fire" candidate per (src, tgt) where
    # capture is INFEASIBLE-NOW but feasible inside MAX_HORIZON.
    # ---------------------------------------------------------------
    # Prerank tuple: (cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N).
    # wait_N=0 means fire-now; wait_N>0 means idle-then-fire.
    prerank = []
    for src in my_planets:
        if int(src.ships) < MIN_FLEET_SIZE:
            continue
        for tgt in _nearest_k(target_pool, src, NUM_TARGETS_PER_SOURCE):
            if int(tgt.id) == int(src.id):
                continue
            # Fire-now candidates.
            for ships in _enumerate_ship_counts_basic(src, tgt, model, omega, me, world):
                if ships < MIN_FLEET_SIZE or ships > int(src.ships):
                    continue
                angle, eta = _aim_and_eta(src, tgt, ships, omega)
                horizon = max(eta + SIM_SETTLE_TURNS, MIN_HORIZON)
                if horizon >= len(baseline_favors):
                    horizon = len(baseline_favors) - 1
                cheap = _cheap_marginal_value(
                    src, tgt, ships, eta, world, model, me, wait_N=0,
                )
                if cheap > -10.0:
                    prerank.append(
                        (cheap, src, tgt, ships, angle, eta, horizon, 0)
                    )
            # v10 wait-then-fire candidate (one per pair when applicable).
            wt = _wait_then_fire_candidate(src, tgt, model, omega, me)
            if wt is not None:
                w_ships, w_wait_N, w_angle, w_eta = wt
                w_horizon = max(w_wait_N + w_eta + SIM_SETTLE_TURNS, MIN_HORIZON)
                if w_horizon < len(baseline_favors):
                    w_cheap = _cheap_marginal_value(
                        src, tgt, w_ships, w_eta, world, model, me,
                        wait_N=w_wait_N,
                    )
                    if w_cheap > -10.0:
                        prerank.append(
                            (w_cheap, src, tgt, w_ships, w_angle, w_eta,
                             w_horizon, w_wait_N)
                        )

    if not prerank:
        return []

    # Stage 2: per-(src, tgt) deduplication — for each (src, tgt), keep the
    # best-cheap-ranked candidate (fire-now OR wait-N — they compete).
    best_per_pair = {}  # (src_id, tgt_id) → entry
    for entry in prerank:
        cheap, src, tgt, _ships, _angle, _eta, _horizon, _wait_N = entry
        key = (int(src.id), int(tgt.id))
        prev = best_per_pair.get(key)
        if prev is None or cheap > prev[0]:
            best_per_pair[key] = entry
    deduped = list(best_per_pair.values())

    # Stage 3: validate the top candidates via fast_sim K-step rollout.
    deduped.sort(key=lambda e: -e[0])
    effective_cap = min(N_VALIDATE, n_affordable)
    top = deduped[:effective_cap]

    t_deadline = time.perf_counter() + wallclock_ms / 1000.0
    candidates = []  # validated (delta, src, tgt, ships, angle, wait_N)
    for _cheap, src, tgt, ships, angle, _eta, horizon, wait_N in top:
        if time.perf_counter() > t_deadline:
            break
        delta = _score_action(
            snap_base, me, num_seats,
            int(src.id), angle, ships,
            horizon, baseline_favors, wait_N=wait_N,
        )
        # v15a: keep every validated candidate. The `delta > 0` gate
        # filtered out "least-bad" actions when the idle baseline was
        # itself a losing trajectory — the dominant failure mode in
        # the Forrest 2P loss (31/32 candidates failed `> 0` at step
        # 190 with 13P/365 ships). Best-per-source dogpile dedup below
        # decides which Δ ≤ 0 actions actually emit.
        candidates.append((delta, src, tgt, ships, angle, wait_N))

    if not candidates:
        return []

    # Greedy non-dogpile emit: max 1 launch per source / per target per turn.
    # A wait-N candidate that "wins" a source RESERVES it — emit nothing
    # this turn; next turn the chooser re-evaluates with one less wait
    # turn needed. The actual launch happens when wait_N decays to 0.
    #
    # v15a: per-source surplus guard. A Δ ≤ 0 candidate is allowed to
    # emit only when the source has enough ships to absorb the loss
    # without crippling itself — guards against early-game weak
    # sources draining on -100 Δ candidates. Late-game large-garrison
    # sources clear the guard easily and always launch their best.
    candidates.sort(key=lambda c: -c[0])
    used_srcs, used_tgts = set(), set()
    moves = []
    for delta, src, tgt, ships, angle, wait_N in candidates:
        sid = int(src.id)
        tid = int(tgt.id)
        if sid in used_srcs or tid in used_tgts:
            continue
        if delta <= 0.0:
            surplus_floor = max(MIN_FLEET_SIZE * 3, int(src.production) * 5)
            if int(src.ships) < surplus_floor:
                continue
        used_srcs.add(sid)
        used_tgts.add(tid)
        if wait_N == 0:
            moves.append([sid, float(angle), int(ships)])
        # else: wait-N picked → reserve src/tgt, emit nothing this turn.
    return moves
