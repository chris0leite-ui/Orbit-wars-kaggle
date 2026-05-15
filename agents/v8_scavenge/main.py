"""v8_scavenge — fast-sim K-step chooser with idle-baseline subtraction.

Approach (PI direction; Fix A on the 2026-05-16 falsification):

Phase 1's depth-0 analytic chooser failed (0/32 vs v7_0) because
`lib.world_model.WorldModel.fleet_target_planet` uses a NON-orbital
straight ray-cast for fleet attribution. For orbiting fleet-target
combos the ray-cast misses (the fleet is AIMED via aim_orbiting at the
target's future position, but the ray-cast checks the planet's
current position). Predictions of "will my fleet capture?" were
unreliable.

Fix A: instead of analytic prediction, run `lib.fast_sim` (the parity-
tested simulator — same physics as the env) for K turns per candidate,
where K = max(eta + SIM_SETTLE_TURNS, MIN_HORIZON). The simulator
correctly handles orbital motion, swept-pair collisions, sun crossings,
and combat resolution — no ray-cast attribution needed.

Scoring:
  Δ = favor(leaf_after_my_action_and_K-1_idle_steps) − favor(leaf_after_K_idle_steps)

Both leaves are at the same horizon, so natural production growth
cancels and the Δ reflects only the action's marginal effect. Bootstrap
session's bug-1 fix: the baseline must be idle-at-same-horizon, NOT
favor-at-current-state (would over-credit by the natural growth during
the K-turn fly time).

The "macro moves" framing is preserved: we don't speculate about opp's
new actions; opp idles inside our rollout. What CAN'T be deferred is
the simulator's exact prediction of where my fleet ends up — which
the analytic ray-cast got wrong for orbital cases. fast_sim is the
right primitive for that prediction.

Phase 1 enumerates basic ship sizes (capture, 2×, full budget); Phase
2 will add scavenge-timed sizes. Phase 3 will use settle_plan instead
of greedy non-dogpile emit.
"""

from __future__ import annotations

import math
import time

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from lib.aim import aim_orbiting
from lib.fast_sim import clone as fs_clone
from lib.fast_sim import from_obs as fs_from_obs
from lib.fast_sim import step as fs_step
from lib.fleet import speed as fleet_speed
from lib.intent import World
from lib.orbit import is_orbiting as _is_orbiting
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
MIN_HORIZON = 15                 # floor — must cover incoming threats arriving
                                 # at our source planets in ~time for fast fleet
MAX_HORIZON = 30                 # baseline cache depth — covers most candidate
                                 # etas (typical eta range 5-25); long-arc
                                 # candidates get clipped to MAX_HORIZON.
                                 # Was 50; lowered to reduce per-turn baseline
                                 # build cost + bound per-candidate rollout.

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

# Safety factor on the per-candidate cost estimate. fast_sim's per-step
# cost varies within a rollout (combat steps are slower than no-combat
# steps), so a one-shot measurement underestimates. 1.5× covers the
# variance.
_PER_CANDIDATE_SAFETY = 1.5
# Reserved for non-validate work (pre-rank, baseline build, emit).
_RESERVED_OVERHEAD_MS = 50.0

# v10: max turns to wait before firing on an INFEASIBLE-NOW target.
# Generates ONE wait-N candidate per (src, tgt) where capture_size >
# src.ships and src can accumulate the shortfall within MAX_WAIT turns.
# Origin: Felipe Ferreira 2P loss (replay 76655989) where my chooser
# fired at far prod-2 target (eta=26) instead of waiting 6 turns to
# fire at near prod-4 (eta=10, ~2× more valuable).
# Bound at 10 to limit per-candidate rollout cost (horizon = wait + eta
# + SETTLE must stay ≤ MAX_HORIZON=30).
MAX_WAIT = 10


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


def _aim_and_eta(src, tgt, ships, omega):
    """Return (lead_aim_angle, integer_eta) for one candidate fleet.

    For ORBITING targets, `lib.aim.aim_orbiting` jointly solves the
    aim angle AND the arrival eta. Using its eta is load-bearing —
    a naïve `distance / speed` is wrong by 3-4× for orbital targets
    (Phase 1 bug, fixed 2026-05-16).

    Falls back to straight-aim + straight-eta for non-orbiting targets.
    """
    if _is_orbiting(list(tgt)):
        res = aim_orbiting(
            (src.x, src.y), src.radius, list(tgt), tgt.radius, ships, omega,
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
    - wait_N to afford capture is within [1, MAX_WAIT]
    - wait_N + eta + SETTLE ≤ MAX_HORIZON (horizon stays in bounds)

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
    if wait_N < 1 or wait_N > MAX_WAIT:
        return None

    # Newton step at post-wait arrival.
    ships_attempt = cap_now + 1
    angle, eta = _aim_and_eta(src, tgt, ships_attempt, omega)
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


def _build_idle_baseline(snap_base, me, num_seats, max_horizon):
    """Pre-compute favor at every idle horizon 0..max_horizon.

    Run all-idle fast_sim from snap_base, recording favor(me) at each
    step. The baseline_favors[k] is the favor when no one acts for k
    turns. Used for per-candidate horizon-matched Δ.
    """
    snap = fs_clone(snap_base)
    out = [_favor(snap.state[me].observation, me, num_seats)]
    idle = [[] for _ in range(num_seats)]
    for _ in range(max_horizon):
        if snap.fake_env.done:
            out.append(out[-1])
            continue
        snap = fs_step(snap, idle, in_place=True)
        out.append(_favor(snap.state[me].observation, me, num_seats))
    return out


def _score_action(snap_base, me, num_seats, src_id, angle, ships,
                  horizon, baseline_favors, wait_N=0):
    """Δ favor at horizon = (leaf with my plan + idle) − idle baseline.

    For wait_N=0 (fire-now): step 0 fires; steps 1..horizon-1 idle.
    For wait_N>0 (wait-then-fire): steps 0..wait_N-1 idle; step wait_N
    fires; steps wait_N+1..horizon-1 idle.

    Same `horizon` and same `baseline_favors[horizon]` regardless of
    wait_N — both leaf and baseline span the same total turns so the
    natural production growth cancels in the Δ subtraction. The wait
    candidate's lift over fire-now (if any) comes purely from arriving
    at a more valuable target via a bigger fleet.
    """
    snap = fs_clone(snap_base)
    idle_actions = [[] for _ in range(num_seats)]

    # Wait phase: idle for wait_N turns.
    for _ in range(int(wait_N)):
        if snap.fake_env.done:
            break
        snap = fs_step(snap, idle_actions, in_place=True)

    # Fire phase: apply my launch one step.
    if not snap.fake_env.done:
        actions = [[] for _ in range(num_seats)]
        actions[me] = [[int(src_id), float(angle), int(ships)]]
        snap = fs_step(snap, actions, in_place=True)

    # Settle phase: idle for the remaining horizon turns.
    remaining = horizon - int(wait_N) - 1
    for _ in range(max(0, remaining)):
        if snap.fake_env.done:
            break
        snap = fs_step(snap, idle_actions, in_place=True)

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
    budget_for_validate = WALLCLOCK_BUDGET_MS - _RESERVED_OVERHEAD_MS
    n_affordable = max(8, int(budget_for_validate / per_cand_ms))

    # Idle baseline at horizons 0..MAX_HORIZON (~6 ms for 30 steps).
    baseline_favors = _build_idle_baseline(snap_base, me, num_seats, MAX_HORIZON)

    # ---------------------------------------------------------------
    # Stage 1: cheap pre-rank via analytic marginal_value (~0.1ms each).
    # Enumerate every (src, tgt, ships) candidate, rank by approximate Δ.
    # v10: also append ONE "wait-then-fire" candidate per (src, tgt)
    # where capture is INFEASIBLE-NOW but feasible within MAX_WAIT.
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

    t_deadline = time.perf_counter() + WALLCLOCK_BUDGET_MS / 1000.0
    candidates = []  # validated (delta, src, tgt, ships, angle, wait_N)
    for _cheap, src, tgt, ships, angle, _eta, horizon, wait_N in top:
        if time.perf_counter() > t_deadline:
            break
        delta = _score_action(
            snap_base, me, num_seats,
            int(src.id), angle, ships,
            horizon, baseline_favors, wait_N=wait_N,
        )
        if delta > 0:
            candidates.append((delta, src, tgt, ships, angle, wait_N))

    if not candidates:
        return []

    # Greedy non-dogpile emit: max 1 launch per source / per target per turn.
    # A wait-N candidate that "wins" a source RESERVES it — emit nothing
    # this turn; next turn the chooser re-evaluates with one less wait
    # turn needed. The actual launch happens when wait_N decays to 0.
    candidates.sort(key=lambda c: -c[0])
    used_srcs, used_tgts = set(), set()
    moves = []
    for _delta, src, tgt, ships, angle, wait_N in candidates:
        sid = int(src.id)
        tid = int(tgt.id)
        if sid in used_srcs or tid in used_tgts:
            continue
        used_srcs.add(sid)
        used_tgts.add(tid)
        if wait_N == 0:
            moves.append([sid, float(angle), int(ships)])
        # else: wait-N picked → reserve src/tgt, emit nothing this turn.
    return moves
