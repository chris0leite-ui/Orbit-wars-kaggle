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
from lib.mission import Mission
from lib.orbit import is_orbiting as _is_orbiting
from lib.planner import settle_plan
from lib.world_model import WorldModel

# ---------------------------------------------------------------------------
# Tunable knobs
# ---------------------------------------------------------------------------

EPISODE_STEPS = 500
NUM_TARGETS_PER_SOURCE = 8       # K nearest non-owned planets per source
MIN_FLEET_SIZE = 2               # 1-ship fleets are slow + rarely useful

# Forward-sim horizon parameters
SIM_SETTLE_TURNS = 2             # extra idle turns after arrival to settle combat
MIN_HORIZON = 15                 # floor — must cover incoming threats arriving
                                 # at our source planets in ~time for fast fleet
MAX_HORIZON = 50                 # baseline cache depth (long enough for any
                                 # plausible candidate eta + settle)

# Wallclock safety. The env's actTimeout is 1000ms. Panel calibration
# at n=192 observed p95=812ms, max=3116ms — outliers blow past the
# budget when the candidate pool is large (many sources × targets ×
# ship-counts × long-K rollouts at mid-game with many in-flight fleets).
# Tightened from 750→600ms AND deadline-check moved inside the ship-
# count inner loop so a single expensive rollout can't push past the
# budget.
WALLCLOCK_BUDGET_MS = 600.0


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


def _capture_size(src, tgt, model, omega, me):
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
        # Reinforce: size to make defense survive predicted enemy arrival
        enemy_eta = model.incoming_enemy_eta(int(tgt.id), me)
        if enemy_eta is None:
            return 0  # no incoming threat; reinforce unnecessary
        # Predicted enemy ships arriving at tgt at enemy_eta (sum across
        # in-flight enemy fleets aimed at tgt within that window).
        enemy_arrivals = model.ledger.get(int(tgt.id), [])
        enemy_ship_sum = sum(
            ships for (eta, owner, ships) in enemy_arrivals
            if owner != me and eta <= enemy_eta + 1
        )
        # Predicted defender at enemy_eta (with production accrual
        # but BEFORE enemy combat applied). Approximation: current
        # garrison + production × enemy_eta.
        my_garrison_at_eta = float(tgt.ships) + float(tgt.production) * enemy_eta
        shortfall = enemy_ship_sum - my_garrison_at_eta + 1
        return max(0, int(math.ceil(shortfall)))
    # Capture (non-mine target)
    initial = max(MIN_FLEET_SIZE, int(tgt.ships) + 1)
    _angle, eta = _aim_and_eta(src, tgt, initial, omega)
    pred = float(model.ships_at(int(tgt.id), eta) or 0.0)
    size = int(math.ceil(pred)) + 1
    return max(MIN_FLEET_SIZE, size)


def _enumerate_ship_counts_basic(src, tgt, model, omega, me):
    """Phase 1 ship-count set: capture/reinforce size, 2×, full budget.

    For reinforce (my own target), size 0 means no threat → skip.
    """
    cap = _capture_size(src, tgt, model, omega, me)
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


# ---------------------------------------------------------------------------
# Favor (F1 + F2) — bootstrap's proven leaf scorer
# ---------------------------------------------------------------------------


def _favor(obs, me):
    """F1 + F2 favor.

    F1 = (my ships on planets + in-flight) − (max-opp ships on planets
         + in-flight). For 2P this is just (my − opp); for 4P it's
         strongest-opp.
    F2 = (my production − max-opp production) × turns_remaining.

    Bootstrap session validated AUC ≈ 0.945 on saved snapshots.
    No comet-decay term here — fast_sim handles comet lifetime exactly
    in the rollout, so by the leaf the comet's ownership is "real."
    """
    planets = obs.planets if hasattr(obs, "planets") else obs.get("planets", [])
    fleets = obs.fleets if hasattr(obs, "fleets") else obs.get("fleets", [])
    step = obs.step if hasattr(obs, "step") else obs.get("step", 0)
    turns_remaining = max(0, EPISODE_STEPS - int(step))

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
    opp_ships_max = max(
        (v for k, v in ships_by_owner.items() if k != me),
        default=0.0,
    )
    opp_prod_max = max(
        (v for k, v in prod_by_owner.items() if k != me),
        default=0.0,
    )

    return (my_ships - opp_ships_max) + (my_prod - opp_prod_max) * turns_remaining


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
    out = [_favor(snap.state[me].observation, me)]
    idle = [[] for _ in range(num_seats)]
    for _ in range(max_horizon):
        if snap.fake_env.done:
            out.append(out[-1])
            continue
        snap = fs_step(snap, idle, in_place=True)
        out.append(_favor(snap.state[me].observation, me))
    return out


def _score_action(snap_base, me, num_seats, src_id, angle, ships,
                  horizon, baseline_favors):
    """Δ favor at horizon = (leaf with my action + idle rest) − idle baseline.

    Step 0: apply my candidate launch (rest idle).
    Steps 1..horizon-1: idle.
    Return favor at the leaf − baseline_favors[horizon].

    The simulator handles orbital motion, swept-pair collisions, sun
    avoidance, and combat resolution EXACTLY (parity-tested vs the env).
    So the leaf state correctly reflects whether my fleet actually
    captures the intended target.
    """
    snap = fs_clone(snap_base)
    actions = [[] for _ in range(num_seats)]
    actions[me] = [[int(src_id), float(angle), int(ships)]]
    snap = fs_step(snap, actions, in_place=True)

    idle = [[] for _ in range(num_seats)]
    for _ in range(horizon - 1):
        if snap.fake_env.done:
            break
        snap = fs_step(snap, idle, in_place=True)

    leaf_favor = _favor(snap.state[me].observation, me)
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

    # Identify threatened MY planets (predicted incoming enemy fleet).
    # Reinforce candidates target these; defensive fleets keep them
    # alive through enemy waves. Mine session diag (2026-05-16): every
    # loss vs v7_0 was "0 planets left, eliminated mid-game" — opp's
    # multi-wave attacks wipe undefended captures. Reinforce closes that.
    threatened_mine = [
        p for p in my_planets
        if model.incoming_enemy_eta(int(p.id), me) is not None
    ]
    # Target pool = capture targets + defensive reinforce targets
    target_pool = other_planets + threatened_mine

    # Build the fast_sim snapshot once per turn (~1 ms).
    snap_base = fs_from_obs(obs, num_seats=num_seats)

    # Idle baseline at horizons 0..MAX_HORIZON (~6 ms for 50 steps).
    baseline_favors = _build_idle_baseline(snap_base, me, num_seats, MAX_HORIZON)

    # Enumerate + score candidates via fast_sim rollout.
    # Deadline checked at THREE levels (source / target / ship-count)
    # so no single expensive rollout pushes us past the budget. Panel
    # n=192 had max=3116ms with only source/target guards; checking
    # inside the ship-count loop bounds the outlier.
    t_deadline = time.perf_counter() + WALLCLOCK_BUDGET_MS / 1000.0
    candidates = []
    bailed = False
    for src in my_planets:
        if bailed or time.perf_counter() > t_deadline:
            bailed = True
            break
        if int(src.ships) < MIN_FLEET_SIZE:
            continue
        for tgt in _nearest_k(target_pool, src, NUM_TARGETS_PER_SOURCE):
            if time.perf_counter() > t_deadline:
                bailed = True
                break
            # Skip self-reinforce (a planet can't reinforce itself).
            if int(tgt.id) == int(src.id):
                continue
            for ships in _enumerate_ship_counts_basic(src, tgt, model, omega, me):
                if time.perf_counter() > t_deadline:
                    bailed = True
                    break
                if ships < MIN_FLEET_SIZE or ships > int(src.ships):
                    continue
                angle, eta = _aim_and_eta(src, tgt, ships, omega)
                horizon = max(eta + SIM_SETTLE_TURNS, MIN_HORIZON)
                if horizon >= len(baseline_favors):
                    horizon = len(baseline_favors) - 1
                delta = _score_action(
                    snap_base, me, num_seats,
                    int(src.id), angle, ships,
                    horizon, baseline_favors,
                )
                if delta > 0:
                    candidates.append((delta, src, tgt, ships, angle, eta))

    if not candidates:
        return []

    # Emission via lib.planner.settle_plan: per-source greedy with a
    # same-turn arrival ledger. Two behavioural improvements over the
    # old "one per src / one per tgt" greedy:
    # 1. Gang-up enabled: when source A's contribution to target X is
    #    insufficient, source B's mission for X is ACCEPTED (the
    #    ledger sees A's pending ships are below the defender threshold).
    # 2. No surplus: if A's contribution suffices, B's mission for X
    #    is skipped — same as the old greedy, expressed via ledger.
    # See lib/planner.py module docstring for the full invariants.
    my_id_set = {int(p.id) for p in my_planets}
    missions = []
    angle_by_key = {}  # (src_id, target_id, ships) → precomputed lead angle
    for _delta, src, tgt, ships, angle, eta in candidates:
        sid = int(src.id)
        tid = int(tgt.id)
        ships_i = int(ships)
        mclass = "reinforce" if tid in my_id_set else "snipe"
        missions.append(Mission(
            mission_class=mclass,
            src_id=sid,
            target_id=tid,
            ships=ships_i,
            score=float(_delta),
            eta=int(eta),
        ))
        angle_by_key[(sid, tid, ships_i)] = float(angle)

    intents = settle_plan(missions, world, model)
    moves = []
    for i in intents:
        key = (int(i.src_id), int(i.target_id), int(i.ships))
        angle = angle_by_key.get(key)
        if angle is None:
            continue  # shouldn't happen — every mission was built with an angle
        moves.append([int(i.src_id), float(angle), int(i.ships)])
    return moves
