"""v8_scavenge — analytic event-horizon chooser ("macro moves").

Approach (PI direction, 2026-05-16):
1. Score each candidate launch by its END-STATE marginal value vs idle.
   The BASELINE `WorldModel` (built once per turn from current obs)
   predicts per-planet ownership + garrison at any future step. For a
   candidate (src, tgt, ships), our fleet's eta is deterministic; the
   model tells us who would own tgt at that eta WITHOUT our fleet, and
   how many ships would defend.
2. If baseline predicts tgt would be ours → reinforcement, zero credit.
   If our ships beat the predicted defenders → we'd capture; credit by
   `production × time_remaining` (production lead until end-of-game),
   double for enemy captures (we gain prod, they lose prod), single
   for neutrals. Subtract combat ship-cost.
   If we'd bounce → all our ships lost; pure cost.
3. No turn-by-turn fast_sim rollout. The WorldModel is analytic and
   captures the "macro view": who owns what at the terminal horizon.
4. Opponent idles in our model — we don't speculate about their next
   action; we delegate that to next turn's chooser.
5. Greedy non-dogpile emit (Phase 1; Phase 3 will use `settle_plan`):
   max one launch per source, max one per target per turn.

Phase 1 (this file) — basic enumeration only:
  for src in my_planets:
    for tgt in nearest-K non-owned:
      for ships in {capture_size, 2x, full_budget}:
        score and pick argmax via marginal_value()

Phase 2 will append scavenge ship-counts (sizes timed to arrive at
predicted enemy-capture eta + small delta — captures the planet just
after the enemy takes it, beating the depleted post-capture garrison).
That mechanic emerges naturally because WorldModel sees the enemy
capture in its timeline; our marginal_value reads the post-capture
predicted garrison and scores accordingly.
"""

from __future__ import annotations

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.aim import aim_orbiting
from lib.fleet import speed as fleet_speed
from lib.intent import World
from lib.orbit import is_orbiting as _is_orbiting
from lib.world_model import WorldModel

# ---------------------------------------------------------------------------
# Tunable knobs (defaults chosen to keep the structural choice in focus;
# tuning is a separate phase if Phase 1 lifts).
# ---------------------------------------------------------------------------

NUM_TARGETS_PER_SOURCE = 8       # K nearest non-owned planets per source
MIN_FLEET_SIZE = 2               # 1-ship fleets are slow + rarely useful

EPISODE_STEPS = 500
# Match lib.value_heads.composite_capture_value's tuned weights
# (v7_4_capture_value used these to credit predicted captures and
# penalise bounces). Calibration: 0.05 × prod × 500 ≈ 25-75 per
# high-value capture; 0.5 × ships ≈ 5-50 per bounce. Same scale.
CAPTURE_WEIGHT = 0.05
WASTE_WEIGHT = 0.5


# ---------------------------------------------------------------------------
# Obs helpers
# ---------------------------------------------------------------------------


def _as_dict(obs):
    """Coerce an obs (Struct or dict) into a dict.

    Kaggle passes a Struct in production; tests sometimes pass a dict.
    Both work with the foundation primitives but consistent access via
    a dict makes downstream helpers simpler.
    """
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


# ---------------------------------------------------------------------------
# Geometry / timing primitives
# ---------------------------------------------------------------------------


def _aim_angle(src, tgt, ships, omega):
    """Lead-aim for orbiting targets; straight-aim for static.

    Wraps `lib.aim.aim_orbiting` (5-iter fixed-point + safe-intercept
    fallback). Returns the angle that the simulator will actually fly.
    """
    if _is_orbiting(list(tgt)):
        res = aim_orbiting(
            (src.x, src.y), src.radius, list(tgt), tgt.radius, ships, omega,
        )
        if res is not None:
            return float(res[0])
    return math.atan2(tgt.y - src.y, tgt.x - src.x)


def _arrival_eta(src, tgt, ships):
    """Integer-turn arrival eta. Matches `lib.aim.flight_distance`'s
    accounting (centre distance minus radii minus 0.1 spawn offset).
    """
    flight = max(
        0.0,
        math.hypot(src.x - tgt.x, src.y - tgt.y)
        - src.radius - tgt.radius - 0.1,
    )
    spd = fleet_speed(ships)
    if spd <= 0:
        return 999
    return int(math.ceil(flight / spd))


def _nearest_k(targets, src, k):
    return sorted(
        targets,
        key=lambda t: math.hypot(src.x - t.x, src.y - t.y),
    )[:k]


# ---------------------------------------------------------------------------
# Capture-size + ship-count enumeration
# ---------------------------------------------------------------------------


def _capture_size(src, tgt, model):
    """WorldModel-aware minimum capture size.

    One Newton-style iteration: pick an initial size from current tgt
    garrison, compute eta, query model for predicted garrison at THAT
    eta, derive final size from prediction.

    This already incorporates production growth + incoming reinforcement
    in the predicted defender count — the WorldModel timeline accounts
    for them analytically.
    """
    initial = max(MIN_FLEET_SIZE, int(tgt.ships) + 1)
    eta = _arrival_eta(src, tgt, initial)
    pred = float(model.ships_at(int(tgt.id), eta) or 0.0)
    size = int(math.ceil(pred)) + 1
    return max(MIN_FLEET_SIZE, size)


def _enumerate_ship_counts_basic(src, tgt, model):
    """Phase 1 ship-count set: capture, 2×capture, full launch budget.

    Scavenge sizes (Phase 2) — ship counts timed to arrive at predicted
    enemy-capture eta + δ — appended in a later phase.
    """
    cap = _capture_size(src, tgt, model)
    budget = int(src.ships)
    sizes = set()
    if MIN_FLEET_SIZE <= cap <= budget:
        sizes.add(cap)
    if 2 * cap <= budget:
        sizes.add(2 * cap)
    if budget >= MIN_FLEET_SIZE and budget > cap:
        sizes.add(budget)
    return sorted(sizes)


# ---------------------------------------------------------------------------
# Marginal end-state value
# ---------------------------------------------------------------------------


def _marginal_value(src, tgt, ships, eta, world, model, my_id):
    """End-state Δ value from launching `ships` from src to tgt.

    Reads the BASELINE WorldModel timeline (built once per turn, before
    any candidate is considered) for the predicted state at our arrival.
    Critically: this is NOT the model with our hypothetical fleet added.
    The "marginal" semantics require comparing "future with my fleet"
    vs "future without my fleet" — using the baseline model gives us
    the latter directly.

    Scoring:
    - pred_owner == my_id → planet would already be ours; no credit.
    - ships > pred_defenders at eta → CAPTURE. Credit by
        capture_weight × prod_factor × tgt.production × time_remaining
      minus combat ship-cost (= pred_defenders ships lost from our
      fleet). prod_factor = 2 for enemy captures (we gain prod, they
      lose prod = double swing on production lead), 1 for neutrals.
    - ships ≤ pred_defenders → BOUNCE. All ships lost.

    Note: Phase 1 ignores hold-time uncertainty (treats capture credit
    as if held to end-game). CAPTURE_WEIGHT=0.5 builds in a coarse
    hold-prob discount. Phase 2+ may refine with model.owner_at lookups
    at eta+δ to penalise quick recaptures.
    """
    pred_owner = model.owner_at(int(tgt.id), eta)
    pred_ships = float(model.ships_at(int(tgt.id), eta) or 0.0)

    if pred_owner == my_id:
        return 0.0  # already ours; my fleet is reinforcement only

    time_remaining = max(0, EPISODE_STEPS - int(world.step) - eta)
    production = float(tgt.production)

    if ships > pred_ships:
        # CAPTURE — credit production × time_remaining (matches
        # composite_capture_value's formula exactly; the combat
        # ship-cost is implicit in the constants and not separately
        # accounted, by design).
        production_gain = production * float(time_remaining)
        return CAPTURE_WEIGHT * production_gain

    # BOUNCE — fleet won't capture; penalise wasted ships.
    return -WASTE_WEIGHT * float(ships)


# ---------------------------------------------------------------------------
# Public agent
# ---------------------------------------------------------------------------


def agent(obs, configuration=None):
    obs_d = _as_dict(obs)
    me = int(obs_d.get("player", 0))
    raw_planets = obs_d.get("planets", []) or []
    if not raw_planets:
        return []

    planets = [Planet(*p) for p in raw_planets]
    my_planets = [p for p in planets if int(p.owner) == me]
    targets = [p for p in planets if int(p.owner) != me]
    if not my_planets or not targets:
        return []

    world = World.from_obs(obs_d)
    model = WorldModel.from_world(world)
    omega = float(obs_d.get("angular_velocity", 0.0))

    # Enumerate + score candidates
    candidates = []  # list of (delta, src, tgt, ships, angle)
    for src in my_planets:
        if int(src.ships) < MIN_FLEET_SIZE:
            continue
        for tgt in _nearest_k(targets, src, NUM_TARGETS_PER_SOURCE):
            for ships in _enumerate_ship_counts_basic(src, tgt, model):
                if ships < MIN_FLEET_SIZE or ships > int(src.ships):
                    continue
                angle = _aim_angle(src, tgt, ships, omega)
                eta = _arrival_eta(src, tgt, ships)
                delta = _marginal_value(src, tgt, ships, eta, world, model, me)
                if delta > 0:
                    candidates.append((delta, src, tgt, ships, angle))

    if not candidates:
        return []

    # Greedy non-dogpile: max 1 launch per source / per target per turn.
    # Multi-launch-per-source (per-target dedup with budget) was tried
    # and slightly regressed in n=32 vs-nearest A/B; reverted. Phase 3
    # will use settle_plan, which has its own arrival ledger.
    candidates.sort(key=lambda c: -c[0])
    used_srcs, used_tgts = set(), set()
    moves = []
    for _delta, src, tgt, ships, angle in candidates:
        sid = int(src.id)
        tid = int(tgt.id)
        if sid in used_srcs or tid in used_tgts:
            continue
        used_srcs.add(sid)
        used_tgts.add(tid)
        moves.append([sid, float(angle), int(ships)])
    return moves
