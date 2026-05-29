"""Per-shot feature encoder for the konbu17-style shot validator.

Pure function `encode_shot_features(emit, obs, focal_seat, *, world=None,
world_model=None) -> ndarray(45,)`.

Lives in `lib/` (not `scripts/`) so:
  - the bundler inlines it into the submission once
  - both `scripts/gen_validator_corpus.py` (training-time) and
    `agents/baseline_validated/main.py` (inference-time) share one
    source of truth for the feature schema

The 45-dim output matches `data/shot_validator/schema.json` v4.

Phase 2 v2 (2026-05-29 — PM5 next-session): expanded from 25-d to 39-d
with 9 new per-shot features (Tier 1 + Tier 2 from the PM4 deep-dive),
plus F3 swap (arrival-time owner replacing launch-time owner at the
existing index slots). Stage 1.5 (2026-05-29 — PI feature audit) added
6 more: enemy fleets inbound to target (2), post-capture nearest-enemy
geometry at arrival time (2), orbital-state flags for source and
target (2). See HANDOVER.md → "Phase 2 v2".

Per-turn `World` + `WorldModel` construction is the caller's
responsibility (~5 ms/turn at 40 planets). When `world` / `world_model`
are omitted, the encoder builds them internally — fine for tests, but
the inference path passes them in to avoid rebuilding per-emit.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Module-level lib imports. Earlier revisions deferred these inside the
# encoder function body to keep cold-path import cost out of tests, but
# `scripts/bundle_validator.py`'s strip-and-rebind logic flattens them
# to col-0, which then IndentationErrors inside the function. Top-level
# imports are also bundler-safe.
from lib.intent import World as _World
from lib.world_model import WorldModel as _WorldModel
from lib.world_model import predict_garrison_at as _pga
from lib.world_model import WAVE_LOOKAHEAD as _WAVE
from lib.trajectory import predict_fleet_fate as _pff
from lib.scoring import pv_horizon as _pvh
from lib.scoring import expected_hold as _eh
from lib.orbit import is_orbiting as _is_orb
from lib.orbit import predict_relative as _pr

FEATURE_DIM = 45

NORM = {
    "max_ships": 2000.0,
    "max_production": 5.0,
    "max_radius": 3.0,
    "max_fleet_speed": 6.0,
    "max_eta": 200.0,
    "board_diagonal": 141.42,
    "max_planets": 40.0,
    "episode_steps": 500.0,
    # F4 pv_capture: γ=0.99 over typical hold horizons gives values
    # ~ production * 100 = up to 5*100 = 500. Set the norm to 500 so
    # well-placed captures saturate near 1.0; mis-placed captures
    # sit well below.
    "max_pv_capture": 500.0,
    # F13 target_growth_field_diff: Σ prod/dist² per side. Typical
    # mid-game balance puts each side's field around 0.5-2.0; the
    # diff is usually within ±2. Set the norm to 5.0 — bigger headroom
    # so close-planet clusters don't clip too easily. Re-calibrate
    # from corpus statistics if F13 saturates often.
    "growth_field_max": 5.0,
}

# Indices into the 39-d vector whose natural range is [-1, +1] rather
# than [0, 1]. Used by tests and downstream sanity checks.
SIGNED_INDICES = (21, 24, 36)


def fleet_speed(ships: float) -> float:
    """Match `lib/fleet.py` and konbu17's per-shot formula."""
    if ships <= 0:
        return 0.0
    return 1.0 + (6.0 - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5


def infer_target_pid(
    src_xy: tuple[float, float],
    angle: float,
    planets: list,
) -> int | None:
    """Project ray from src along `angle`; return planet id with smallest
    perpendicular distance among forward candidates. Matches
    `scripts/label_shot_outcomes._infer_target_pid` and
    `scripts/extended_features._infer_target_pid`."""
    sx, sy = src_xy
    dx, dy = math.cos(angle), math.sin(angle)
    best_id = None
    best_score = float("inf")
    for p in planets:
        pid = int(p[0])
        px, py = float(p[2]), float(p[3])
        if abs(px - sx) < 1e-6 and abs(py - sy) < 1e-6:
            continue
        rx, ry = px - sx, py - sy
        fwd = rx * dx + ry * dy
        if fwd <= 0:
            continue
        perp = math.hypot(rx - fwd * dx, ry - fwd * dy)
        score = perp + 0.001 * fwd
        if score < best_score:
            best_score = score
            best_id = pid
    return best_id


def _build_world_if_needed(obs: Any, world):
    """Build `lib.intent.World` lazily when the caller didn't provide one.
    Inference passes `world` in; tests/back-compat fall through here."""
    if world is not None:
        return world
    return _World.from_obs(obs)


def _build_model_if_needed(world, world_model):
    """Build `lib.world_model.WorldModel` lazily."""
    if world_model is not None:
        return world_model
    return _WorldModel.from_world(world)


def encode_features(
    src_planet: Any,
    target_planet: Any,
    ships_sent: float,
    distance: float,
    eta: float,
    fs: float,
    all_planets: list,
    all_fleets: list,
    focal_seat: int,
    step: int,
    *,
    obs: Any = None,
    world: Any = None,
    world_model: Any = None,
    aim_angle: float | None = None,
) -> list[float]:
    """Build the 39-dim feature vector. Values normalised to [0, 1]
    except SIGNED_INDICES = (21, 24, 36) which sit in [-1, 1].

    Tuple indexing follows the kaggle_environments schema:
      Planet = (id=0, owner=1, x=2, y=3, radius=4, ships=5, production=6)
      Fleet  = (id=0, owner=1, x=2, y=3, angle=4, from_planet_id=5, ships=6)

    Tier-2 features (F3, F6, F4, F8, F10, F11, F7, F13, F9) need a
    pre-built `world` (lib.intent.World) and `world_model`
    (lib.world_model.WorldModel). When not supplied, they're built
    here from `obs` — slow path used by tests only; callers in the
    training + inference paths supply both.
    """
    # ----- basic per-shot features (existing 25 - 3 dropped for F3 swap) -----
    sps_ships = src_planet[5] / NORM["max_ships"]
    sps_prod = src_planet[6] / NORM["max_production"]
    sps_rad = src_planet[4] / NORM["max_radius"]

    tgt_ships = target_planet[5] / NORM["max_ships"]
    tgt_prod = target_planet[6] / NORM["max_production"]
    tgt_rad = target_planet[4] / NORM["max_radius"]

    src_garrison = max(1.0, float(src_planet[5]))
    shot_ships = min(1.0, ships_sent / NORM["max_ships"])
    shot_frac = min(1.0, ships_sent / src_garrison)
    shot_dist = min(1.0, distance / NORM["board_diagonal"])
    shot_eta = min(1.0, eta / NORM["max_eta"])
    shot_fs = min(1.0, fs / NORM["max_fleet_speed"])

    n_allied = 0
    ship_allied = 0.0
    n_enemy = 0
    ship_enemy = 0.0
    for f in all_fleets:
        owner = int(f[1])
        ships = float(f[6])
        if owner == focal_seat:
            n_allied += 1
            ship_allied += ships
        elif owner != -1:
            n_enemy += 1
            ship_enemy += ships
    in_flight_n_allied = min(1.0, n_allied / NORM["max_planets"])
    in_flight_n_enemy = min(1.0, n_enemy / NORM["max_planets"])
    in_flight_ship_allied = min(1.0, ship_allied / NORM["max_ships"])
    in_flight_ship_enemy = min(1.0, ship_enemy / NORM["max_ships"])

    my_total_ships = sum(
        float(p[5]) for p in all_planets if int(p[1]) == focal_seat
    ) + ship_allied
    enemy_total_ships = sum(
        float(p[5]) for p in all_planets
        if int(p[1]) not in (-1, focal_seat)
    ) + ship_enemy
    ship_diff = max(-1.0, min(1.0,
        (my_total_ships - enemy_total_ships) / NORM["max_ships"]))
    my_total_ships_n = min(1.0, my_total_ships / NORM["max_ships"])
    enemy_total_ships_n = min(1.0, enemy_total_ships / NORM["max_ships"])
    meta_turn = step / NORM["episode_steps"]
    my_planet_count = sum(1 for p in all_planets if int(p[1]) == focal_seat)
    enemy_planet_count = sum(
        1 for p in all_planets if int(p[1]) not in (-1, focal_seat)
    )
    my_pc_n = my_planet_count / NORM["max_planets"]
    enemy_pc_n = enemy_planet_count / NORM["max_planets"]

    # F2 combat_margin_at_arrival (PM5): production-walk only, ignores
    # in-flight defenders. Kept as a fast, model-agnostic shape signal;
    # F3 + the timeline-aware features below pick up the in-flight side.
    tgt_owner_launch = int(target_planet[1])
    if tgt_owner_launch != -1:
        pred_garrison_simple = float(target_planet[5]) + float(target_planet[6]) * float(eta)
    else:
        pred_garrison_simple = float(target_planet[5])
    pred_denom_simple = max(1.0, pred_garrison_simple)
    combat_margin = max(-1.0, min(1.0,
        (ships_sent - pred_denom_simple) / pred_denom_simple))

    # ----- Tier 1 + Tier 2 features (Phase 2 v2 additions) -----
    # All Tier-2 features need world + world_model. Build lazily when
    # the caller didn't provide them (tests).
    world = _build_world_if_needed(obs, world)
    world_model = _build_model_if_needed(world, world_model)

    src_pid = int(src_planet[0])
    tgt_pid = int(target_planet[0])

    # F3 owner_at_arrival_one_hot — REPLACES launch-time owner at indices 6-8.
    # Uses predict_garrison_at against the per-target arrival ledger.
    tgt_planet_obj = world.planets_by_id.get(tgt_pid)
    if tgt_planet_obj is not None:
        arrivals_at_tgt = world_model.ledger.get(tgt_pid, [])
        owner_at_arr, pred_garrison_full = _pga(
            tgt_planet_obj, int(round(eta)), arrivals_at_tgt
        )
    else:
        owner_at_arr = tgt_owner_launch
        pred_garrison_full = pred_garrison_simple
    owner_mine = 1.0 if owner_at_arr == focal_seat else 0.0
    owner_neutral = 1.0 if owner_at_arr == -1 else 0.0
    owner_enemy = 1.0 if (owner_at_arr != -1 and owner_at_arr != focal_seat) else 0.0

    # F6 path_fate_one_hot — outcomes from predict_fleet_fate.
    # The four buckets are mutually exclusive; "timeout" maps to all-zero
    # (very rare on a 100x100 board).
    src_planet_obj = world.planets_by_id.get(src_pid)
    fate_target = fate_planet = fate_sun = fate_oob = 0.0
    if src_planet_obj is not None and tgt_planet_obj is not None:
        # F6 must reflect the AGENT'S actual aim. The centre-to-centre
        # recomputation was muddying the signal (Rule 47 substrate trace
        # showed 12.18% waste from naive aim vs ~real-aim ground truth).
        if aim_angle is not None:
            angle_for_fate = float(aim_angle)
        else:
            angle_for_fate = math.atan2(
                float(tgt_planet_obj.y) - float(src_planet_obj.y),
                float(tgt_planet_obj.x) - float(src_planet_obj.x),
            )
        # max_steps cap: cover the trajectory plus a small slack window
        # (target may move under us; +20 covers orbital drift cases).
        max_steps_cap = max(20, int(eta) + 20)
        fate = _pff(
            src_planet_obj, tgt_planet_obj, angle_for_fate,
            int(round(ships_sent)), world, max_steps=max_steps_cap,
        )
        if fate.outcome == "target":
            fate_target = 1.0
        elif fate.outcome == "planet":
            fate_planet = 1.0
        elif fate.outcome == "sun":
            fate_sun = 1.0
        elif fate.outcome == "oob":
            fate_oob = 1.0
        # "timeout" → all four zero

    # F10 same_target_friendly_inflight {count, ships}.
    arrivals_at_tgt = world_model.ledger.get(tgt_pid, [])
    friendly_at_tgt = [
        (e_, s_) for (e_, o_, s_) in arrivals_at_tgt if o_ == focal_seat
    ]
    friendly_inflight_n = min(1.0,
        len(friendly_at_tgt) / NORM["max_planets"])
    friendly_inflight_ships = min(1.0,
        sum(s_ for _, s_ in friendly_at_tgt) / NORM["max_ships"])

    # F8 src_safe_departure_ratio + shot_drains_safely.
    # safe_dep = src.ships + prod * enemy_eta - inbound_enemy_ships - 1
    # ratio = min(1, safe_dep / ships_sent); binary version on the side.
    enemy_eta_at_src = world_model.incoming_enemy_eta(src_pid, focal_seat)
    if enemy_eta_at_src is None:
        # No inbound enemy — source is fully safe. Use a generous horizon.
        horizon_for_safe = _WAVE
        inbound_enemy_ships = 0.0
    else:
        horizon_for_safe = enemy_eta_at_src
        arrivals_at_src = world_model.ledger.get(src_pid, [])
        inbound_enemy_ships = sum(
            s_ for (e_, o_, s_) in arrivals_at_src
            if o_ != focal_seat and e_ <= enemy_eta_at_src
        )
    src_prod = float(src_planet[6])
    src_ships_raw = float(src_planet[5])
    safe_dep_raw = (
        src_ships_raw + src_prod * float(horizon_for_safe)
        - inbound_enemy_ships - 1.0
    )
    src_safe_dep_ratio = max(0.0, min(1.0,
        safe_dep_raw / max(1.0, float(ships_sent))))
    shot_drains_safely = 1.0 if safe_dep_raw >= float(ships_sent) else 0.0

    # F4 pv_capture: γ=0.99 over expected_hold-truncated horizon × target.production.
    if tgt_planet_obj is not None:
        hold = _eh(tgt_pid, int(round(eta)), world, world_model)
        t_total_for_pv = int(world.step) + int(round(eta)) + int(hold)
        pv_raw = _pvh(
            int(world.step), int(round(eta)),
            gamma=0.99, t_total=t_total_for_pv,
        ) * float(tgt_planet_obj.production)
    else:
        pv_raw = 0.0
    pv_capture = min(1.0, pv_raw / NORM["max_pv_capture"])

    # F11 joint_arrival_count_at_eta — ±1-step same-owner stack count.
    eta_int = int(round(eta))
    joint_arr_n = sum(
        1 for (e_, o_, s_) in arrivals_at_tgt
        if o_ == focal_seat and abs(e_ - eta_int) <= 1
    )
    joint_arrival_count = min(1.0, joint_arr_n / NORM["max_planets"])

    # F7 intercept_enemy_eta — earliest enemy arrival at target.
    # Saturates at 1.0 when no inbound; smaller when enemy is close.
    intercept_eta = world_model.incoming_enemy_eta_after(
        tgt_pid, focal_seat, after=0
    )
    if intercept_eta is None:
        intercept_norm = 1.0
    else:
        intercept_norm = min(1.0, intercept_eta / NORM["max_eta"])

    # F13 target_growth_field_diff — zvold's electrostatic field.
    # Σ prod/dist² over my planets minus enemy planets, normalised.
    # Substrate not in `lib/` — inline impl.
    if tgt_planet_obj is not None:
        tx = float(tgt_planet_obj.x)
        ty = float(tgt_planet_obj.y)
        field_mine = 0.0
        field_enemy = 0.0
        for p_obj in world.planets_by_id.values():
            if p_obj.id == tgt_pid:
                continue
            dx = float(p_obj.x) - tx
            dy = float(p_obj.y) - ty
            d_sq = max(0.5, dx * dx + dy * dy)
            contrib = float(p_obj.production) / d_sq
            if p_obj.owner == focal_seat:
                field_mine += contrib
            elif p_obj.owner != -1:
                field_enemy += contrib
        growth_field_diff = max(-1.0, min(1.0,
            (field_mine - field_enemy) / NORM["growth_field_max"]))
    else:
        growth_field_diff = 0.0

    # F9 src_time_to_nearest_enemy_threat + src_is_frontier.
    # threat == None means saturate (no enemy can plausibly reach src).
    src_threat = world_model.time_to_enemy_threat(
        src_pid, focal_seat, world,
    ) if src_planet_obj is not None else None
    if src_threat is None:
        src_threat_norm = 1.0
        src_is_frontier = 0.0
    else:
        src_threat_norm = min(1.0, src_threat / NORM["max_eta"])
        src_is_frontier = 1.0 if src_threat < 25 else 0.0

    # ----- Stage 1.5 additions (PI: post-capture geometry, enemy-inbound,
    # orbital state). See data/shot_validator/schema.json v4 for index map.

    # Enemy fleets inbound to target — mirror of the friendly side.
    # Uses the same per-target arrival_ledger slice friendly_at_tgt was
    # carved out of (`arrivals_at_tgt`).
    enemy_at_tgt = [
        (e_, s_) for (e_, o_, s_) in arrivals_at_tgt
        if o_ != focal_seat and o_ != -1
    ]
    enemy_inflight_n = min(1.0,
        len(enemy_at_tgt) / NORM["max_planets"])
    enemy_inflight_ships = min(1.0,
        sum(s_ for _, s_ in enemy_at_tgt) / NORM["max_ships"])

    # Post-capture geometry — where is the nearest enemy planet at
    # step + eta + LABEL_BUFFER, relative to where the target will be?
    # Captures the "is this an exposed capture or a safe one" question.
    # Uses predict_relative for orbiting planets (fine for orbital;
    # comets fall back to launch-time position which is approximate but
    # consistent).
    omega = float(getattr(world, "omega", 0.0) or 0.0)
    POST_CAPTURE_LEAD = int(round(eta)) + 10  # = LABEL_BUFFER

    if tgt_planet_obj is not None:
        tgt_tuple_pos = [
            tgt_pid, int(tgt_planet_obj.owner),
            float(tgt_planet_obj.x), float(tgt_planet_obj.y),
            float(tgt_planet_obj.radius), float(tgt_planet_obj.ships),
            float(tgt_planet_obj.production),
        ]
        if _is_orb(tgt_tuple_pos) and omega != 0.0:
            tx_arr, ty_arr = _pr(tgt_tuple_pos, omega, POST_CAPTURE_LEAD)
        else:
            tx_arr, ty_arr = float(tgt_planet_obj.x), float(tgt_planet_obj.y)
    else:
        tx_arr = float(target_planet[2])
        ty_arr = float(target_planet[3])

    nearest_enemy_dist_raw = float("inf")
    nearest_enemy_prod_raw = 0.0
    for p_obj in world.planets_by_id.values():
        if int(p_obj.id) == tgt_pid:
            continue
        if int(p_obj.owner) == focal_seat or int(p_obj.owner) == -1:
            continue
        p_tuple_pos = [
            int(p_obj.id), int(p_obj.owner),
            float(p_obj.x), float(p_obj.y),
            float(p_obj.radius), float(p_obj.ships),
            float(p_obj.production),
        ]
        if _is_orb(p_tuple_pos) and omega != 0.0:
            ex, ey = _pr(p_tuple_pos, omega, POST_CAPTURE_LEAD)
        else:
            ex, ey = float(p_obj.x), float(p_obj.y)
        dist = math.hypot(ex - tx_arr, ey - ty_arr)
        if dist < nearest_enemy_dist_raw:
            nearest_enemy_dist_raw = dist
            nearest_enemy_prod_raw = float(p_obj.production)
    if nearest_enemy_dist_raw == float("inf"):
        post_capture_nearest_dist = 1.0  # no enemy -> max distance
        post_capture_nearest_prod = 0.0
    else:
        post_capture_nearest_dist = min(1.0,
            nearest_enemy_dist_raw / NORM["board_diagonal"])
        post_capture_nearest_prod = min(1.0,
            nearest_enemy_prod_raw / NORM["max_production"])

    # Orbital state — binary flags. Lets the model distinguish static-
    # geometry shots from orbital ones (different aim/timing dynamics).
    src_tuple_pos = [
        int(src_planet[0]), int(src_planet[1]),
        float(src_planet[2]), float(src_planet[3]),
        float(src_planet[4]), float(src_planet[5]),
        float(src_planet[6]),
    ]
    tgt_tuple_pos_basic = [
        int(target_planet[0]), int(target_planet[1]),
        float(target_planet[2]), float(target_planet[3]),
        float(target_planet[4]), float(target_planet[5]),
        float(target_planet[6]),
    ]
    src_is_orbiting_f = 1.0 if _is_orb(src_tuple_pos) else 0.0
    tgt_is_orbiting_f = 1.0 if _is_orb(tgt_tuple_pos_basic) else 0.0

    return [
        # 0-5: planet-static features
        sps_ships, sps_prod, sps_rad,
        tgt_ships, tgt_prod, tgt_rad,
        # 6-8: F3 arrival-time owner one-hot (Phase 2 v2 swap)
        owner_mine, owner_neutral, owner_enemy,
        # 9-13: shot-static features
        shot_ships, shot_frac, shot_dist, shot_eta, shot_fs,
        # 14-17: in-flight totals
        in_flight_n_allied, in_flight_ship_allied,
        in_flight_n_enemy, in_flight_ship_enemy,
        # 18-23: meta
        meta_turn, my_total_ships_n, enemy_total_ships_n,
        ship_diff, my_pc_n, enemy_pc_n,
        # 24: F2 combat margin (PM5)
        combat_margin,
        # 25-28: F6 path-fate one-hot
        fate_target, fate_planet, fate_sun, fate_oob,
        # 29-30: F10 friendly inflight at target
        friendly_inflight_n, friendly_inflight_ships,
        # 31-32: F8 source-drain safety
        src_safe_dep_ratio, shot_drains_safely,
        # 33: F4 pv_capture
        pv_capture,
        # 34: F11 joint arrival count at eta
        joint_arrival_count,
        # 35: F7 intercept enemy eta (norm)
        intercept_norm,
        # 36: F13 growth-field diff (signed)
        growth_field_diff,
        # 37-38: F9 source-side threat features
        src_threat_norm, src_is_frontier,
        # 39-40: enemy fleets inbound to target (Stage 1.5)
        enemy_inflight_n, enemy_inflight_ships,
        # 41-42: post-capture geometry — nearest enemy at arrival (Stage 1.5)
        post_capture_nearest_dist, post_capture_nearest_prod,
        # 43-44: orbital state flags (Stage 1.5)
        src_is_orbiting_f, tgt_is_orbiting_f,
    ]


def encode_shot_features(
    emit: list,
    obs: Any,
    focal_seat: int,
    *,
    world: Any = None,
    world_model: Any = None,
) -> np.ndarray | None:
    """Inference-time wrapper. Returns None if the emit is malformed or
    cannot be associated with a target planet via ray-cast.

    `emit` = [src_pid, angle, ships]
    `obs` exposes `.planets`, `.fleets`, `.step` (dict or Struct).

    Optional `world` and `world_model`: pass pre-built per-turn instances
    to avoid the ~5 ms per-call build. When omitted, both are built
    here from `obs` (slow path; fine for tests).
    """
    if not emit or len(emit) < 3:
        return None
    try:
        src_pid = int(emit[0])
        angle = float(emit[1])
        ships = float(emit[2])
    except (TypeError, ValueError):
        return None

    planets = list(obs.get("planets", []) if isinstance(obs, dict)
                   else getattr(obs, "planets", []) or [])
    fleets = list(obs.get("fleets", []) if isinstance(obs, dict)
                  else getattr(obs, "fleets", []) or [])
    step = int(obs.get("step", 0) if isinstance(obs, dict)
               else getattr(obs, "step", 0) or 0)

    by_id = {int(p[0]): p for p in planets}
    src = by_id.get(src_pid)
    if src is None:
        return None
    target_pid = infer_target_pid(
        (float(src[2]), float(src[3])), angle, planets
    )
    if target_pid is None:
        return None
    target = by_id.get(target_pid)
    if target is None:
        return None

    d = math.hypot(float(target[2]) - float(src[2]),
                   float(target[3]) - float(src[3]))
    v = fleet_speed(ships)
    eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0

    feats = encode_features(
        src, target, ships, d, eta, v,
        planets, fleets, focal_seat, step,
        obs=obs, world=world, world_model=world_model,
        aim_angle=angle,
    )
    return np.asarray(feats, dtype=np.float32)


def target_owned_by(emit: list, obs: Any, focal_seat: int) -> bool:
    """Self-reinforcement check: is the emit's ray-cast target already
    owned by `focal_seat`? Used by the validator agent to bypass filtering
    on self-reinforce shots (konbu17 design — these are never filtered)."""
    if not emit or len(emit) < 3:
        return False
    try:
        src_pid = int(emit[0])
        angle = float(emit[1])
    except (TypeError, ValueError):
        return False
    planets = list(obs.get("planets", []) if isinstance(obs, dict)
                   else getattr(obs, "planets", []) or [])
    by_id = {int(p[0]): p for p in planets}
    src = by_id.get(src_pid)
    if src is None:
        return False
    target_pid = infer_target_pid(
        (float(src[2]), float(src[3])), angle, planets
    )
    if target_pid is None:
        return False
    target = by_id.get(target_pid)
    if target is None:
        return False
    return int(target[1]) == focal_seat
