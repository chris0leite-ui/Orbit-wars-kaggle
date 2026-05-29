"""Per-shot feature encoder for the konbu17-style shot validator MLP.

Pure function `encode_shot_features(emit, obs, focal_seat) -> ndarray(25,)`.
Lives in `lib/` (not `scripts/`) so:

  - the bundler inlines it into the submission once
  - both `scripts/gen_validator_corpus.py` (training-time) and
    `agents/baseline_validated/main.py` (inference-time) share one
    source of truth for the feature schema

The 25-dim output matches `data/shot_validator/schema.json` exactly.
See `knowledge-base/thoughts/2026-05-28-pm3-h14-recipe-locked-from-konbu17.md`
for the feature definitions verified against konbu17's notebook cells 8 + 16.

Feature 24 (`combat_margin_at_arrival`) was added in Stage 2 of PM5
(2026-05-28): the model-readable form of the binary label. Production-walk
prediction of the target garrison at ETA (owner's production accrues each
tick if the planet is owned); margin = (ships_sent - pred) / max(1, pred),
clamped to [-1, 1]. This approximation ignores in-flight defending fleets;
F3 (`enemy_defenders_in_range`) will cover that orthogonal signal later.

Normalisation constants are the same `_NORM` dict used by
`scripts/label_shot_outcomes.py` — both modules import from here now.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

FEATURE_DIM = 25

NORM = {
    "max_ships": 2000.0,
    "max_production": 5.0,
    "max_radius": 3.0,
    "max_fleet_speed": 6.0,
    "max_eta": 200.0,
    "board_diagonal": 141.42,
    "max_planets": 40.0,
    "episode_steps": 500.0,
}


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
) -> list[float]:
    """Build the 25-dim feature vector. All values normalised to [0, 1]
    except `ship_diff` (index 21) and `combat_margin` (index 24), both
    in [-1, 1].

    Tuple indexing follows the kaggle_environments schema:
      Planet = (id=0, owner=1, x=2, y=3, radius=4, ships=5, production=6)
      Fleet  = (id=0, owner=1, x=2, y=3, angle=4, from_planet_id=5, ships=6)
    """
    sps_ships = src_planet[5] / NORM["max_ships"]
    sps_prod = src_planet[6] / NORM["max_production"]
    sps_rad = src_planet[4] / NORM["max_radius"]

    tgt_ships = target_planet[5] / NORM["max_ships"]
    tgt_prod = target_planet[6] / NORM["max_production"]
    tgt_rad = target_planet[4] / NORM["max_radius"]

    tgt_owner = int(target_planet[1])
    owner_mine = 1.0 if tgt_owner == focal_seat else 0.0
    owner_neutral = 1.0 if tgt_owner == -1 else 0.0
    owner_enemy = 1.0 if (tgt_owner != -1 and tgt_owner != focal_seat) else 0.0

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

    # F2 combat_margin_at_arrival: production-walk prediction of the
    # target's garrison at ETA, then signed margin of ships_sent against
    # it. Owned planets accrue `production` per tick; neutrals don't.
    # Ignores in-flight fleets (F3 covers that orthogonal signal). The
    # raw target tuple's owner / ships / production carry the unnormalised
    # values we need; encoder receives raw tuples by design.
    if tgt_owner != -1:
        pred_garrison = float(target_planet[5]) + float(target_planet[6]) * float(eta)
    else:
        pred_garrison = float(target_planet[5])
    pred_denom = max(1.0, pred_garrison)
    combat_margin = max(-1.0, min(1.0, (ships_sent - pred_denom) / pred_denom))

    return [
        sps_ships, sps_prod, sps_rad,
        tgt_ships, tgt_prod, tgt_rad,
        owner_mine, owner_neutral, owner_enemy,
        shot_ships, shot_frac, shot_dist, shot_eta, shot_fs,
        in_flight_n_allied, in_flight_ship_allied,
        in_flight_n_enemy, in_flight_ship_enemy,
        meta_turn, my_total_ships_n, enemy_total_ships_n,
        ship_diff, my_pc_n, enemy_pc_n,
        combat_margin,
    ]


def encode_shot_features(
    emit: list,
    obs: Any,
    focal_seat: int,
) -> np.ndarray | None:
    """Inference-time wrapper. Returns None if the emit is malformed or
    cannot be associated with a target planet via ray-cast.

    `emit` = [src_pid, angle, ships]
    `obs` exposes `.planets`, `.fleets`, `.step` (dict or Struct).
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
