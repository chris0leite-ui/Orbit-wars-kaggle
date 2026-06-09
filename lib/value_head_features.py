"""Per-candidate feature encoder for the Reframe B.2 value head.

The encoder produces a 14-d feature vector per candidate decision (src,
tgt, ships, angle, wait_N, eta). The chooser augments the vector with
the candidate's leaf_delta (the pv_eta-discounted Δ from
`score_candidate_v4`) as the 15th column at predict time.

Feature schema (FEATURE_DIM_BASE = 14):
  0  ships_sent
  1  eta
  2  owner_at_launch_me        (1-hot, focal=launch-time tgt owner==me)
  3  owner_at_launch_neutral
  4  owner_at_launch_enemy
  5  owner_at_arrival_me       (1-hot, from predict_garrison_at)
  6  owner_at_arrival_neutral
  7  owner_at_arrival_enemy
  8  combat_margin_at_arrival  ((ships - garrison) / max(1, garrison),
                                clipped [-1, +1])
  9  src_production
  10 tgt_production
  11 src_distance_to_sun
  12 tgt_distance_to_sun_at_eta
  13 tgt_distance_to_opp_centroid_at_eta

The caller injects leaf_delta as feats[14] before prediction. LightGBM
trees don't need normalized inputs — raw values are kept here for
interpretability (saves a normalization round-trip).

Origin: B.1 within-owner stratified probe (audit/2026-05-29-pveta-leaf-
residual-within-owner.md). The probe found within-cell signal in:
  - enemy-launch ship-quintile (F=4.66 at K=10) — captured by feats[0,8]
  - me-launch target_id (F=4.27 at K=10) — captured by per-planet feats
    [9-13] × owner_at_launch interaction (LightGBM trees learn the
    interaction natively via splits)

The encoder is a pure function — both training-time corpus gen and
inference-time chooser hook the same code path.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from lib.orbit import is_orbiting as _is_orb, predict_relative as _pr
from lib.world_model import predict_garrison_at as _pga

# Sun position. Matches lib/geometry.CENTER = 50.0; duplicated here to
# avoid a cross-module dep just for one constant.
SUN_CENTER = 50.0

FEATURE_NAMES = [
    "ships_sent",                          # 0
    "eta",                                 # 1
    "owner_at_launch_me",                  # 2
    "owner_at_launch_neutral",             # 3
    "owner_at_launch_enemy",               # 4
    "owner_at_arrival_me",                 # 5
    "owner_at_arrival_neutral",            # 6
    "owner_at_arrival_enemy",              # 7
    "combat_margin_at_arrival",            # 8
    "src_production",                      # 9
    "tgt_production",                      # 10
    "src_distance_to_sun",                 # 11
    "tgt_distance_to_sun_at_eta",          # 12
    "tgt_distance_to_opp_centroid_at_eta", # 13
    "leaf_delta",                          # 14 — injected at predict time
]
FEATURE_DIM_BASE = 14   # encoder fills feats[0..13]
FEATURE_DIM_FULL = 15   # caller fills feats[14]


def _owner_one_hot(planet_owner: int, focal_seat: int,
                   ) -> tuple[float, float, float]:
    """Return (me, neutral, enemy). neutral encodes planet_owner == -1;
    enemy encodes any non-focal non-neutral owner (covers 2P and 4P)."""
    if planet_owner == -1:
        return (0.0, 1.0, 0.0)
    if planet_owner == focal_seat:
        return (1.0, 0.0, 0.0)
    return (0.0, 0.0, 1.0)


def _planet_position_at(planet, omega: float,
                        lead_turns: int) -> tuple[float, float]:
    """Return (x, y) at `lead_turns` from now. Orbital planets use
    `lib.orbit.predict_relative`; static planets return current (x, y)."""
    if lead_turns == 0:
        return float(planet.x), float(planet.y)
    if _is_orb(planet):
        return _pr(planet, omega, lead_turns)
    return float(planet.x), float(planet.y)


def _opp_centroid_at_launch(world, focal_seat: int,
                            ) -> tuple[float, float] | None:
    """Production-weighted mean of non-focal, non-neutral planets at the
    current step. Returns None when no such planets exist.

    For 4P games this lumps all opponents into one centroid — coarser
    than per-opponent centroids but matches the 2P feature shape. 4P
    refinement is deferred (see HANDOVER B.2 plan, out-of-scope §)."""
    total_w = 0.0
    sx = 0.0
    sy = 0.0
    for p in world.planets_by_id.values():
        if p.owner == focal_seat or p.owner == -1:
            continue
        w = float(p.production)
        if w <= 0:
            continue
        total_w += w
        sx += w * float(p.x)
        sy += w * float(p.y)
    if total_w <= 0:
        return None
    return (sx / total_w, sy / total_w)


def encode_features(
    src,                # Planet at launch (namedtuple-like with .id, .owner, .x, .y, .ships, .production)
    tgt,                # Planet at launch
    ships: int,
    eta: int,
    me: int,            # focal seat
    world,              # lib.intent.World (provides planets_by_id, omega, step)
    world_model,        # lib.world_model.WorldModel (provides ledger)
) -> np.ndarray:
    """Encode the FEATURE_DIM_BASE-d candidate feature vector. Pure
    function — no env vars, no I/O. Caller appends leaf_delta at index
    14 before model predict."""
    feats = np.zeros(FEATURE_DIM_BASE, dtype=np.float32)

    feats[0] = float(ships)
    feats[1] = float(eta)

    # Owner at launch (raw target owner now).
    me_l, n_l, e_l = _owner_one_hot(int(tgt.owner), int(me))
    feats[2] = me_l
    feats[3] = n_l
    feats[4] = e_l

    # Predicted owner + garrison at arrival.
    arrivals = (world_model.ledger.get(int(tgt.id)) or []) if world_model is not None else []
    try:
        arr_owner, arr_garrison = _pga(tgt, int(eta), arrivals)
    except Exception:
        # Fall back to launch state if the prediction fails.
        arr_owner = int(tgt.owner)
        arr_garrison = float(tgt.ships)
    me_a, n_a, e_a = _owner_one_hot(int(arr_owner), int(me))
    feats[5] = me_a
    feats[6] = n_a
    feats[7] = e_a

    # Combat margin at arrival, clipped [-1, +1].
    denom = max(1.0, float(arr_garrison))
    margin = (float(ships) - float(arr_garrison)) / denom
    if margin < -1.0:
        margin = -1.0
    elif margin > 1.0:
        margin = 1.0
    feats[8] = float(margin)

    # Per-planet covariates.
    feats[9] = float(src.production)
    feats[10] = float(tgt.production)

    # Distance to sun: src now, tgt at eta (orbital planets move).
    omega = float(getattr(world, "omega", 0.0))
    feats[11] = math.hypot(float(src.x) - SUN_CENTER,
                           float(src.y) - SUN_CENTER)
    tgt_x_at, tgt_y_at = _planet_position_at(tgt, omega, int(eta))
    feats[12] = math.hypot(tgt_x_at - SUN_CENTER,
                           tgt_y_at - SUN_CENTER)

    # Distance to opponent centroid (at-launch centroid, target position
    # at-arrival — the natural geometric question is "where will I land
    # relative to where the opponent's mass currently is").
    centroid = _opp_centroid_at_launch(world, int(me))
    if centroid is None:
        feats[13] = 0.0
    else:
        feats[13] = math.hypot(tgt_x_at - centroid[0],
                               tgt_y_at - centroid[1])

    return feats
