"""Learned shot-success filter (the May shot-validator MLP, finally trained).

A tiny MLP scores every planned ATTACK wave with P(target still ours 10
turns after arrival), trained on live-ladder episode outcomes. Waves below
a threshold are dropped before dispatch — reject-only, never proposes.

Train/serve contract
--------------------
`encode_shot_features` is the single feature encoder. The labeling script
(`scripts/label_shot_outcomes.py`) calls it on raw replay arrays; the
in-agent veto calls it on the identical positional layout rebuilt from
ParsedObs tensors (obs.py documents that the layouts match the replay
JSON). Any feature change must bump FEATURE_VERSION and retrain.

Gate: PRODUCER_PLUS_SHOT_MLP=<threshold in (0,1)>  (unset/0 = OFF)
      PRODUCER_PLUS_SHOT_MLP_2P_ONLY=1             (optional)

Weights are baked between the BEGIN/END TRAINED WEIGHTS markers by
`scripts/train_shot_mlp.py`. If the gate is on but no weights are baked,
the filter no-ops and warns once (a mis-built bundle must not lose games
by raising mid-episode); `tests/test_shot_mlp.py` asserts bundles built
with the gate on carry weights.
"""

from __future__ import annotations

import base64
import math
import sys

FEATURE_VERSION = 1
N_FEATURES = 24
LABEL_BUFFER = 10   # steps after eta to check ownership

# Normalisation constants — must match data/shot_validator/schema.json.
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
    """Engine fleet-speed curve (mirrors lib/fleet.py incl. the 1-ship floor
    and the 1000-ship cap; kept import-free)."""
    if ships <= 1:
        return 1.0
    if ships >= 1000:
        return 6.0
    return 1.0 + (6.0 - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5


def encode_shot_features(
    src_planet, target_planet, ships_sent, distance, eta, fs,
    all_planets, all_fleets, focal_seat, step,
):
    """24-dim feature vector, all values in [0, 1] (ship_diff in [-1, 1]).

    Planet rows are positional 7-tuples (id, owner, x, y, radius, ships,
    production); fleet rows (id, owner, x, y, angle, from_planet, ships) —
    the raw replay-JSON layout.
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

    src_garrison = max(1, src_planet[5])
    shot_ships = min(1.0, ships_sent / NORM["max_ships"])
    # Logically a launch can't exceed the source garrison; cap at 1.0
    # (records sometimes have stale src.ships from pre-launch state).
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

    my_total_ships = sum(p[5] for p in all_planets if int(p[1]) == focal_seat) + ship_allied
    enemy_total_ships = sum(p[5] for p in all_planets
                            if int(p[1]) not in (-1, focal_seat)) + ship_enemy
    # ship_diff is signed in [-1, 1] (clipped). Top-10 games can produce
    # thousands of total ships; norm chosen to keep the typical
    # distribution centred without saturating extreme blowouts.
    ship_diff = max(-1.0, min(1.0,
        (my_total_ships - enemy_total_ships) / NORM["max_ships"]))
    my_total_ships_n = min(1.0, my_total_ships / NORM["max_ships"])
    enemy_total_ships_n = min(1.0, enemy_total_ships / NORM["max_ships"])
    meta_turn = step / NORM["episode_steps"]
    my_planet_count = sum(1 for p in all_planets if int(p[1]) == focal_seat)
    enemy_planet_count = sum(1 for p in all_planets if int(p[1]) not in (-1, focal_seat))
    my_pc_n = my_planet_count / NORM["max_planets"]
    enemy_pc_n = enemy_planet_count / NORM["max_planets"]

    return [
        sps_ships, sps_prod, sps_rad,
        tgt_ships, tgt_prod, tgt_rad,
        owner_mine, owner_neutral, owner_enemy,
        shot_ships, shot_frac, shot_dist, shot_eta, shot_fs,
        in_flight_n_allied, in_flight_ship_allied,
        in_flight_n_enemy, in_flight_ship_enemy,
        meta_turn, my_total_ships_n, enemy_total_ships_n,
        ship_diff, my_pc_n, enemy_pc_n,
    ]


# === BEGIN TRAINED WEIGHTS (written by scripts/train_shot_mlp.py) ===
WEIGHTS_B64 = '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'
WEIGHTS_META = {'dims': [24, 32, 16, 8, 1]}
# === END TRAINED WEIGHTS ===

_MLP_CACHE = None
_WARNED_NO_WEIGHTS = False


def _decode_weights():
    """Decode WEIGHTS_B64 into a list of (W, b) numpy pairs, or None."""
    global _MLP_CACHE
    if _MLP_CACHE is not None:
        return _MLP_CACHE
    if not WEIGHTS_B64:
        return None
    import numpy as np
    raw = np.frombuffer(base64.b64decode(WEIGHTS_B64), dtype=np.float32)
    dims = WEIGHTS_META["dims"]          # e.g. [24, 32, 16, 8, 1]
    layers = []
    off = 0
    for i in range(len(dims) - 1):
        n_in, n_out = dims[i], dims[i + 1]
        w = raw[off:off + n_in * n_out].reshape(n_in, n_out).copy()
        off += n_in * n_out
        b = raw[off:off + n_out].copy()
        off += n_out
        layers.append((w, b))
    assert off == raw.size, "weight blob size mismatch vs dims"
    _MLP_CACHE = layers
    return layers


def predict_success(features_rows) -> "object":
    """P(shot succeeds) for each feature row. Returns np.ndarray [N]."""
    import numpy as np
    layers = _decode_weights()
    x = np.asarray(features_rows, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    for i, (w, b) in enumerate(layers):
        x = x @ w + b
        if i < len(layers) - 1:
            x = np.maximum(x, 0.0)        # ReLU hidden
    return 1.0 / (1.0 + np.exp(-x[:, 0]))  # sigmoid head


def apply_shot_mlp_veto(entries, *, obs, threshold: float):
    """Drop valid ATTACK waves with predicted P(success) < threshold.

    ``obs`` is the ParsedObs for the CURRENT (un-debited) observation —
    feature math mirrors the labeler: current positions, straight-line
    distance, eta recomputed from the engine speed curve (NOT the
    planner's intercept eta).
    """
    global _WARNED_NO_WEIGHTS
    if _decode_weights() is None:
        if not _WARNED_NO_WEIGHTS:
            print("shot_mlp: gate ON but no trained weights baked — no-op",
                  file=sys.stderr)
            _WARNED_NO_WEIGHTS = True
        return entries
    import torch

    valid = entries.valid
    if int(valid.sum().item()) == 0:
        return entries
    P = int(obs.P)
    tgt_safe = entries.target_slots.clamp(0, P - 1)
    is_attack = valid & ~obs.owned[tgt_safe]
    idx = is_attack.nonzero(as_tuple=True)[0]
    if int(idx.shape[0]) == 0:
        return entries

    # Rebuild the replay-JSON positional rows from ParsedObs (layouts match).
    alive_idx = obs.alive.nonzero(as_tuple=True)[0].tolist()
    px = obs.x.tolist(); py = obs.y.tolist(); pr = obs.r.tolist()
    pships = obs.ships.tolist(); pprod = obs.prod.tolist()
    powner = obs.owner_abs.tolist()
    planets_rows = [
        (i, powner[i], px[i], py[i], pr[i], pships[i], pprod[i])
        for i in alive_idx
    ]
    by_slot = {i: row for i, row in zip(alive_idx, planets_rows)}
    f_alive = obs.f_alive.nonzero(as_tuple=True)[0].tolist()
    fo = obs.f_owner.tolist(); fs_ = obs.f_ships.tolist()
    fleets_rows = [(0, fo[i], 0.0, 0.0, 0.0, 0, fs_[i]) for i in f_alive]

    pid = int(obs.player_id)
    step = float(obs.step.flatten()[0].item())
    rows = []
    row_entry = []
    src_l = entries.source_slots.tolist()
    tgt_l = entries.target_slots.tolist()
    ships_l = entries.ships.tolist()
    for e in idx.tolist():
        src = by_slot.get(int(src_l[e]))
        tgt = by_slot.get(int(tgt_l[e]))
        if src is None or tgt is None:
            continue
        n_ships = float(ships_l[e])
        d = math.hypot(tgt[2] - src[2], tgt[3] - src[3])
        v = fleet_speed(n_ships)
        eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
        rows.append(encode_shot_features(
            src, tgt, n_ships, d, eta, v,
            planets_rows, fleets_rows, pid, step,
        ))
        row_entry.append(e)
    if not rows:
        return entries

    proba = predict_success(rows)
    drop = [e for e, p in zip(row_entry, proba) if float(p) < threshold]
    import os
    if os.environ.get("PRODUCER_PLUS_SHOT_MLP_DEBUG"):
        print(f"shot_mlp[t={int(step)}] scored {len(rows)} attack waves, "
              f"dropped {len(drop)} "
              f"(p: {' '.join(f'{float(p):.2f}' for p in proba)})",
              file=sys.stderr)
    if not drop:
        return entries
    new_valid = entries.valid.clone()
    new_valid[torch.tensor(drop, dtype=torch.long, device=new_valid.device)] = False
    import dataclasses
    return dataclasses.replace(entries, valid=new_valid)
