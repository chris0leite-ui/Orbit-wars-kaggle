"""ML-logit chooser term — Reframe A.

Exposes the per-shot LightGBM Booster (the same model used by
`agents/baseline_validated/main.py` as a filter, falsified 2026-05-29)
as an additive term inside the trajectory chooser's scalar score.

Public API:
- `is_enabled() -> bool` — `BASELINE_ML_LAMBDA != 0.0`.
- `get_lambda() -> float` — the swept knob.
- `featurize_prerank(prerank, world, world_model)` — batch-featurize
  every prerank entry; returns a dict keyed by candidate identity.
- `score_candidates(feats_map)` — single batched `predict_proba` call,
  returns dict[key, logit(P) - logit(0.5)] for the centered-logit form.
- `lookup(scores, src_id, tgt_id, ships, angle, wait_N) -> float` —
  helper used by the chooser at scoring sites.

At λ=0 (default), this module is bypassed entirely — byte-identical
to bare pv_eta. Submit-time inference uses the pure-Python tree walker
(no lightgbm dependency).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

# === Constants and lazy state =============================================

ML_LAMBDA: float = float(os.environ.get("BASELINE_ML_LAMBDA", "0.0"))
_LOGIT_CENTER: float = float(os.environ.get("BASELINE_ML_CENTER", "0.5"))
_EPS: float = 1e-6

# Bundler patches `_BOOSTER_B64` to a gzip+base64 LightGBM
# `model_to_string()` dump. In source mode (empty blob), we fall back
# to reading `data/shot_validator/validator_booster.txt` from disk.
_BOOSTER_B64: str = ""

_PARSED = None
_LOAD_FAILED: bool = False

# Candidate-key precision: prerank angles are floats. Two candidates
# differing only in the 7th decimal of angle would key-collide; this is
# vanishingly rare. Misses fall through to ml_logit=0 (no-op).
_ANGLE_KEY_DECIMALS: int = 6


def is_enabled() -> bool:
    return ML_LAMBDA != 0.0


def get_lambda() -> float:
    return ML_LAMBDA


def _logit_center() -> float:
    p = min(max(_LOGIT_CENTER, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _candidate_key(src_id: int, tgt_id: int, ships: int,
                   angle: float, wait_N: int) -> tuple:
    return (
        int(src_id), int(tgt_id), int(ships),
        round(float(angle), _ANGLE_KEY_DECIMALS),
        int(wait_N),
    )


# === Booster load ==========================================================

def _load_parsed() -> Any | None:
    """Lazy-load the parsed Booster. Returns None on any failure (caller
    treats this as ML disabled for the run)."""
    global _PARSED, _LOAD_FAILED
    if _PARSED is not None or _LOAD_FAILED:
        return _PARSED
    try:
        from lib._validator_tree_walker import parse_booster_text
        if _BOOSTER_B64:
            import base64
            import gzip
            text = gzip.decompress(base64.b64decode(_BOOSTER_B64)).decode()
        else:
            booster_path = (
                Path(__file__).resolve().parents[2]
                / "data" / "shot_validator" / "validator_booster.txt"
            )
            text = booster_path.read_text()
        _PARSED = parse_booster_text(text)
        return _PARSED
    except Exception:
        _LOAD_FAILED = True
        return None


# === Featurization + prediction ============================================

def featurize_prerank(prerank: list, world: Any, world_model: Any,
                      ) -> dict[tuple, np.ndarray | None]:
    """Encode features for every prerank entry. Returns dict keyed by
    candidate identity. Entries with None features (encoder rejected
    the emit) are stored as None — scoring will skip them and the
    chooser will get a zero ML correction for that candidate.

    `prerank` rows: `(cheap_delta, src, tgt, ships, angle, eta_hint,
    prop_horizon, wait_N)`. `src` and `tgt` are Planet objects with
    `.id`. The encoder expects `emit=[src_id, angle, ships]` and
    pre-built `world` + `world_model` to skip the ~5 ms rebuild."""
    from lib.shot_features import encode_shot_features
    me = int(getattr(world, "my_id", 0))
    obs = getattr(world, "obs_raw", None)
    feats: dict[tuple, np.ndarray | None] = {}
    if obs is None:
        return feats
    for row in prerank:
        # prerank shape: (cheap_delta, src, tgt, ships, angle, eta_hint,
        #                 prop_horizon, wait_N)
        if len(row) < 8:
            continue
        _, src, tgt, ships, angle, _, _, wait_N = row
        key = _candidate_key(int(src.id), int(tgt.id), int(ships),
                             float(angle), int(wait_N))
        if key in feats:
            continue
        try:
            arr = encode_shot_features(
                [int(src.id), float(angle), int(ships)],
                obs, me, world=world, world_model=world_model,
            )
        except Exception:
            arr = None
        feats[key] = arr
    return feats


def score_candidates(feats_map: dict[tuple, np.ndarray | None],
                     ) -> dict[tuple, float]:
    """Batched centered-logit prediction. Returns dict[key, logit(P) -
    logit(center)]. None entries are dropped. Returns empty dict if the
    Booster couldn't be loaded."""
    parsed = _load_parsed()
    if parsed is None:
        return {}
    valid_keys: list[tuple] = []
    rows: list[np.ndarray] = []
    for key, arr in feats_map.items():
        if arr is None:
            continue
        valid_keys.append(key)
        rows.append(arr)
    if not rows:
        return {}
    try:
        from lib._validator_tree_walker import predict_proba
        X = np.asarray(rows, dtype=np.float32)
        ps = predict_proba(parsed, X)
    except Exception:
        return {}
    centered = _logit_center()
    out: dict[tuple, float] = {}
    for key, p in zip(valid_keys, ps):
        pv = min(max(float(p), _EPS), 1.0 - _EPS)
        out[key] = math.log(pv / (1.0 - pv)) - centered
    return out


def lookup(scores: dict[tuple, float], src_id: int, tgt_id: int,
           ships: int, angle: float, wait_N: int) -> float:
    """Lookup a candidate's centered-logit. Returns 0.0 on key miss
    (treated as no-op for that candidate)."""
    key = _candidate_key(src_id, tgt_id, ships, angle, wait_N)
    return float(scores.get(key, 0.0))
