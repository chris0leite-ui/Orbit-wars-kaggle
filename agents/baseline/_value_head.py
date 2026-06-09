"""Value-head additive term — Reframe B.2.

Per-candidate LightGBM regressor predicting seat-0 ship-delta over the
next K=10 turns conditional on the candidate decision. The chooser adds
`λ * head_output` to each candidate's scalar score.

Public API:
- `vh_is_enabled() -> bool` — `BASELINE_VH_LAMBDA != 0.0`.
- `vh_get_lambda() -> float` — the additive coefficient.
- `vh_featurize_prerank(prerank, world, world_model)` — batch-encode
  the 14-d base feature vector per candidate; returns dict keyed by
  candidate identity. Skips the per-candidate leaf_delta (column 14)
  because that's only known inside the chooser's scoring loop.
- `vh_predict_one(feats_map, src_id, tgt_id, ships, angle, wait_N,
  leaf_delta) -> float` — inject leaf_delta into the cached feature
  vector and return the raw regressor output. Returns 0.0 on key miss,
  encoder failure, or model load fail (i.e. no-op for that candidate).

At λ=0 (default), this module is bypassed entirely — byte-identical to
bare pv_eta. Submit-time inference uses the pure-Python tree walker; no
lightgbm dependency at submit time.

Distinct from `agents/baseline/_ml_logit.py` (Reframe A — falsified):
  - ML: per-shot BINARY classifier on landing success, centered-logit
    additive term. Falsified 2026-05-29.
  - VH (this file): per-candidate REGRESSION on K=10 ship-delta, raw
    output × λ additive term. Uses leaf_delta as one input feature.

Coexists additively with ML in the chooser: solo score updates are
`score = leaf_Δ + λ_ml · ml_logit + λ_vh · vh_output`. Both default to
λ=0; we ship VH at λ=1.0 (head's natural scale).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

# === Constants and lazy state =============================================

VH_LAMBDA: float = float(os.environ.get("BASELINE_VH_LAMBDA", "0.0"))

# Empirical-mean centering. The shipped VH was trained on a corpus of
# ACCEPTED candidates (regression target = K=10 ship-delta); since
# accepted moves are positive-EV in expectation, the head learned a
# heavy positive bias. Probe (2026-06-02, 500 random feature rows):
# mean(head_out) = +102, std = 78. Adding raw head_out at λ=1.0 swamps
# the chooser's base score (O(10-100)) — every candidate gets a
# +90..+225 boost, blowing past the emit gate. Subtract BIAS before
# scaling so the centered correction is mean-zero. Default 0.0
# preserves the pre-fix behavior; set BASELINE_VH_BIAS=102.0 (or
# game-measured value) to activate centering.
VH_BIAS: float = float(os.environ.get("BASELINE_VH_BIAS", "0.0"))

# Rerank mode (2026-06-03 Phase H — Option C). When > 0, the chooser
# does NOT add λ·head_out to candidate scores. Instead, it takes the
# top-K solo candidates by base score and re-sorts them by VH head
# output (descending). Joints unaffected. Base score is preserved on
# each tuple — only the order changes. Sidesteps the magnitude-swamp
# failure mode that closed the VH-additive axis (5 falsifications,
# Rule 37). Consumes the model's verified Spearman ρ=+0.386 rank
# signal without letting the head's prediction magnitude touch the
# emit gate.
VH_RERANK_K: int = int(os.environ.get("BASELINE_VH_RERANK_K", "0"))

# Bundler patches `_VH_MODEL_B64` to a gzip+base64 LightGBM
# `model_to_string()` dump (regression objective). In source mode (empty
# blob), we fall back to reading `data/value_head/value_head_model.txt`
# from disk.
_VH_MODEL_B64: str = ""

_PARSED = None
_LOAD_FAILED: bool = False

# Candidate-key precision: prerank angles are floats. Two candidates
# differing only in the 7th decimal of angle would key-collide; this is
# vanishingly rare. Misses fall through to vh_output=0 (no-op).
_ANGLE_KEY_DECIMALS: int = 6


def vh_is_enabled() -> bool:
    # True if EITHER additive or rerank consumption is active.
    return VH_LAMBDA != 0.0 or VH_RERANK_K > 0


def vh_get_lambda() -> float:
    return VH_LAMBDA


def vh_get_bias() -> float:
    return VH_BIAS


def vh_rerank_k() -> int:
    return VH_RERANK_K


def _candidate_key(src_id: int, tgt_id: int, ships: int,
                   angle: float, wait_N: int) -> tuple:
    return (
        int(src_id), int(tgt_id), int(ships),
        round(float(angle), _ANGLE_KEY_DECIMALS),
        int(wait_N),
    )


# === Model load ===========================================================

def _load_parsed() -> Any | None:
    """Lazy-load the parsed regressor. Returns None on any failure
    (caller treats this as VH disabled for the run)."""
    global _PARSED, _LOAD_FAILED
    if _PARSED is not None or _LOAD_FAILED:
        return _PARSED
    try:
        from lib._validator_tree_walker import parse_booster_text
        if _VH_MODEL_B64:
            import base64
            import gzip
            text = gzip.decompress(base64.b64decode(_VH_MODEL_B64)).decode()
        else:
            model_path = (
                Path(__file__).resolve().parents[2]
                / "data" / "value_head" / "value_head_model.txt"
            )
            text = model_path.read_text()
        _PARSED = parse_booster_text(text)
        return _PARSED
    except Exception:
        _LOAD_FAILED = True
        return None


# === Featurization + prediction ===========================================

def vh_featurize_prerank(prerank: list, world: Any, world_model: Any,
                         ) -> dict[tuple, np.ndarray | None]:
    """Encode the 14-d base feature vector for every prerank entry.
    Returns dict keyed by candidate identity. Entries with None features
    (encoder rejected the emit) are stored as None — `vh_predict_one`
    will skip them.

    `prerank` rows: `(cheap_delta, src, tgt, ships, angle, eta_hint,
    prop_horizon, wait_N)`. `src` and `tgt` are Planet objects with
    `.id`, `.owner`, `.x`, `.y`, `.ships`, `.production`.

    Single-line import; bundler's per-line strip regex leaks
    parenthesised multi-line imports as indented orphans (same
    constraint as `_ml_logit.ml_featurize_prerank`).
    """
    from lib.value_head_features import encode_features
    me = int(getattr(world, "my_id", 0))
    feats: dict[tuple, np.ndarray | None] = {}
    for row in prerank:
        if len(row) < 8:
            continue
        _, src, tgt, ships, angle, eta_hint, _, wait_N = row
        key = _candidate_key(int(src.id), int(tgt.id), int(ships),
                             float(angle), int(wait_N))
        if key in feats:
            continue
        try:
            arr = encode_features(
                src, tgt, int(ships), int(eta_hint),
                me, world, world_model,
            )
        except Exception:
            arr = None
        feats[key] = arr
    return feats


def vh_predict_one(feats_map: dict[tuple, np.ndarray | None],
                   src_id: int, tgt_id: int, ships: int,
                   angle: float, wait_N: int,
                   leaf_delta: float) -> float:
    """Per-candidate prediction. Inject `leaf_delta` at feats[14] and
    run a single forward pass through the regressor. Returns the raw
    head output (interpreted as predicted K=10 ship-delta), or 0.0 on
    any failure (no-op for that candidate)."""
    parsed = _load_parsed()
    if parsed is None:
        return 0.0
    key = _candidate_key(src_id, tgt_id, ships, angle, wait_N)
    base = feats_map.get(key)
    if base is None:
        return 0.0
    try:
        from lib._validator_tree_walker import predict_raw
        full = np.empty(15, dtype=np.float32)
        full[:14] = base
        full[14] = float(leaf_delta)
        X = full.reshape(1, -1)
        return float(predict_raw(parsed, X)[0])
    except Exception:
        return 0.0
