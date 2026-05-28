"""baseline + konbu17-style shot validator filter.

Wraps `agents.baseline.main.agent`. Calls the inner agent to produce the
per-turn emit list, then filters out emits whose ensemble-averaged
P(success) is below threshold (default 0.30).

Self-reinforcement emits (target already owned by us) are passed through
unfiltered — konbu17 design: these are never filtered at training or
inference.

NO topk1 filter applied (PM3 decision): the production chooser already
does multi-source coordination; topk1 would destroy that strength. We
only drop sub-threshold emits.

Weights live in `_WEIGHTS_B64` (base64-encoded npz). The placeholder
below is empty; `scripts/embed_validator_weights.py` patches it from
`data/shot_validator/validator_ensemble_weights.npz` after training.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import numpy as np

from agents.baseline.main import agent as _inner_agent
# Single-line imports below: the bundler's per-line import-stripping regex
# leaks continuation lines from a parenthesised multi-line import as indented
# orphans (IndentationError at runtime). Friction tag:
# `bundler-modular-agent-namespace-access-breaks-bundle`.
from lib.shot_features import FEATURE_DIM, encode_shot_features, target_owned_by

# === inlined weights (base64-encoded npz) ===
# Populated by `scripts/embed_validator_weights.py` after training.
_WEIGHTS_B64 = ""
# === end inlined weights ===

# Cached parsed weights (3-model ensemble).
_MODELS: list[dict] | None = None
_THRESHOLD: float | None = None
_LOAD_FAILED: bool = False


def _load_weights() -> None:
    global _MODELS, _THRESHOLD, _LOAD_FAILED
    if not _WEIGHTS_B64:
        _LOAD_FAILED = True
        return
    try:
        blob = base64.b64decode(_WEIGHTS_B64)
        with np.load(io.BytesIO(blob)) as npz:
            threshold = float(npz["threshold"])
            models: list[dict] = []
            for i in range(3):
                P = {}
                for k in ("W0", "b0", "W1", "b1", "W2", "b2"):
                    P[k] = np.ascontiguousarray(npz[f"m{i}_{k}"]).astype(np.float32)
                models.append(P)
        _MODELS = models
        _THRESHOLD = threshold
    except Exception:
        _LOAD_FAILED = True


def _ensemble_proba(X: np.ndarray) -> np.ndarray:
    """Average sigmoid across the 3 MLPs. X is (B, 24)."""
    assert _MODELS is not None
    out = np.zeros(len(X), dtype=np.float32)
    for P in _MODELS:
        h = np.maximum(0.0, X @ P["W0"] + P["b0"])
        h = np.maximum(0.0, h @ P["W1"] + P["b1"])
        s = (h @ P["W2"] + P["b2"]).ravel()
        out += 1.0 / (1.0 + np.exp(-np.clip(s, -30.0, 30.0)))
    return out / float(len(_MODELS))


def _focal_seat_from_obs(obs: Any) -> int:
    """Read `obs.player` (kaggle_environments observation field)."""
    if isinstance(obs, dict):
        return int(obs.get("player", 0))
    return int(getattr(obs, "player", 0))


def agent(obs: Any, configuration: Any = None) -> list:
    inner = _inner_agent(obs, configuration) or []
    if not inner:
        return []

    if _MODELS is None and not _LOAD_FAILED:
        _load_weights()
    # No weights -> pass-through (matches production baseline).
    if _MODELS is None or _THRESHOLD is None:
        return inner

    focal_seat = _focal_seat_from_obs(obs)

    # Bypass: env var override (debugging / ablation).
    if os.environ.get("BASELINE_VALIDATOR", "1") == "0":
        return inner

    # Encode per-emit features; carve out self-reinforce.
    to_score: list[tuple[int, np.ndarray]] = []  # (index_in_inner, feats)
    survivors_mask = [True] * len(inner)
    for i, emit in enumerate(inner):
        if target_owned_by(emit, obs, focal_seat):
            continue  # self-reinforce, pass-through unfiltered
        feats = encode_shot_features(emit, obs, focal_seat)
        if feats is None or feats.shape[0] != FEATURE_DIM:
            continue  # malformed; let it pass through (safer than dropping)
        to_score.append((i, feats))

    if not to_score:
        return inner

    X = np.stack([f for _, f in to_score]).astype(np.float32)
    probs = _ensemble_proba(X)
    for (idx_in_inner, _), p in zip(to_score, probs):
        if p < _THRESHOLD:
            survivors_mask[idx_in_inner] = False

    filtered = [emit for emit, keep in zip(inner, survivors_mask) if keep]
    return filtered
