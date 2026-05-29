"""Pure-numpy forward pass for the trained shot-validator MLP ensemble.

Lifted from `agents/baseline_validated/main.py` (sibling branch
`claude/competition-objective-alignment-hqNVM` commit `4a8e4c0`) so the
forward pass is reusable from any consumer that wants a learned
`P(launch will succeed)` estimate. The agent wrapper there used it to
filter OUR-side emits; this module exists so the OPP-model layer (in
`lib/opp_model.py`) can use the same network with the opponent's seat
as the focal seat.

Single public entry: `ensemble_proba(X)` — `np.ndarray((B, 25)) ->
np.ndarray((B,))`. Lazy-loads the weights on first call; sets a sticky
failure flag on any load error so the consumer can fall back to a
non-learned policy without crashing.
"""

from __future__ import annotations

import base64
import io

import numpy as np

from lib._validator_weights import _WEIGHTS_B64

_MODELS: list[dict] | None = None
_LOAD_FAILED: bool = False


def _load_weights() -> None:
    global _MODELS, _LOAD_FAILED
    if not _WEIGHTS_B64:
        _LOAD_FAILED = True
        return
    try:
        blob = base64.b64decode(_WEIGHTS_B64)
        with np.load(io.BytesIO(blob)) as npz:
            models: list[dict] = []
            for i in range(3):
                P = {}
                for k in ("W0", "b0", "W1", "b1", "W2", "b2"):
                    P[k] = np.ascontiguousarray(npz[f"m{i}_{k}"]).astype(np.float32)
                models.append(P)
        _MODELS = models
    except Exception:
        _LOAD_FAILED = True


def is_ready() -> bool:
    """True once the weights have been parsed; lazy-triggers a load
    attempt on first call. False if the blob is missing or malformed."""
    global _MODELS
    if _MODELS is None and not _LOAD_FAILED:
        _load_weights()
    return _MODELS is not None


def ensemble_proba(X: np.ndarray) -> np.ndarray:
    """Average sigmoid across the 3-MLP ensemble.

    X is float32 (B, 25). Returns float32 (B,). Each model: two ReLU
    hidden layers (25 -> 64 -> 32) and a sigmoid output. Clips the pre-
    sigmoid logits to [-30, 30] for numerical safety (matches the
    sibling implementation byte-for-byte).
    """
    if not is_ready():
        raise RuntimeError("validator MLP weights not loaded")
    out = np.zeros(len(X), dtype=np.float32)
    for P in _MODELS:  # type: ignore[union-attr]
        h = np.maximum(0.0, X @ P["W0"] + P["b0"])
        h = np.maximum(0.0, h @ P["W1"] + P["b1"])
        s = (h @ P["W2"] + P["b2"]).ravel()
        out += 1.0 / (1.0 + np.exp(-np.clip(s, -30.0, 30.0)))
    return out / float(len(_MODELS))  # type: ignore[arg-type]
