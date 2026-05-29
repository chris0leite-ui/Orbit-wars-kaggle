"""baseline + konbu17-style shot validator filter (Phase 2 v2 — LightGBM).

Wraps `agents.baseline.main.agent`. Calls the inner agent to produce the
per-turn emit list, then filters out emits whose LightGBM-Booster
P(success) is below threshold.

Self-reinforcement emits (target already owned by us) are passed through
unfiltered — konbu17 design: these are never filtered at training or
inference.

NO topk1 filter applied (PM3 decision): the production chooser already
does multi-source coordination; topk1 would destroy that strength. We
only drop sub-threshold emits.

The booster is embedded as a gzip+base64 LightGBM model_to_string()
dump in `_BOOSTER_B64`. `scripts/embed_validator_weights.py` patches
both the blob and the matching `_THRESHOLD_DEFAULT` after training
(`scripts/train_validator.py`).
"""

from __future__ import annotations

import base64
import gzip
import os
from typing import Any

import numpy as np

from agents.baseline.main import agent as _inner_agent
# Single-line imports below: the bundler's per-line import-stripping regex
# leaks continuation lines from a parenthesised multi-line import as indented
# orphans (IndentationError at runtime). Friction tag:
# `bundler-modular-agent-namespace-access-breaks-bundle`.
from lib.shot_features import FEATURE_DIM, encode_shot_features, target_owned_by
from lib._validator_tree_walker import parse_booster_text, predict_proba

# === inlined booster (gzip+base64 of LightGBM model_to_string()) ===
# Populated by `scripts/embed_validator_weights.py` after training.
_BOOSTER_B64 = ""
_THRESHOLD_DEFAULT = 0.30
# === end inlined booster ===

_PARSED = None  # cached ParsedBooster
_THRESHOLD: float | None = None
_LOAD_FAILED: bool = False


def _load_booster() -> None:
    global _PARSED, _THRESHOLD, _LOAD_FAILED
    if not _BOOSTER_B64:
        _LOAD_FAILED = True
        return
    try:
        blob_gz = base64.b64decode(_BOOSTER_B64)
        text = gzip.decompress(blob_gz).decode("utf-8")
        _PARSED = parse_booster_text(text)
        _THRESHOLD = float(_THRESHOLD_DEFAULT)
    except Exception:
        _LOAD_FAILED = True


def _focal_seat_from_obs(obs: Any) -> int:
    """Read `obs.player` (kaggle_environments observation field)."""
    if isinstance(obs, dict):
        return int(obs.get("player", 0))
    return int(getattr(obs, "player", 0))


def agent(obs: Any, configuration: Any = None) -> list:
    inner = _inner_agent(obs, configuration) or []
    if not inner:
        return []

    if _PARSED is None and not _LOAD_FAILED:
        _load_booster()
    # No booster -> pass-through (matches production baseline).
    if _PARSED is None or _THRESHOLD is None:
        return inner

    focal_seat = _focal_seat_from_obs(obs)

    # Bypass: env var override (debugging / ablation).
    if os.environ.get("BASELINE_VALIDATOR", "1") == "0":
        return inner

    # Build World + WorldModel ONCE per turn. The encoder's Tier 1+2 +
    # Stage 1.5 features each cost ~5 ms on construction; with ~50
    # emits/turn the per-emit rebuild would be 250 ms/turn. Build
    # failures fall through to None → encoder slow-path builds them
    # itself (matches synthetic-obs behaviour).
    world = None
    world_model = None
    try:
        from lib.intent import World
        from lib.world_model import WorldModel
        world = World.from_obs(obs)
        world_model = WorldModel.from_world(world)
    except Exception:
        world = None
        world_model = None

    # Encode per-emit features; carve out self-reinforce.
    to_score: list[tuple[int, np.ndarray]] = []  # (index_in_inner, feats)
    survivors_mask = [True] * len(inner)
    for i, emit in enumerate(inner):
        if target_owned_by(emit, obs, focal_seat):
            continue  # self-reinforce, pass-through unfiltered
        feats = encode_shot_features(
            emit, obs, focal_seat, world=world, world_model=world_model,
        )
        if feats is None or feats.shape[0] != FEATURE_DIM:
            continue  # malformed; let it pass through (safer than dropping)
        to_score.append((i, feats))

    if not to_score:
        return inner

    X = np.stack([f for _, f in to_score]).astype(np.float32)
    probs = predict_proba(_PARSED, X)
    for (idx_in_inner, _), p in zip(to_score, probs):
        if p < _THRESHOLD:
            survivors_mask[idx_in_inner] = False

    filtered = [emit for emit, keep in zip(inner, survivors_mask) if keep]
    return filtered
