"""Opt-in candidate trace hook for the trajectory chooser.

Gated by `BASELINE_TRAJECTORY_TRACE=<jsonl_path>`. When set, every
scored candidate (solo or joint) is featurized via
`lib.shot_features.encode_shot_features` and scored by the on-disk
LightGBM Booster (`data/shot_validator/validator_booster.txt`) using
the pure-Python tree walker, and one JSON line is appended with
`(step, kind, src_id, tgt_id, ships, angle, wait_N, eta, delta, p)`.

Used by `scripts/probe_ml_logit_signal.py` to gate Reframe A
implementation: confirms the Booster has discriminative power on
pv_eta's actual candidate distribution and carries information
independent of the chooser's `delta`.

Default behaviour (env var unset): all entry points are no-ops; no
import cost beyond the module load.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_TRACE_PATH: str = os.environ.get("BASELINE_TRAJECTORY_TRACE", "").strip()
_ENABLED: bool = bool(_TRACE_PATH)

_TRACE_FILE = None
_PARSED = None
_LOAD_FAILED: bool = False

# Side-channel for the chooser's ACCEPTED candidate set (Reframe B.1
# diagnostic probe). Independent of the per-scored-candidate trace
# above; no Booster dependency. Gated by BASELINE_ACCEPTED_TRACE.
_ACCEPTED_PATH: str = os.environ.get("BASELINE_ACCEPTED_TRACE", "").strip()
_ACCEPTED_ENABLED: bool = bool(_ACCEPTED_PATH)
_ACCEPTED_FILE = None
_ACCEPTED_LOAD_FAILED: bool = False


def _ensure_loaded() -> bool:
    """Lazy-open the trace file and lazy-load the Booster. Returns True
    iff both are ready to use. Never raises."""
    global _TRACE_FILE, _PARSED, _LOAD_FAILED
    if not _ENABLED or _LOAD_FAILED:
        return False
    if _TRACE_FILE is None:
        try:
            Path(_TRACE_PATH).parent.mkdir(parents=True, exist_ok=True)
            _TRACE_FILE = open(_TRACE_PATH, "a", buffering=1)
        except OSError:
            _LOAD_FAILED = True
            return False
    if _PARSED is None:
        try:
            from lib._validator_tree_walker import parse_booster_text
            booster_path = (
                Path(__file__).resolve().parents[2]
                / "data" / "shot_validator" / "validator_booster.txt"
            )
            _PARSED = parse_booster_text(booster_path.read_text())
        except Exception:
            _LOAD_FAILED = True
            return False
    return True


def _predict_one(world: Any, world_model: Any, me: int,
                 src_id: int, ships: int, angle: float) -> float | None:
    """Featurize a single candidate and return P_success. Returns None
    if featurization or prediction fails."""
    try:
        import numpy as np
        from lib.shot_features import encode_shot_features
        from lib._validator_tree_walker import predict_proba
        emit = [int(src_id), float(angle), int(ships)]
        feats = encode_shot_features(
            emit, world.obs_raw, int(me),
            world=world, world_model=world_model,
        )
        if feats is None:
            return None
        X = np.asarray(feats, dtype=np.float32).reshape(1, -1)
        return float(predict_proba(_PARSED, X)[0])
    except Exception:
        return None


def trace_solo(world: Any, world_model: Any, me: int,
               src_id: int, tgt_id: int, ships: int, angle: float,
               wait_N: int, eta: int, delta: float) -> None:
    """Emit one JSON line for a scored solo candidate. No-op when
    `BASELINE_TRAJECTORY_TRACE` is unset."""
    if not _ENABLED or not _ensure_loaded():
        return
    try:
        p = _predict_one(world, world_model, me, src_id, ships, angle)
        rec = {
            "step": int(getattr(world, "step", 0)),
            "kind": "solo",
            "src_id": int(src_id),
            "tgt_id": int(tgt_id),
            "ships": int(ships),
            "angle": float(angle),
            "wait_N": int(wait_N),
            "eta": int(eta),
            "delta": float(delta),
            "p": p,
        }
        _TRACE_FILE.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _ensure_accepted_loaded() -> bool:
    """Lazy-open the accepted-trace file. Returns True iff ready. Never
    raises."""
    global _ACCEPTED_FILE, _ACCEPTED_LOAD_FAILED
    if not _ACCEPTED_ENABLED or _ACCEPTED_LOAD_FAILED:
        return False
    if _ACCEPTED_FILE is None:
        try:
            Path(_ACCEPTED_PATH).parent.mkdir(parents=True, exist_ok=True)
            _ACCEPTED_FILE = open(_ACCEPTED_PATH, "a", buffering=1)
        except OSError:
            _ACCEPTED_LOAD_FAILED = True
            return False
    return True


def trace_accepted(world: Any, me: int, accepted: list) -> None:
    """Emit one JSON line per accepted candidate the chooser committed
    this turn. `accepted` is a list of dicts with keys: kind ('solo' or
    'joint'), src_id, tgt_id, ships, angle, wait_N, eta, delta_pred,
    and optionally joint_id (turn-local counter shared across legs of
    the same joint coalition). No-op when BASELINE_ACCEPTED_TRACE is
    unset."""
    if not _ACCEPTED_ENABLED or not _ensure_accepted_loaded():
        return
    try:
        step = int(getattr(world, "step", 0))
        for entry in accepted:
            rec = {
                "step": step,
                "me": int(me),
                "kind": str(entry["kind"]),
                "src_id": int(entry["src_id"]),
                "tgt_id": int(entry["tgt_id"]),
                "ships": int(entry["ships"]),
                "angle": float(entry["angle"]),
                "wait_N": int(entry["wait_N"]),
                "eta": int(entry["eta"]),
                "delta_pred": float(entry["delta_pred"]),
            }
            if "joint_id" in entry:
                rec["joint_id"] = int(entry["joint_id"])
            _ACCEPTED_FILE.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def trace_joint(world: Any, world_model: Any, me: int,
                launches: list, delta: float) -> None:
    """Emit one JSON line for a scored joint candidate. Per-leg P
    values are computed; `delta` is the joint's combined score.
    `launches` is the list of `(src, tgt, ships, angle, wait_N)`
    tuples."""
    if not _ENABLED or not _ensure_loaded():
        return
    try:
        legs = []
        for (src, tgt, ships, angle, wait_N) in launches:
            p = _predict_one(
                world, world_model, me,
                int(src.id), int(ships), float(angle),
            )
            legs.append({
                "src_id": int(src.id),
                "tgt_id": int(tgt.id),
                "ships": int(ships),
                "angle": float(angle),
                "wait_N": int(wait_N),
                "p": p,
            })
        rec = {
            "step": int(getattr(world, "step", 0)),
            "kind": "joint",
            "legs": legs,
            "delta": float(delta),
        }
        _TRACE_FILE.write(json.dumps(rec) + "\n")
    except Exception:
        pass
