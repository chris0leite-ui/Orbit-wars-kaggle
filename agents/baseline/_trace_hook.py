"""Opt-in accepted-candidate trace for the trajectory chooser.

Gated by `BASELINE_ACCEPTED_TRACE=<jsonl_path>`. When set, the chooser
calls `trace_accepted(world, world_model, me, accepted)` at end-of-turn
with the list of candidates it actually committed (solo `moves` and
solo wait-N `commits`; joints are NOT included — VH training is
solo-only per the B.2 design).

When `BASELINE_VH_TRACE_FEATURES=1` is ALSO set, each accepted-solo
record additionally carries `features` — the 14-d base feature vector
from `lib.value_head_features.encode_features`. The 15th feature
(leaf_delta) lives in the `delta_pred` field of the same record.

Used by `scripts/gen_b2_corpus.py` (stage 1 self-play emission). At
default env (both vars unset), `trace_accepted` is a no-op — no file
I/O, no encoder import, no module-level side effects beyond env-var
parsing on import.

Ported from `claude/competition-objective-alignment-hqNVM` commit
9d32066 with trace_solo / trace_joint removed (those depend on the
shot_validator booster, which the value-head trainer doesn't need).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_ACCEPTED_PATH: str = os.environ.get("BASELINE_ACCEPTED_TRACE", "").strip()
_ACCEPTED_ENABLED: bool = bool(_ACCEPTED_PATH)
_ACCEPTED_FILE = None
_ACCEPTED_LOAD_FAILED: bool = False

_VH_TRACE_FEATURES: bool = os.environ.get(
    "BASELINE_VH_TRACE_FEATURES", "0").strip() == "1"


def _ensure_accepted_loaded() -> bool:
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


def trace_accepted(world: Any, world_model: Any, me: int,
                   accepted: list) -> None:
    """Emit one JSON line per accepted candidate the chooser committed
    this turn. `accepted` is a list of dicts with keys: kind ('solo'),
    src_id, tgt_id, ships, angle, wait_N, eta, delta_pred. Joints are
    not included by caller (solo-only training).

    When BASELINE_VH_TRACE_FEATURES=1 is set, also emit the 14-d feature
    vector from lib.value_head_features.encode_features as `features`.

    No-op when BASELINE_ACCEPTED_TRACE is unset."""
    if not _ACCEPTED_ENABLED or not _ensure_accepted_loaded():
        return
    try:
        step = int(getattr(world, "step", 0))
        encode_features = None
        if _VH_TRACE_FEATURES and world_model is not None:
            try:
                from lib.value_head_features import encode_features as _enc
                encode_features = _enc
            except Exception:
                encode_features = None
        planets_by_id = getattr(world, "planets_by_id", {}) or {}
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
            if (encode_features is not None
                    and str(entry["kind"]) == "solo"):
                src = planets_by_id.get(int(entry["src_id"]))
                tgt = planets_by_id.get(int(entry["tgt_id"]))
                if src is not None and tgt is not None:
                    try:
                        feats = encode_features(
                            src, tgt, int(entry["ships"]),
                            int(entry["eta"]), int(me),
                            world, world_model,
                        )
                        rec["features"] = [float(v) for v in feats]
                    except Exception:
                        pass
            _ACCEPTED_FILE.write(json.dumps(rec) + "\n")
    except Exception:
        pass
