"""Map a turn-0 observation (or a replay) to one of the 32 panel archetypes.

The seed panel taxonomy lives in ``data/seed_panel_128.json`` (built by
``scripts/build_seed_panel.py``). It stratifies on three axes:
``total_production`` × ``rotating_share`` × ``size_split``. Bin edges are
empirical percentiles from a 10K-seed pool, persisted in the JSON.

This module reuses those edges to label ARBITRARY observations — e.g.
turn-0 obs from a downloaded replay whose seed isn't in our panel.

Two entry points:
- ``archetype_of_obs(obs_or_dict)`` — accepts either a kaggle_environments
  observation (anything that exposes ``planets`` / ``angular_velocity``)
  or a plain dict with the same keys.
- ``archetype_of_replay(replay, focal_idx=0)`` — accepts a KE-format
  replay dict (``replay["steps"][0][focal_idx]["observation"]``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .orbit import is_orbiting

_PANEL_JSON = Path(__file__).resolve().parents[1] / "data" / "seed_panel_128.json"
_panel = json.loads(_PANEL_JSON.read_text())

_BIN_EDGES: dict[str, list[float]] = _panel["bin_edges"]
_NAMES: dict[str, list[str]] = _panel["archetype_names"]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _features_from_planets(planets: list, angular_velocity: float) -> dict[str, float]:
    rot = [p for p in planets if is_orbiting(p)]
    stat = [p for p in planets if not is_orbiting(p)]
    return {
        "total_production": float(sum(p[6] for p in planets)),
        "rotating_share": (len(rot) / len(planets)) if planets else 0.0,
        "size_split": _mean([p[4] for p in rot]) - _mean([p[4] for p in stat]),
        "angular_velocity": float(angular_velocity),
        "n_planets": len(planets),
    }


def _bin_index(value: float, edges: list[float]) -> int:
    for i, e in enumerate(edges):
        if value < e:
            return i
    return len(edges)


def archetype_of_features(features: dict[str, float]) -> str:
    """Given a feature dict (matching ``_features_from_planets`` output),
    return the archetype name string used in the panel.
    """
    prod_bin = _bin_index(features["total_production"], _BIN_EDGES["total_production"])
    rot_bin = _bin_index(features["rotating_share"], _BIN_EDGES["rotating_share"])
    split_bin = _bin_index(features["size_split"], _BIN_EDGES["size_split"])
    return (
        f"{_NAMES['total_production'][prod_bin]}__"
        f"{_NAMES['rotating_share'][rot_bin]}__"
        f"{_NAMES['size_split'][split_bin]}"
    )


def _resolve_planets_angular(obs: Any) -> tuple[list, float]:
    """Accept either a Struct-like obs (kaggle_environments) or a plain dict."""
    if isinstance(obs, dict):
        return obs.get("planets", []), float(obs.get("angular_velocity", 0.0))
    return list(getattr(obs, "planets", [])), float(getattr(obs, "angular_velocity", 0.0))


def archetype_of_obs(obs: Any) -> str:
    planets, omega = _resolve_planets_angular(obs)
    return archetype_of_features(_features_from_planets(planets, omega))


def archetype_of_replay(replay: dict, focal_idx: int = 0) -> str:
    """Replay format: KE schema where ``replay["steps"][0]`` is a list of
    per-seat dicts with ``observation``. Falls back gracefully if the
    schema is the ``tournament._build_replay()`` flat shape.
    """
    steps = replay.get("steps") or []
    if not steps:
        raise ValueError("empty replay")
    s0 = steps[0]
    if isinstance(s0, list):
        # KE raw format: list of seat dicts
        obs = s0[focal_idx].get("observation", {})
        return archetype_of_obs(obs)
    if isinstance(s0, dict) and "planets" in s0:
        # tournament._build_replay flat shape
        return archetype_of_obs(s0)
    raise ValueError(f"unknown replay step shape: {type(s0).__name__}")


def features_of_replay(replay: dict, focal_idx: int = 0) -> dict[str, float]:
    """Convenience: same input as ``archetype_of_replay`` but returns the
    feature dict (so callers can inspect WHY a replay landed in its bin).
    """
    steps = replay.get("steps") or []
    if not steps:
        raise ValueError("empty replay")
    s0 = steps[0]
    if isinstance(s0, list):
        obs = s0[focal_idx].get("observation", {})
        planets, omega = _resolve_planets_angular(obs)
    elif isinstance(s0, dict) and "planets" in s0:
        planets, omega = _resolve_planets_angular(s0)
    else:
        raise ValueError(f"unknown replay step shape: {type(s0).__name__}")
    return _features_from_planets(planets, omega)
