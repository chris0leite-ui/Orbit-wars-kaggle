"""Tests for lib/mechanism.comet_aim — path-indexed lead for comet targets.

Comets follow pre-computed paths (`obs["comets"][g].paths[i]`), not the
rotation formula. comet_aim looks the path up, projects forward by ETA,
and aims at the future position. If the path runs out before the fleet
arrives, the intent is dropped.
"""

from __future__ import annotations

import math

import pytest

from lib.intent import Intent, World
from lib.mechanism import comet_aim


def _obs_with_comet(*, target_id=20, path=None, path_index=0):
    """Single-source single-comet obs.
    `path` is a list of [x, y] points; defaults to a straight line on +x."""
    if path is None:
        path = [[20.0 + i * 4.0, 50.0] for i in range(20)]   # 20 steps long
    src = [0, 0, 5.0, 5.0, 1.0, 100, 1]
    # Comets appear in `planets` AND in `comet_planet_ids`; the planet entry
    # uses the current path point as its (x, y).
    cur_x, cur_y = path[path_index]
    comet_planet = [target_id, -1, cur_x, cur_y, 1.0, 30, 1]
    return {
        "player": 0,
        "planets": [src, comet_planet],
        "angular_velocity": 0.04,
        "comet_planet_ids": [target_id],
        "step": 50,
        "comets": [
            {
                "planet_ids": [target_id],
                "paths": [path],
                "path_index": path_index,
            },
        ],
    }


def _world(obs):
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# Pass-through cases
# ---------------------------------------------------------------------------


def test_comet_aim_passes_through_when_no_comets_in_obs():
    """If `comet_planet_ids` is empty, comet_aim is a fast no-op."""
    obs = {
        "player": 0,
        "planets": [[0, 0, 5.0, 5.0, 1.0, 100, 1], [1, -1, 30.0, 30.0, 1.0, 5, 1]],
        "angular_velocity": 0.04,
        "comet_planet_ids": [],
        "step": 1,
    }
    intent = Intent(src_id=0, target_id=1, ships=10)
    out = comet_aim([intent], World.from_obs(obs))
    assert out[0].aim_angle is None   # untouched


def test_comet_aim_passes_through_non_comet_target():
    obs = _obs_with_comet()
    obs["planets"].append([2, -1, 80.0, 80.0, 1.0, 5, 1])   # regular neutral
    intent = Intent(src_id=0, target_id=2, ships=10)        # NOT the comet
    out = comet_aim([intent], _world(obs))
    assert out[0].aim_angle is None


def test_comet_aim_does_not_overwrite_existing_aim_angle():
    obs = _obs_with_comet()
    intent = Intent(src_id=0, target_id=20, ships=10, aim_angle=1.234)
    out = comet_aim([intent], _world(obs))
    assert out[0].aim_angle == 1.234


# ---------------------------------------------------------------------------
# Path-indexed lead populates aim_angle
# ---------------------------------------------------------------------------


def test_comet_aim_populates_aim_at_future_position():
    """ETA is computed from current target xy + fleet speed; aim is atan2 to
    `path[path_index + eta]`."""
    # Path: long enough that future_index is in-bounds for any reasonable eta.
    path = [[20.0 + i * 4.0, 50.0] for i in range(100)]
    obs = _obs_with_comet(path=path, path_index=0)
    intent = Intent(src_id=0, target_id=20, ships=1)   # 1-ship fleet → speed 1
    out = comet_aim([intent], _world(obs))
    assert out[0].aim_angle is not None
    # The aim should NOT equal atan2 to the current comet position (path[0]),
    # because the comet has moved forward by eta steps.
    cur = path[0]
    naive = math.atan2(cur[1] - 5.0, cur[0] - 5.0)
    assert out[0].aim_angle != pytest.approx(naive)


# ---------------------------------------------------------------------------
# Path-runs-out: drop the intent
# ---------------------------------------------------------------------------


def test_comet_aim_drops_intent_when_comet_exits_before_arrival():
    """5-step path, fleet ETA much greater than 5 → comet has exited the
    board by the time fleet arrives → drop."""
    short_path = [[20.0, 50.0], [24.0, 50.0], [28.0, 50.0], [32.0, 50.0], [36.0, 50.0]]
    obs = _obs_with_comet(path=short_path, path_index=0)
    intent = Intent(src_id=0, target_id=20, ships=1)
    out = comet_aim([intent], _world(obs))
    assert out == []


def test_comet_aim_drops_intent_when_path_index_already_at_end():
    """path_index at last valid index → any non-zero ETA pushes future_index
    past the end, comet exits before arrival, drop."""
    path = [[20.0 + i * 4.0, 50.0] for i in range(5)]   # indices 0..4
    obs = _obs_with_comet(path=path, path_index=4)
    intent = Intent(src_id=0, target_id=20, ships=1)    # ETA >> 1 from src
    out = comet_aim([intent], _world(obs))
    assert out == []


# ---------------------------------------------------------------------------
# Defensive: malformed obs / unknown comet
# ---------------------------------------------------------------------------


def test_comet_aim_defensive_when_comet_id_missing_from_groups():
    """`comet_planet_ids` lists an id but `comets` doesn't contain it
    (theoretically possible mid-tick) — pass-through, don't drop."""
    obs = _obs_with_comet()
    # Mutate comets so the planet_id mapping is stale.
    obs["comets"][0]["planet_ids"] = [99]
    intent = Intent(src_id=0, target_id=20, ships=10)
    out = comet_aim([intent], _world(obs))
    assert len(out) == 1
    assert out[0].aim_angle is None
