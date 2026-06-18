"""LR_ONE_ACTION (2026-06-18): streamline each round into ONE coordinated action.

Watching a replay, the shipped agent expanded strongly (~14 planets) then
collapsed because its force was fragmented into many small fleets (measured: 38%
of active turns launch at >=2 distinct targets). LR_ONE_ACTION commits ONLY the
single best coordinated capture each round (sized to overwhelm), launching nothing
else -- so the played move targets exactly one planet.

Board: angular_velocity=0 so each launch maps to its target by direction; 100x100
board, sun at (50,50) r10, planets clear of the sun and non-collinear. For the OFF
baseline we neutralize the commit value-floor (LR_ROI_FLOOR very negative) and turn
the 2-ply off so the shipped greedy commits a multi-target move; the ON path
bypasses both (it just plays the single best strike).
"""
import importlib.util
import math
import os

import pytest
from kaggle_environments import make

_MAIN = os.path.join(os.path.dirname(__file__), "..", "agents",
                     "least_resistance", "main.py")

# three corner sources; three cheap enemies + two neutrals, none collinear with a
# source (so a launch maps unambiguously to its target by bearing).
_PLANETS = [
    [0, 0, 15.0, 15.0, 1.0, 120, 1],
    [1, 0, 85.0, 85.0, 1.0, 120, 1],
    [2, 0, 15.0, 85.0, 1.0, 120, 1],
    [3, 1, 30.0, 20.0, 1.0,   1, 9],
    [4, 1, 70.0, 80.0, 1.0,   1, 9],
    [5, 1, 20.0, 70.0, 1.0,   1, 9],
    [6, -1, 20.0, 30.0, 1.0,  3, 6],
    [7, -1, 80.0, 70.0, 1.0,  3, 6],
]


def _load_agent_module():
    spec = importlib.util.spec_from_file_location("lr_main_one_action_test", _MAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _obs():
    env = make("orbit_wars", debug=False)
    env.reset(num_agents=2)
    obs = dict(env.state[0].observation)
    obs.update(player=0, step=5, fleets=[], comets=[], comet_planet_ids=[],
               angular_velocity=0.0,
               planets=[list(p) for p in _PLANETS],
               initial_planets=[list(p) for p in _PLANETS])
    return obs, dict(env.configuration)


def _distinct_targets(emit, me=0):
    """Distinct non-own planets the launches are aimed at (the targets we act on)."""
    by_id = {p[0]: p for p in _PLANETS}
    hit = set()
    for src_id, ang, _ships in emit:
        sx, sy = by_id[src_id][2], by_id[src_id][3]
        best, berr = None, 1e9
        for p in _PLANETS:
            if p[0] == src_id or p[1] == me:
                continue
            d = abs(math.atan2(p[3] - sy, p[2] - sx) - ang)
            d = min(d, 2 * math.pi - d)
            if d < berr:
                berr, best = d, p
        if best is not None:
            hit.add(best[0])
    return hit


def test_one_action_streamlines_to_a_single_target(monkeypatch):
    for k in list(os.environ):
        if k.startswith("LR_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LR_TWOPLY", "0")        # played move = committed plan
    monkeypatch.setenv("LR_ROI_FLOOR", "-1e9")  # commit every affordable capture (multi-target)
    lr = _load_agent_module()
    if not lr._ORBIT_OK:
        pytest.skip("orbit_lite evaluator unavailable (no torch)")
    obs, cfg = _obs()

    off = _distinct_targets(lr.agent(obs, cfg))         # spreads across targets
    monkeypatch.setenv("LR_ONE_ACTION", "1")
    on = _distinct_targets(lr.agent(obs, cfg))          # streamlined to one

    assert len(off) >= 2, f"baseline should spread across targets, got {sorted(off)}"
    assert len(on) == 1, f"LR_ONE_ACTION must keep exactly one target, got {sorted(on)}"
