"""LR_ONE_CAPTURE pivot (2026-06-18): commit at most ONE capture per turn.

The pivot caps each turn's expansion at a single high-confidence capture -- the
best planet we can take AND hold (sized to out-gun the visible enemy force that
can reach it) -- instead of the greedy stacking several thin captures that flip
straight back (the recapture churn). Defenses are NOT capped.

This test isolates the COMMIT cap from the 2-ply re-pick by turning the 2-ply
off: under the gate the 2-ply step only chooses between the committed plan and
doing nothing -- it can never ADD a capture -- so a committed plan with <=1
capture guarantees a final move with <=1 capture.

The board uses angular_velocity=0 so each launch [src, angle, ships] maps back
to its target planet by direction (atan2); distinct non-own targets == captures.
"""
import importlib.util
import math
import os

import pytest
from kaggle_environments import make

_MAIN = os.path.join(os.path.dirname(__file__), "..", "agents",
                     "least_resistance", "main.py")

# omega=0 board: two distant sources, four solo-capturable neutrals, one far
# (unattractive) enemy planet so the modelled player count resolves to 2.
_PLANETS = [
    [0, 0,  40.0,  40.0, 1.0, 200, 1],   # my source A
    [1, 0, 160.0, 160.0, 1.0, 200, 1],   # my source B
    [2, -1, 58.0,  40.0, 1.0,   5, 5],   # neutral near A
    [3, -1, 40.0,  58.0, 1.0,   5, 5],   # neutral near A
    [4, -1, 142.0, 160.0, 1.0,  5, 5],   # neutral near B
    [5, -1, 160.0, 142.0, 1.0,  5, 5],   # neutral near B
    [6, 1, 185.0,  40.0, 1.0,  80, 1],   # far enemy
]


def _load_agent_module():
    """Load least_resistance/main.py by path -- avoids the kaggle_environments
    `agents` package-name collision -- as a fresh module."""
    spec = importlib.util.spec_from_file_location("lr_main_one_capture_test", _MAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _obs():
    env = make("orbit_wars", debug=False)
    env.reset(num_agents=2)
    obs = dict(env.state[0].observation)          # real schema (all keys present)
    obs.update(player=0, step=5, fleets=[], comets=[], comet_planet_ids=[],
               angular_velocity=0.0,
               planets=[list(p) for p in _PLANETS],
               initial_planets=[list(p) for p in _PLANETS])
    return obs, dict(env.configuration)


def _capture_targets(emit, me=0):
    """Distinct non-own planets the launches are aimed at (== captures made).
    Exact for an omega=0 board where the emit angle points straight at the target."""
    by_id = {p[0]: p for p in _PLANETS}
    hit = set()
    for src_id, ang, _ships in emit:
        sx, sy = by_id[src_id][2], by_id[src_id][3]
        best, berr = None, 1e9
        for p in _PLANETS:
            if p[0] == src_id:
                continue
            d = abs(math.atan2(p[3] - sy, p[2] - sx) - ang)
            d = min(d, 2 * math.pi - d)
            if d < berr:
                berr, best = d, p
        if best is not None and best[1] != me:
            hit.add(best[0])
    return hit


def test_one_capture_caps_a_multi_capture_turn_to_one(monkeypatch):
    for k in list(os.environ):
        if k.startswith("LR_"):                   # clean lever environment
            monkeypatch.delenv(k, raising=False)
    lr = _load_agent_module()
    if not lr._ORBIT_OK:
        pytest.skip("orbit_lite evaluator unavailable (no torch); the cap is "
                    "evaluator-independent but the multi-capture baseline needs it")
    monkeypatch.setattr(lr, "TWOPLY", False)      # isolate the commit cap
    obs, cfg = _obs()

    off = _capture_targets(lr.agent(obs, cfg))    # gate OFF -> greedy stacks
    monkeypatch.setenv("LR_ONE_CAPTURE", "1")
    on = _capture_targets(lr.agent(obs, cfg))     # gate ON  -> single capture

    assert len(off) >= 2, f"baseline should stack multiple captures, got {sorted(off)}"
    assert len(on) == 1, f"LR_ONE_CAPTURE must commit exactly one capture, got {sorted(on)}"
