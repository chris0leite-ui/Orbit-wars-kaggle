"""LR_ONE_CAPTURE (softened pivot, 2026-06-18): cap CONTESTED (enemy) captures
to one per turn, but leave free-neutral expansion uncapped.

An earlier version capped ALL captures (free neutrals included). It under-expanded
and lost 0/4 in 4P -- out-expanded in the opening, eliminated by midgame (see
audit/2026-06-18-one-capture.md). The softened gate caps only enemy captures.

Board: angular_velocity=0 so each launch [src, angle, ships] maps to its target
by direction; 100x100 board, sun at (50,50) r10, every planet clear of the sun,
and NO target collinear with a source (so a launch maps unambiguously). We
neutralize the commit value-floor (LR_ROI_FLOOR very negative, set before import)
so EVERY affordable capture commits -- isolating the split/cap logic from the
scorer's (conservative) marginal-value threshold, which otherwise commits only
one or two captures per synthetic turn regardless of the gate.
"""
import importlib.util
import math
import os

import pytest
from kaggle_environments import make

_MAIN = os.path.join(os.path.dirname(__file__), "..", "agents",
                     "least_resistance", "main.py")

# sources in corners; three cheap high-value enemies (one near each source);
# two free neutrals. No target shares a bearing with a source.
_PLANETS = [
    [0, 0, 15.0, 15.0, 1.0, 120, 1],   # my source A
    [1, 0, 85.0, 85.0, 1.0, 120, 1],   # my source B
    [2, 0, 15.0, 85.0, 1.0, 120, 1],   # my source C
    [3, 1, 30.0, 20.0, 1.0,   1, 9],   # enemy near A
    [4, 1, 70.0, 80.0, 1.0,   1, 9],   # enemy near B
    [5, 1, 20.0, 70.0, 1.0,   1, 9],   # enemy near C
    [6, -1, 20.0, 30.0, 1.0,  3, 6],   # free neutral
    [7, -1, 80.0, 70.0, 1.0,  3, 6],   # free neutral
]


def _load_agent_module():
    """Load least_resistance/main.py by path -- avoids the kaggle_environments
    `agents` package-name collision -- as a fresh module (so module-level
    constants like ROI_FLOOR/TWOPLY are read from the current environment)."""
    spec = importlib.util.spec_from_file_location("lr_main_one_capture_test", _MAIN)
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


def _targets(emit, me=0):
    """Split the launches into the enemy and neutral planets they capture.
    Each launch maps to the non-own planet whose bearing matches the launch
    angle (exact for this omega=0, non-collinear board)."""
    by_id = {p[0]: p for p in _PLANETS}
    enemy, neutral = set(), set()
    for src_id, ang, _ships in emit:
        sx, sy = by_id[src_id][2], by_id[src_id][3]
        best, berr = None, 1e9
        for p in _PLANETS:
            if p[0] == src_id or p[1] == me:       # only captures of non-own planets
                continue
            d = abs(math.atan2(p[3] - sy, p[2] - sx) - ang)
            d = min(d, 2 * math.pi - d)
            if d < berr:
                berr, best = d, p
        if best is not None:
            (enemy if best[1] != -1 else neutral).add(best[0])
    return enemy, neutral


def test_softened_caps_enemy_captures_not_neutral_expansion(monkeypatch):
    for k in list(os.environ):
        if k.startswith("LR_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LR_TWOPLY", "0")        # isolate the commit cap from the 2-ply re-pick
    monkeypatch.setenv("LR_ROI_FLOOR", "-1e9")  # commit every affordable capture (isolate the cap)
    lr = _load_agent_module()
    if not lr._ORBIT_OK:
        pytest.skip("orbit_lite evaluator unavailable (no torch)")
    obs, cfg = _obs()

    en_off, ne_off = _targets(lr.agent(obs, cfg))     # gate OFF -> stacks every capture
    monkeypatch.setenv("LR_ONE_CAPTURE", "1")
    en_on, ne_on = _targets(lr.agent(obs, cfg))       # gate ON -> caps enemy captures only

    assert len(en_off) >= 2, f"baseline should take multiple enemy planets, got {sorted(en_off)}"
    assert len(en_on) <= 1, f"gate must cap enemy captures to one, got {sorted(en_on)}"
    assert len(ne_on) >= 2, f"free-neutral expansion must NOT be capped, got {sorted(ne_on)}"
    assert ne_on == ne_off, f"neutral expansion should be unchanged by the gate: {sorted(ne_off)} vs {sorted(ne_on)}"
