"""LR_CONFIDENT (2026-06-18): commit only high-confidence actions.

Of the launches we are about to play, keep a capture only if we can HOLD the
planet we take -- the garrison we land with (ships sent minus the target's
defenders) is at least the enemy force that can still reach it. Safe grabs
(nothing can reach them) and reinforcements of our own planets are always kept;
thin contested grabs that would flip back are dropped. No count cap.

Tests `_keep_confident_launches` directly (omega=0 so the emit angle is a straight
bearing the target-matcher resolves exactly; targets are non-collinear from the
source).
"""
import importlib.util
import math
import os

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

_MAIN = os.path.join(os.path.dirname(__file__), "..", "agents",
                     "least_resistance", "main.py")


def _load():
    spec = importlib.util.spec_from_file_location("lr_main_confident_test", _MAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#  P0 (mine, source) ; A neutral & safe (no enemy in reach) ; B enemy with a big
#  enemy neighbour E in reach (so a thin grab of B can't hold) ; P_own (mine).
_PLANETS = [
    Planet(0, 0, 20.0, 20.0, 1.0, 100, 1),    # my source
    Planet(4, 0, 35.0, 30.0, 1.0,   3, 5),    # my own planet (reinforce target)
    Planet(1, -1, 40.0, 20.0, 1.0,  5, 5),    # neutral, safe
    Planet(2, 1, 20.0, 80.0, 1.0,   5, 5),    # enemy, contested
    Planet(3, 1, 25.0, 85.0, 1.0,  60, 1),    # big enemy next to B (the threat)
]
_BY_ID = {int(p.id): p for p in _PLANETS}


def _ang(src, tgt):
    return math.atan2(tgt.y - src.y, tgt.x - src.x)


def _call(lr, move):
    return lr._keep_confident_launches(move, _PLANETS, [], _BY_ID, 0,
                                       frozenset(), {}, 0.0)


def test_drops_thin_contested_keeps_safe_grab():
    lr = _load()
    P0, A, B = _BY_ID[0], _BY_ID[1], _BY_ID[2]
    move = [
        [0, _ang(P0, A), 10],   # neutral A, no threat -> can hold -> KEEP
        [0, _ang(P0, B), 7],    # enemy B with E (60) next to it -> 7-5=2 < ~30 -> DROP
    ]
    kept = _call(lr, move)
    assert len(kept) == 1, f"thin contested grab should be dropped, kept {kept}"
    assert abs(float(kept[0][1]) - _ang(P0, A)) < 1e-6, "the safe grab must be the one kept"


def test_keeps_strong_contested_capture():
    lr = _load()
    P0, B = _BY_ID[0], _BY_ID[2]
    # Send enough to hold B against E: 40 - 5 = 35 surplus >= 0.5*60 = 30 threat.
    move = [[0, _ang(P0, B), 40]]
    kept = _call(lr, move)
    assert len(kept) == 1, "a contested capture sized to hold must be kept"


def test_keeps_reinforcement_of_own_planet():
    lr = _load()
    P0, P_own = _BY_ID[0], _BY_ID[4]
    move = [[0, _ang(P0, P_own), 5]]   # reinforce our own planet -> always kept
    kept = _call(lr, move)
    assert len(kept) == 1, "reinforcing our own planet is always high-confidence"
