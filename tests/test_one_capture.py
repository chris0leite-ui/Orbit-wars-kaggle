"""Unit tests for the one-capture-per-round cap (LR_ONE_CAPTURE).

_cap_emit is the final-move enforcement: keep every non-offensive launch
(defense / reinforcement of our own planets) plus the single highest-ship
offensive target; drop other offensive targets. OFF -> unchanged (byte-identical).
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agents" / "producer"))


def _load():
    p = REPO / "agents" / "least_resistance" / "main.py"
    spec = importlib.util.spec_from_file_location("_oc_main", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


M = _load()

# Planets: [id, owner, x, y, radius, ships, production]. me = player 0.
# id0 mine @ (10,10); id1 mine @ (12,10); id2 enemy @ (90,10); id3 neutral @ (10,90).
OBS = {
    "player": 0,
    "planets": [
        [0, 0, 10.0, 10.0, 2.0, 50.0, 3.0],
        [1, 0, 12.0, 10.0, 2.0, 40.0, 2.0],
        [2, 1, 90.0, 10.0, 2.0, 30.0, 3.0],
        [3, -1, 10.0, 90.0, 2.0, 20.0, 2.0],
    ],
}


def _ang(src, dst):
    return math.atan2(dst[1] - src[1], dst[0] - src[0])


def test_off_path_is_identity():
    os.environ["LR_ONE_CAPTURE"] = "0"
    action = [[0, _ang((10, 10), (90, 10)), 20], [1, _ang((12, 10), (10, 90)), 15]]
    assert M._cap_emit([list(a) for a in action], OBS, 0) == action


def test_caps_to_one_offensive_target_keeps_biggest():
    os.environ["LR_ONE_CAPTURE"] = "1"
    # Two offensive launches: 20 ships at enemy id2, 15 ships at neutral id3.
    a_enemy = [0, _ang((10, 10), (90, 10)), 20]      # -> id2 (more ships)
    a_neutral = [1, _ang((12, 10), (10, 90)), 15]    # -> id3
    out = M._cap_emit([a_enemy, a_neutral], OBS, 0)
    assert a_enemy in out and a_neutral not in out   # keep the larger attack only
    assert len(out) == 1


def test_defense_launches_are_uncapped():
    os.environ["LR_ONE_CAPTURE"] = "1"
    # Reinforce own planet id1 (from id0) + one attack at id2 + one attack at id3.
    d = [0, _ang((10, 10), (12, 10)), 10]            # -> own planet id1 (defense)
    atk1 = [1, _ang((12, 10), (90, 10)), 25]         # -> enemy id2 (biggest)
    atk2 = [0, _ang((10, 10), (10, 90)), 12]         # -> neutral id3
    out = M._cap_emit([d, atk1, atk2], OBS, 0)
    assert d in out                                   # defense always kept
    assert atk1 in out and atk2 not in out            # one attack (the bigger) kept
    assert len(out) == 2


def test_single_offensive_target_unchanged():
    os.environ["LR_ONE_CAPTURE"] = "1"
    # Gang-up: two sources -> same enemy target id2. One target -> unchanged.
    g1 = [0, _ang((10, 10), (90, 10)), 20]
    g2 = [1, _ang((12, 10), (90, 10)), 18]
    out = M._cap_emit([g1, g2], OBS, 0)
    assert out == [g1, g2]


def teardown_function(_):
    os.environ.pop("LR_ONE_CAPTURE", None)
