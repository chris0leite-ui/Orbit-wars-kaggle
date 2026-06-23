"""Unit tests for the contested-only one-capture-per-round cap (LR_ONE_CAPTURE).

Cap counts only CONTESTED attacks (enemy-held planets, or neutrals a rival is
racing us for). Uncontested neutral grabs (open expansion) and defensive regroup
flow freely. OFF -> unchanged (byte-identical).
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
# id0,id1 mine; id2 enemy @(90,10); id3 enemy @(90,90); id4 neutral @(10,50) FAR
# from both enemies (uncontested); id5 neutral @(88,10) adjacent to enemy id2
# (contested). Well-separated directions from id0/id1 so the cone match is clean.
OBS = {
    "player": 0,
    "fleets": [],
    "planets": [
        [0, 0, 10.0, 10.0, 2.0, 50.0, 3.0],
        [1, 0, 12.0, 10.0, 2.0, 40.0, 2.0],
        [2, 1, 90.0, 10.0, 2.0, 30.0, 3.0],
        [3, 1, 90.0, 90.0, 2.0, 28.0, 3.0],
        [4, -1, 10.0, 50.0, 2.0, 20.0, 2.0],
        [5, -1, 88.0, 10.0, 2.0, 10.0, 2.0],
    ],
}
PL = {int(p[0]): p for p in OBS["planets"]}


def _ang(src, dst):
    return math.atan2(dst[1] - src[1], dst[0] - src[0])


def _row(i):
    return PL[i]


def test_classifier_enemy_and_contested_neutral_vs_open_grab():
    fl, ps, me = OBS["fleets"], OBS["planets"], 0
    assert M._contested_attack(_row(2), 10.0, ps, fl, me) is True   # enemy-held
    assert M._contested_attack(_row(5), 10.0, ps, fl, me) is True   # neutral by enemy id2
    assert M._contested_attack(_row(4), 5.0, ps, fl, me) is False   # open neutral, far


def test_off_path_is_identity():
    os.environ["LR_ONE_CAPTURE"] = "0"
    action = [[0, _ang((10, 10), (90, 10)), 20], [1, _ang((12, 10), (90, 90)), 15]]
    assert M._cap_emit([list(a) for a in action], OBS, 0) == action


def test_two_enemy_attacks_keep_biggest():
    os.environ["LR_ONE_CAPTURE"] = "1"
    atk2 = [0, _ang((10, 10), (90, 10)), 25]      # -> enemy id2 (bigger)
    atk3 = [1, _ang((12, 10), (90, 90)), 18]      # -> enemy id3
    out = M._cap_emit([atk2, atk3], OBS, 0)
    assert atk2 in out and atk3 not in out
    assert len(out) == 1


def test_uncontested_neutral_grab_is_free():
    os.environ["LR_ONE_CAPTURE"] = "1"
    atk2 = [0, _ang((10, 10), (90, 10)), 25]      # -> enemy id2 (contested)
    grab4 = [1, _ang((12, 10), (10, 50)), 15]     # -> open neutral id4 (free)
    out = M._cap_emit([atk2, grab4], OBS, 0)
    assert atk2 in out and grab4 in out           # both kept: 1 attack + free expansion
    assert len(out) == 2


def test_defense_and_expansion_uncapped():
    os.environ["LR_ONE_CAPTURE"] = "1"
    d = [0, _ang((10, 10), (12, 10)), 10]         # reinforce own id1 (defense)
    atk2 = [1, _ang((12, 10), (90, 10)), 25]      # -> enemy id2 (the one attack)
    grab4 = [0, _ang((10, 10), (10, 50)), 12]     # -> open neutral id4 (free)
    out = M._cap_emit([d, atk2, grab4], OBS, 0)
    assert d in out and grab4 in out and atk2 in out
    assert len(out) == 3


def teardown_function(_):
    os.environ.pop("LR_ONE_CAPTURE", None)
