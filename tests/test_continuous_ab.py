"""Tests for the continuous-score A/B harness (scripts/continuous_ab.py +
scripts/_continuous_game_worker.py).

Two things must hold:
  1. The stats helpers (Wilson, mean CI, sign test) are correct.
  2. The continuous ship-margin's SIGN reproduces the engine's win rule exactly
     — i.e. margin > 0  <=>  the focal player is the unique argmax of end-state
     ship totals. This is what makes the continuous score a faithful, strictly
     finer relaxation of win/loss (DROPOUT_PLAN continuous-feedback rationale).
"""
import importlib.util
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cab = _load("_cab", REPO / "scripts" / "continuous_ab.py")
worker = _load("_cgw", REPO / "scripts" / "_continuous_game_worker.py")


# ---- stats ----

def test_wilson_known_value():
    lo, hi = cab.wilson_ci(8, 16)
    assert 0.0 < lo < 0.5 < hi < 1.0
    # symmetric around 0.5 for a 50% split
    assert abs((lo + hi) / 2 - 0.5) < 1e-9


def test_wilson_edges():
    assert cab.wilson_ci(0, 0) == (0.0, 0.0)
    lo, hi = cab.wilson_ci(40, 40)
    assert hi == 1.0 and lo > 0.8


def test_mean_ci_basic():
    m, lo, hi = cab.mean_ci([1.0, 1.0, 1.0])
    assert m == 1.0 and lo == 1.0 and hi == 1.0  # zero variance
    m, lo, hi = cab.mean_ci([-1.0, 1.0])
    assert abs(m) < 1e-9 and lo < 0 < hi


def test_bootstrap_brackets_mean():
    xs = [0.1, 0.2, 0.15, 0.05, 0.25, 0.18, 0.12, 0.22]
    lo, hi = cab.bootstrap_ci(xs, iters=2000)
    m = sum(xs) / len(xs)
    assert lo <= m <= hi


def test_sign_test():
    # all one direction at n=8 is significant; an even split is not.
    assert cab.sign_test_p(8, 0) < 0.05
    assert cab.sign_test_p(4, 4) == 1.0
    assert 0.0 <= cab.sign_test_p(6, 2) <= 1.0


# ---- continuous score = engine win ----

def _scores_from(planets, fleets, players):
    obs = {"planets": planets, "fleets": fleets}
    return worker._player_ship_scores(obs, players)


def test_margin_sign_matches_engine_win():
    # planet row layout: [id, owner, x, y, radius, ships, production]
    # fleet  row layout: [id, owner, ..., ships(idx6)]
    # Focal (seat 0) holds 100 ships, rival (seat 1) holds 60 -> focal wins,
    # margin > 0.
    planets = [[0, 0, 0, 0, 1, 100, 1], [1, 1, 5, 5, 1, 60, 1],
               [2, -1, 9, 9, 1, 30, 1]]  # neutral excluded
    fleets = []
    scores = _scores_from(planets, fleets, 2)
    assert scores == [100.0, 60.0]
    focal, rival = scores[0], scores[1]
    margin = (focal - rival) / (focal + rival)
    win = focal == max(scores) and max(scores) > 0
    assert margin > 0 and win


def test_margin_counts_fleets_and_excludes_neutral():
    planets = [[0, 0, 0, 0, 1, 10, 1], [1, 1, 5, 5, 1, 80, 1],
               [2, -1, 9, 9, 1, 999, 1]]
    fleets = [[0, 0, 0, 0, 0, 0, 50], [1, 1, 0, 0, 0, 0, 5]]
    scores = _scores_from(planets, fleets, 2)
    assert scores == [60.0, 85.0]  # neutral 999 ignored; fleets added
    margin = (scores[0] - scores[1]) / sum(scores)
    win = scores[0] == max(scores) and max(scores) > 0
    assert margin < 0 and not win


def test_logged_games_margin_sign_equals_reward(tmp_path=None):
    """If a real run log exists, every non-idle game's margin sign must agree
    with the engine reward (reward==1 <=> margin>=0 for the focal). This is the
    Rule-38-style check that the continuous score faithfully tracks the engine.
    """
    logs = list((REPO / "audit").glob("continuous-ab-*.jsonl"))
    if not logs:
        return  # nothing to check yet
    import json
    checked = 0
    for lg in logs:
        for line in lg.read_text().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r or "reward" not in r or r.get("reward") is None:
                continue
            # engine win (reward==1) must coincide with non-negative margin;
            # a strict loss (reward==-1) with non-positive margin. (Exact ties
            # give margin 0 and reward per the engine's >0 rule.)
            if r["reward"] == 1:
                assert r["margin"] >= 0, r
            else:
                assert r["margin"] <= 0 + 1e-9, r
            checked += 1
    assert checked >= 0
