"""Unit tests for v7_minimax — the maximin picker + obs-swap path.

Validates the algorithmic correctness of the maximin choice independent
of any forward simulation (which is integration-tested elsewhere).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_v7():
    spec = importlib.util.spec_from_file_location(
        "v7_minimax", REPO / "agents" / "v7_minimax" / "main.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def v7():
    return _load_v7()


# ---------------------------------------------------------------------------
# _maximin_pick — the core picker
# ---------------------------------------------------------------------------


def test_maximin_pick_simple(v7):
    """Standard 2x2 payoff:
       opp_a opp_b
    us_a   5     1       worst(us_a) = 1
    us_b   3     2       worst(us_b) = 2   <- maximin
    """
    P = [[5.0, 1.0], [3.0, 2.0]]
    unfilled = [[False, False], [False, False]]
    assert v7._maximin_pick(P, unfilled) == 1


def test_maximin_pick_us_a_wins(v7):
    """When row 0's worst > row 1's worst, pick row 0."""
    P = [[3.0, 2.0], [5.0, 1.0]]
    unfilled = [[False, False], [False, False]]
    assert v7._maximin_pick(P, unfilled) == 0


def test_maximin_pick_tie_prefers_lower_index(v7):
    """Equal worst → prefer row 0 (v3 incumbent)."""
    P = [[2.0, 2.0], [2.0, 2.0]]
    unfilled = [[False, False], [False, False]]
    assert v7._maximin_pick(P, unfilled) == 0


def test_maximin_pick_negative_payoffs(v7):
    """Negative payoffs work correctly — pick the least bad worst."""
    P = [[-1.0, -10.0], [-5.0, -3.0]]
    unfilled = [[False, False], [False, False]]
    # worst(0) = -10, worst(1) = -5 → row 1 wins
    assert v7._maximin_pick(P, unfilled) == 1


def test_maximin_pick_partial_row_falls_back(v7):
    """If row 1 has all columns unfilled, row 0 wins by default
    (row 1's worst = -inf)."""
    P = [[3.0, 5.0], [0.0, 0.0]]
    unfilled = [[False, False], [True, True]]
    assert v7._maximin_pick(P, unfilled) == 0


def test_maximin_pick_partial_row_uses_evaluated_only(v7):
    """Row 1 has only col 0 evaluated; worst(1) = P[1][0] = 4 > worst(0) = 3."""
    P = [[3.0, 5.0], [4.0, 999.0]]
    unfilled = [[False, False], [False, True]]
    # worst(0) = 3; worst(1) = 4 (only col 0 evaluated). Row 1 wins.
    assert v7._maximin_pick(P, unfilled) == 1


def test_maximin_pick_single_row(v7):
    """Degenerate N=1 case."""
    P = [[1.0, 2.0, 3.0]]
    unfilled = [[False, False, False]]
    assert v7._maximin_pick(P, unfilled) == 0


def test_maximin_pick_single_column(v7):
    """Degenerate M=1 case → maximin = max."""
    P = [[1.0], [3.0], [2.0]]
    unfilled = [[False], [False], [False]]
    assert v7._maximin_pick(P, unfilled) == 1


# ---------------------------------------------------------------------------
# _drop_smallest — drop the smallest-ship launch
# ---------------------------------------------------------------------------


def test_drop_smallest_empty(v7):
    assert v7._drop_smallest([]) == []


def test_drop_smallest_single(v7):
    """One launch → drop it → empty action."""
    assert v7._drop_smallest([[1, 0.5, 10]]) == []


def test_drop_smallest_picks_min(v7):
    a = [[1, 0.5, 10], [2, 1.0, 3], [3, 1.5, 7]]
    # smallest ships=3 at idx 1 → drop it
    assert v7._drop_smallest(a) == [[1, 0.5, 10], [3, 1.5, 7]]


def test_drop_smallest_ties_pick_earlier(v7):
    """Two launches tied at smallest → drop the EARLIER one (σ-deterministic)."""
    a = [[1, 0.5, 5], [2, 1.0, 5], [3, 1.5, 10]]
    # both 5s tied; idx 0 wins; drop it
    assert v7._drop_smallest(a) == [[2, 1.0, 5], [3, 1.5, 10]]


# ---------------------------------------------------------------------------
# _swap_obs_player — obs.player swap for opp-POV invocation
# ---------------------------------------------------------------------------


def test_swap_obs_dict(v7):
    obs = {"player": 0, "planets": [[0, 0, 0, 0, 0, 0, 0]], "step": 5}
    out = v7._swap_obs_player(obs, opp_id=1)
    assert out["player"] == 1
    assert out["planets"] == obs["planets"]  # shared
    assert out["step"] == 5
    assert obs["player"] == 0  # original NOT mutated


def test_swap_obs_object(v7):
    """Works on object-style obs (some env paths return Munch-like)."""
    class _O:
        pass
    o = _O()
    o.player = 0
    o.planets = [[0, 0, 0, 0, 0, 0, 0]]
    o.step = 3
    o.angular_velocity = 0.04
    out = v7._swap_obs_player(o, opp_id=1)
    assert out["player"] == 1
    assert out["planets"] == o.planets
    assert out["step"] == 3
    assert o.player == 0  # original NOT mutated


# ---------------------------------------------------------------------------
# _detect_num_players
# ---------------------------------------------------------------------------


def test_detect_num_players_2p(v7):
    planets = [[0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0], [2, -1, 0, 0, 0, 0, 0]]
    assert v7._detect_num_players(planets) == 2


def test_detect_num_players_4p(v7):
    planets = [[0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0],
               [2, 2, 0, 0, 0, 0, 0], [3, 3, 0, 0, 0, 0, 0]]
    assert v7._detect_num_players(planets) == 4
