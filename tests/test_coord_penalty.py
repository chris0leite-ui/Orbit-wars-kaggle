"""Unit tests for the wave-attack coordination BONUS (v2).

v1 used a coord PENALTY (subtract α·spread²) which biased the chooser
AWAY from joints. v2 replaces it with a per-step cohort-concentration
BONUS (add α·Σ cohort² / Σ cohort) which rewards combat-rule-1
stacking — same-step arrivals from multiple sources at the same target.

The bonus lives at agents/baseline/chooser_trajectory._coord_bonus and
is applied inside score_candidate_v4_joint at leaf-scoring time. These
tests exercise the pure-math function directly so they avoid the cost
of spinning up a fast_sim rollout.
"""
from __future__ import annotations

import os
import contextlib

from agents.baseline.chooser_trajectory import _coord_bonus


@contextlib.contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# `launches` is a list of (src, tgt, ships, angle, wait_N) tuples; the
# tests only need the ships slot (index 2) so pass None for src/tgt etc.
def _launch(ships):
    return (None, None, int(ships), 0.0, 0)


def test_coord_zero_when_flag_off():
    with _env(BASELINE_COORD_BONUS=None):
        assert _coord_bonus([_launch(30), _launch(30)], [10, 10]) == 0.0


def test_coord_zero_when_alpha_zero():
    with _env(BASELINE_COORD_BONUS="1", BASELINE_COORD_ALPHA="0"):
        assert _coord_bonus([_launch(30), _launch(30)], [10, 10]) == 0.0


def test_coord_zero_with_fewer_than_two_legs():
    """Solo emits use this function via the v4 path. Single leg → 0."""
    with _env(BASELINE_COORD_BONUS="1", BASELINE_COORD_ALPHA="0.5"):
        assert _coord_bonus([], []) == 0.0
        assert _coord_bonus([_launch(50)], [15]) == 0.0


def test_coord_same_step_two_legs_equal():
    """2×30 ships landing step 12 → cohort=60, conc=60, bonus=α·60."""
    with _env(BASELINE_COORD_BONUS="1", BASELINE_COORD_ALPHA="0.5"):
        got = _coord_bonus([_launch(30), _launch(30)], [12, 12])
    assert abs(got - 0.5 * 60.0) < 1e-9


def test_coord_staggered_two_legs_lower_than_same_step():
    """2×30 ships at different steps → two cohorts of 30, conc=30."""
    with _env(BASELINE_COORD_BONUS="1", BASELINE_COORD_ALPHA="0.5"):
        same = _coord_bonus([_launch(30), _launch(30)], [12, 12])
        stag = _coord_bonus([_launch(30), _launch(30)], [10, 18])
    assert abs(stag - 0.5 * 30.0) < 1e-9
    assert same > stag  # core property: same-step beats staggered


def test_coord_three_legs_same_step_scales():
    """3×30 ships same step → cohort=90, conc=90, bonus=α·90."""
    with _env(BASELINE_COORD_BONUS="1", BASELINE_COORD_ALPHA="0.5"):
        got = _coord_bonus(
            [_launch(30), _launch(30), _launch(30)], [12, 12, 12],
        )
    assert abs(got - 0.5 * 90.0) < 1e-9


def test_coord_three_legs_split_two_plus_one():
    """3×30 ships: 2 same step + 1 stagger → cohorts {60,30}.
       conc = (60² + 30²)/(60+30) = 4500/90 = 50."""
    with _env(BASELINE_COORD_BONUS="1", BASELINE_COORD_ALPHA="0.5"):
        got = _coord_bonus(
            [_launch(30), _launch(30), _launch(30)], [12, 12, 18],
        )
    assert abs(got - 0.5 * 50.0) < 1e-9


def test_coord_invariant_to_leg_order():
    """Same legs in different input order produce the same bonus."""
    with _env(BASELINE_COORD_BONUS="1", BASELINE_COORD_ALPHA="0.5"):
        a = _coord_bonus(
            [_launch(40), _launch(20), _launch(30)], [10, 18, 10],
        )
        b = _coord_bonus(
            [_launch(30), _launch(40), _launch(20)], [10, 10, 18],
        )
    assert abs(a - b) < 1e-9


def test_coord_unequal_ships_same_step():
    """50+10 ships same step → cohort=60, conc=60, bonus=α·60."""
    with _env(BASELINE_COORD_BONUS="1", BASELINE_COORD_ALPHA="0.5"):
        got = _coord_bonus([_launch(50), _launch(10)], [12, 12])
    assert abs(got - 0.5 * 60.0) < 1e-9


def test_coord_zero_when_all_zero_ships():
    """Degenerate input (zero ships) → no bonus, no div-by-zero."""
    with _env(BASELINE_COORD_BONUS="1", BASELINE_COORD_ALPHA="0.5"):
        got = _coord_bonus([_launch(0), _launch(0)], [10, 10])
    assert got == 0.0


def test_coord_respects_alpha_env_override():
    """α=2.0 vs α=0.5 → 4× the bonus at the same cohort."""
    with _env(BASELINE_COORD_BONUS="1", BASELINE_COORD_ALPHA="2.0"):
        big = _coord_bonus([_launch(30), _launch(30)], [12, 12])
    with _env(BASELINE_COORD_BONUS="1", BASELINE_COORD_ALPHA="0.5"):
        small = _coord_bonus([_launch(30), _launch(30)], [12, 12])
    assert abs(big / small - 4.0) < 1e-6
