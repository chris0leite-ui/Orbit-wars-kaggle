"""Unit tests for the wave-attack coordination-cost penalty.

The penalty lives at agents/baseline/chooser_trajectory._coord_penalty and
is applied inside score_candidate_v4_joint at leaf-scoring time. These
tests exercise the pure-math function directly so they avoid the cost of
spinning up a fast_sim rollout.
"""
from __future__ import annotations

import os
import contextlib

from agents.baseline.chooser_trajectory import _coord_penalty


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


def test_coord_zero_when_flag_off():
    with _env(BASELINE_COORD_PENALTY=None):
        assert _coord_penalty([10, 18]) == 0.0


def test_coord_zero_when_alpha_zero():
    with _env(BASELINE_COORD_PENALTY="1", BASELINE_COORD_ALPHA="0"):
        assert _coord_penalty([10, 18]) == 0.0


def test_coord_zero_when_fewer_than_two_legs():
    """Solo emissions have no spread to penalise."""
    with _env(BASELINE_COORD_PENALTY="1", BASELINE_COORD_ALPHA="0.2"):
        assert _coord_penalty([]) == 0.0
        assert _coord_penalty([15]) == 0.0


def test_coord_zero_on_simultaneous_arrival():
    """Combat rule 1: same-step arrivals stack additively — no penalty."""
    with _env(BASELINE_COORD_PENALTY="1", BASELINE_COORD_ALPHA="0.2"):
        assert _coord_penalty([12, 12, 12]) == 0.0


def test_coord_quadratic_in_spread():
    """spread=8 → α·64; spread=4 → α·16 → ratio = 4."""
    with _env(BASELINE_COORD_PENALTY="1", BASELINE_COORD_ALPHA="0.2"):
        p_wide = _coord_penalty([10, 18])
        p_narrow = _coord_penalty([10, 14])
    assert abs(p_wide - 0.2 * 64.0) < 1e-9
    assert abs(p_narrow - 0.2 * 16.0) < 1e-9
    assert abs(p_wide / p_narrow - 4.0) < 1e-9


def test_coord_uses_max_minus_min_across_more_than_two_legs():
    """Three-leg coalition with arrivals (10, 12, 18) → spread=8."""
    with _env(BASELINE_COORD_PENALTY="1", BASELINE_COORD_ALPHA="0.5"):
        got = _coord_penalty([10, 12, 18])
    assert abs(got - 0.5 * 64.0) < 1e-9


def test_coord_respects_alpha_env_override():
    """α=2.0 vs default α=0.2 → 10× the penalty for the same spread."""
    with _env(BASELINE_COORD_PENALTY="1", BASELINE_COORD_ALPHA="2.0"):
        big = _coord_penalty([5, 15])
    with _env(BASELINE_COORD_PENALTY="1", BASELINE_COORD_ALPHA="0.2"):
        small = _coord_penalty([5, 15])
    assert abs(big / small - 10.0) < 1e-6
