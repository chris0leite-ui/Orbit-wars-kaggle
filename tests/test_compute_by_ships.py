"""Tests for the compute_by_ships lever (PI observation 2026-06-03).

Covers the two helpers (Lever 1: per-source enumeration breadth, Lever 2:
per-source K bonus) at the unit level. End-to-end behavior (rear planet emits
a launch that previously sat idle) is verified separately via a fast.py play
replay against a fixed seed — these unit tests are necessary but NOT
sufficient per Rule 38 (fix-verification reproduces the failure state).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Lever 2: per-source K bonus in capture_horizon_k via _apply_src_ratio_bonus
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    for var in (
        "BASELINE_COMPUTE_BY_SHIPS",
        "BASELINE_ADAPTIVE_K",
        "BASELINE_STATE_DRIVEN_K",
        "BASELINE_CAPTURE_HORIZON_K",
        "BASELINE_LAUNCH_RULES",
    ):
        monkeypatch.delenv(var, raising=False)


def test_apply_src_ratio_bonus_off_returns_base():
    from agents.baseline.launch_rules import _apply_src_ratio_bonus
    # env unset → lever OFF → no bonus regardless of ratio
    assert _apply_src_ratio_bonus(10, 4.0) == 10
    assert _apply_src_ratio_bonus(20, 8.0) == 20


def test_apply_src_ratio_bonus_none_ratio(monkeypatch):
    from agents.baseline.launch_rules import _apply_src_ratio_bonus
    monkeypatch.setenv("BASELINE_COMPUTE_BY_SHIPS", "1")
    # None or sub-average source → no bonus
    assert _apply_src_ratio_bonus(10, None) == 10
    assert _apply_src_ratio_bonus(10, 0.5) == 10
    assert _apply_src_ratio_bonus(10, 1.0) == 10


def test_apply_src_ratio_bonus_surplus_log_scale(monkeypatch):
    from agents.baseline.launch_rules import _apply_src_ratio_bonus
    monkeypatch.setenv("BASELINE_COMPUTE_BY_SHIPS", "1")
    # base=10, ratio=2.0 → bonus = round(10 * 0.3 * 1.0) = 3 → 13
    assert _apply_src_ratio_bonus(10, 2.0) == 13
    # base=10, ratio=4.0 → bonus = round(10 * 0.3 * 2.0) = 6 BUT cap = 5 → 15
    assert _apply_src_ratio_bonus(10, 4.0) == 15
    # base=10, ratio=8.0 → bonus capped at 5 → 15
    assert _apply_src_ratio_bonus(10, 8.0) == 15
    # base=20, ratio=2.0 → bonus = round(20 * 0.3 * 1.0) = 6 → 26
    assert _apply_src_ratio_bonus(20, 2.0) == 26


def test_apply_src_ratio_bonus_only_raises_never_lowers(monkeypatch):
    from agents.baseline.launch_rules import _apply_src_ratio_bonus
    monkeypatch.setenv("BASELINE_COMPUTE_BY_SHIPS", "1")
    # Cannot go below base_k for any input.
    for base in (5, 10, 15, 20):
        for ratio in (None, -1.0, 0.0, 0.5, 1.0, 2.0, 100.0):
            assert _apply_src_ratio_bonus(base, ratio) >= base


def test_capture_horizon_k_static_default_floor():
    """Champion default: env-clean → returns the static floor (10)."""
    from agents.baseline.launch_rules import capture_horizon_k
    assert capture_horizon_k() == 10
    # src_ratio passed but lever off → still floor
    assert capture_horizon_k(src_ratio=4.0) == 10


def test_capture_horizon_k_adaptive_alone(monkeypatch):
    """Adaptive K ON, compute_by_ships OFF: src_ratio is ignored."""
    from agents.baseline.launch_rules import capture_horizon_k
    monkeypatch.setenv("BASELINE_ADAPTIVE_K", "1")
    # step=0 → k_open=20
    assert capture_horizon_k(0) == 20
    # step=30 → floor=10
    assert capture_horizon_k(30) == 10
    # src_ratio without compute_by_ships → ignored
    assert capture_horizon_k(0, src_ratio=4.0) == 20


def test_capture_horizon_k_compound_adaptive_plus_ships(monkeypatch):
    """Adaptive + compute_by_ships compose: bonus applies after base K."""
    from agents.baseline.launch_rules import capture_horizon_k
    monkeypatch.setenv("BASELINE_ADAPTIVE_K", "1")
    monkeypatch.setenv("BASELINE_COMPUTE_BY_SHIPS", "1")
    # step=30, base=10, ratio=4.0 → 10 + min(5, round(3)) = 13 → cap=5 → 15
    assert capture_horizon_k(30, src_ratio=4.0) == 15
    # step=0, base=20, ratio=2.0 → 20 + min(10, round(6)) = 26
    assert capture_horizon_k(0, src_ratio=2.0) == 26
    # step=0, base=20, ratio=4.0 → 20 + min(10, round(12)) = 30
    assert capture_horizon_k(0, src_ratio=4.0) == 30


# ---------------------------------------------------------------------------
# Lever 1: _targets_for_src
# ---------------------------------------------------------------------------


def test_targets_for_src_off_returns_default():
    from agents.baseline.proposer import _targets_for_src
    src = SimpleNamespace(ships=100)
    # enabled=False → always 8 regardless of avg
    assert _targets_for_src(src, 10.0, enabled=False) == 8
    assert _targets_for_src(src, 100.0, enabled=False) == 8
    # enabled=True but avg=0 → falls back to 8 (no average to compare against)
    assert _targets_for_src(src, 0.0, enabled=True) == 8


def test_targets_for_src_average_planet():
    from agents.baseline.proposer import _targets_for_src
    src = SimpleNamespace(ships=10)
    # ratio = 1.0 → log2(1) = 0 → 8 * (1+0) = 8 (default)
    assert _targets_for_src(src, 10.0, enabled=True) == 8


def test_targets_for_src_surplus_scaling():
    from agents.baseline.proposer import _targets_for_src
    # ratio = 2.0 → log2(2) = 1 → 8 * (1 + 0.4) = 11.2 → 11
    src = SimpleNamespace(ships=20)
    assert _targets_for_src(src, 10.0, enabled=True) == 11
    # ratio = 4.0 → log2(4) = 2 → 8 * (1 + 0.8) = 14.4 → 14
    src = SimpleNamespace(ships=40)
    assert _targets_for_src(src, 10.0, enabled=True) == 14
    # ratio = 8.0 → log2(8) = 3 → 8 * (1 + 1.2) = 17.6 → clamp to 16
    src = SimpleNamespace(ships=80)
    assert _targets_for_src(src, 10.0, enabled=True) == 16


def test_targets_for_src_floor_for_small():
    from agents.baseline.proposer import _targets_for_src
    # ratio = 0.5 → returns FLOOR=4 (clamp branch)
    src = SimpleNamespace(ships=5)
    assert _targets_for_src(src, 10.0, enabled=True) == 4
    # ratio = 0.1 → still FLOOR
    src = SimpleNamespace(ships=1)
    assert _targets_for_src(src, 10.0, enabled=True) == 4


def test_targets_for_src_never_below_floor_never_above_ceil():
    from agents.baseline.proposer import _targets_for_src
    for ships in (1, 2, 5, 10, 25, 50, 100, 500, 5000):
        src = SimpleNamespace(ships=ships)
        n = _targets_for_src(src, 10.0, enabled=True)
        assert 4 <= n <= 16, f"ships={ships} got n={n}"
