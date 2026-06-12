"""Unit tests for the adaptive K_eta formula in ``producer_plus``.

Mirrors the env-var-driven test style of
``tests/test_compute_by_ships.py::test_capture_horizon_k_adaptive_alone``.
Covers Step 2 of ``state/MIGRATION_PLAN.md`` — the bit-identical default
and the shrink-only schedule when adaptive K is enabled.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCER_PLUS_DIR = os.path.join(REPO_ROOT, "agents", "producer_plus")
PRODUCER_DIR = os.path.join(REPO_ROOT, "agents", "producer")


@pytest.fixture(scope="module")
def producer_plus_main():
    for p in (PRODUCER_DIR, PRODUCER_PLUS_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        "producer_plus_main_test", os.path.join(PRODUCER_PLUS_DIR, "main.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test"] = module
    spec.loader.exec_module(module)
    return module


def test_default_off_returns_H(monkeypatch, producer_plus_main):
    monkeypatch.delenv("PRODUCER_PLUS_ADAPTIVE_K", raising=False)
    for step in (0, 5, 15, 30, 999):
        assert producer_plus_main.compute_k_eta_for_step(step, H=18) == 18


def test_explicit_off_returns_H(monkeypatch, producer_plus_main):
    monkeypatch.setenv("PRODUCER_PLUS_ADAPTIVE_K", "0")
    assert producer_plus_main.compute_k_eta_for_step(0, H=18) == 18
    assert producer_plus_main.compute_k_eta_for_step(30, H=18) == 18


def test_on_schedule_clamped_to_H(monkeypatch, producer_plus_main):
    monkeypatch.setenv("PRODUCER_PLUS_ADAPTIVE_K", "1")
    # step=0: raw=20, clamped to H=18.
    assert producer_plus_main.compute_k_eta_for_step(0, H=18) == 18
    # step=15: raw = 20 - 10*15/30 = 15, no clamp.
    assert producer_plus_main.compute_k_eta_for_step(15, H=18) == 15
    # step=20: raw = 20 - 10*20/30 ≈ 13.33 → round → 13.
    assert producer_plus_main.compute_k_eta_for_step(20, H=18) == 13
    # step=30: at the settle boundary → floor.
    assert producer_plus_main.compute_k_eta_for_step(30, H=18) == 10
    # step well past settle → still floor.
    assert producer_plus_main.compute_k_eta_for_step(999, H=18) == 10


@pytest.mark.parametrize(
    "env, step, H, expected",
    [
        ({"PRODUCER_PLUS_ADAPTIVE_K_OPEN": "15"}, 0, 18, 15),
        ({"PRODUCER_PLUS_ADAPTIVE_K_FLOOR": "6"}, 30, 18, 6),
        ({"PRODUCER_PLUS_ADAPTIVE_K_TSETTLE": "60"}, 30, 18, 15),
        ({"PRODUCER_PLUS_ADAPTIVE_K_TSETTLE": "0"}, 0, 18, 10),
    ],
)
def test_env_overrides(monkeypatch, producer_plus_main, env, step, H, expected):
    monkeypatch.setenv("PRODUCER_PLUS_ADAPTIVE_K", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert producer_plus_main.compute_k_eta_for_step(step, H=H) == expected


def test_H_below_floor_clamps_to_H(monkeypatch, producer_plus_main):
    # 4P preset uses H=13. floor=10 < 13, so step=0 → min(20, 13) = 13.
    monkeypatch.setenv("PRODUCER_PLUS_ADAPTIVE_K", "1")
    assert producer_plus_main.compute_k_eta_for_step(0, H=13) == 13
    assert producer_plus_main.compute_k_eta_for_step(30, H=13) == 10
    # Extreme: H=5 < floor=10. Result should not exceed H.
    assert producer_plus_main.compute_k_eta_for_step(0, H=5) == 5
    assert producer_plus_main.compute_k_eta_for_step(30, H=5) == 5


def test_K_OPEN_equals_floor_returns_floor(monkeypatch, producer_plus_main):
    monkeypatch.setenv("PRODUCER_PLUS_ADAPTIVE_K", "1")
    monkeypatch.setenv("PRODUCER_PLUS_ADAPTIVE_K_OPEN", "10")
    assert producer_plus_main.compute_k_eta_for_step(0, H=18) == 10
    assert producer_plus_main.compute_k_eta_for_step(30, H=18) == 10
