"""Tests for the source-safety drain cap (PRODUCER_PLUS_SOURCE_SAFETY).

A source may shed only what keeps it locally defensible:
    drain <= g_s + min_k( prod_s*k + help(s,k) - w*threat(s,k) )
Default 0 = byte-identical (allowance is None, no cap applied).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCER_PLUS_DIR = os.path.join(REPO_ROOT, "agents", "producer_plus")
PRODUCER_DIR = os.path.join(REPO_ROOT, "agents", "producer")

for p in (PRODUCER_DIR, PRODUCER_PLUS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="module")
def pp_main():
    spec = importlib.util.spec_from_file_location(
        "producer_plus_main_test_srcsafety",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_srcsafety"] = module
    spec.loader.exec_module(module)
    return module


def _obs(P, owners, ships):
    """owners: list of 'me'/'enemy'/'neutral' per planet."""
    owned = torch.tensor([o == "me" for o in owners])
    is_enemy = torch.tensor([o == "enemy" for o in owners])
    return SimpleNamespace(
        P=P,
        device=torch.device("cpu"),
        player_id=0,
        ships=torch.tensor(ships, dtype=torch.float32),
        alive=torch.ones(P, dtype=torch.bool),
        owned=owned,
        is_enemy=is_enemy,
        is_neutral=~(owned | is_enemy),
    )


def _cache(dist):
    """dist: [P, P] symmetric distances."""
    return SimpleNamespace(cross_dist=[torch.tensor(dist, dtype=torch.float32)])


def test_gate_default_off(monkeypatch, pp_main):
    monkeypatch.delenv("PRODUCER_PLUS_SOURCE_SAFETY", raising=False)
    obs = _obs(2, ["me", "enemy"], [50.0, 50.0])
    out = pp_main._source_safety_allowance(
        obs, _cache([[0.0, 5.0], [5.0, 0.0]]),
        source_idx=torch.tensor([0]), prod=torch.ones(2), K=8,
    )
    assert out is None


def test_no_enemies_no_constraint(monkeypatch, pp_main):
    monkeypatch.setenv("PRODUCER_PLUS_SOURCE_SAFETY", "1.0")
    obs = _obs(2, ["me", "neutral"], [50.0, 10.0])
    out = pp_main._source_safety_allowance(
        obs, _cache([[0.0, 5.0], [5.0, 0.0]]),
        source_idx=torch.tensor([0]), prod=torch.ones(2), K=8,
    )
    assert out is None


def test_isolated_source_vs_big_reserve(monkeypatch, pp_main):
    # Lone planet (g=50, prod=1) vs enemy reserve 80 reachable at k>=3.
    # Worst tick k=3: allowed = 50 + (1*3 + 0 - 80) = -27 -> clamp 0.
    monkeypatch.setenv("PRODUCER_PLUS_SOURCE_SAFETY", "1.0")
    obs = _obs(2, ["me", "enemy"], [50.0, 80.0])
    # distance such that enemy eta ~ 2.x: reaches from k=3 on
    d = [[0.0, 25.0], [25.0, 0.0]]
    out = pp_main._source_safety_allowance(
        obs, _cache(d), source_idx=torch.tensor([0]),
        prod=torch.tensor([1.0, 1.0]), K=8,
    )
    assert out is not None and float(out[0]) == 0.0


def test_small_threat_allows_partial_drain(monkeypatch, pp_main):
    # Threat 20 first reachable at k_hit (from the engine's speed law);
    # allowed = 50 + (k_hit - 20): partial drain, neither 0 nor full.
    monkeypatch.setenv("PRODUCER_PLUS_SOURCE_SAFETY", "1.0")
    speed = float(pp_main.fleet_speed(torch.tensor([20.0]))[0])
    d_val = speed * 2.5                                  # eta 2.5 -> hits at k=3
    obs = _obs(2, ["me", "enemy"], [50.0, 20.0])
    d = [[0.0, d_val], [d_val, 0.0]]
    out = pp_main._source_safety_allowance(
        obs, _cache(d), source_idx=torch.tensor([0]),
        prod=torch.tensor([1.0, 1.0]), K=8,
    )
    assert out is not None
    k_hit = 3.0
    assert float(out[0]) == pytest.approx(50.0 + k_hit - 20.0)


def test_friendly_help_restores_drain(monkeypatch, pp_main):
    # Same threat, but a friendly planet with 100 ships sits adjacent to the
    # source (helps from k=1): slack ~ k + 99 - 80 > 0 -> full drain allowed.
    monkeypatch.setenv("PRODUCER_PLUS_SOURCE_SAFETY", "1.0")
    obs = _obs(3, ["me", "enemy", "me"], [50.0, 80.0, 100.0])
    d = [[0.0, 25.0, 1.0], [25.0, 0.0, 25.0], [1.0, 25.0, 0.0]]
    out = pp_main._source_safety_allowance(
        obs, _cache(d), source_idx=torch.tensor([0]),
        prod=torch.tensor([1.0, 1.0, 1.0]), K=8,
    )
    assert out is not None and float(out[0]) >= 50.0


def test_threat_weight_scales(monkeypatch, pp_main):
    # w=0.5 halves the 80 threat -> allowed = 50 + min_k(k - 40) = 13.
    monkeypatch.setenv("PRODUCER_PLUS_SOURCE_SAFETY", "0.5")
    obs = _obs(2, ["me", "enemy"], [50.0, 80.0])
    d = [[0.0, 25.0], [25.0, 0.0]]
    out = pp_main._source_safety_allowance(
        obs, _cache(d), source_idx=torch.tensor([0]),
        prod=torch.tensor([1.0, 1.0]), K=8,
    )
    assert out is not None
    assert 5.0 <= float(out[0]) <= 20.0


def test_helper_excludes_source_itself(monkeypatch, pp_main):
    # Single owned planet: help must be zero (source can't help itself),
    # not its own garrison double-counted.
    monkeypatch.setenv("PRODUCER_PLUS_SOURCE_SAFETY", "1.0")
    obs = _obs(2, ["me", "enemy"], [50.0, 80.0])
    h = pp_main._friendly_support_margin(
        obs, _cache([[0.0, 25.0], [25.0, 0.0]]),
        torch.tensor([0]), 8,
    )
    assert h is not None and float(h.abs().sum()) == 0.0
