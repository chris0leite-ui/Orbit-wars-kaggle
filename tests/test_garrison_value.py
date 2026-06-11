"""Tests for the garrison-deficit reinforcement value (PRODUCER_PLUS_GARRISON_VALUE).

An own-target send earns lambda_g * prod_t when the target planet's local
balance vs the enemy's uncommitted reserve is negative at/after arrival and
the send covers the deficit. Default 0 = byte-identical.
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
        "producer_plus_main_test_garval",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_garval"] = module
    spec.loader.exec_module(module)
    return module


def _obs(P, owners, ships):
    owned = torch.tensor([o == "me" for o in owners])
    is_enemy = torch.tensor([o == "enemy" for o in owners])
    owner_abs = torch.where(owned, 0.0, torch.where(is_enemy, 1.0, -1.0))
    return SimpleNamespace(
        P=P, device=torch.device("cpu"), player_id=0,
        ships=torch.tensor(ships, dtype=torch.float32),
        alive=torch.ones(P, dtype=torch.bool),
        owner_abs=owner_abs,
        owned=owned, is_enemy=is_enemy, is_neutral=~(owned | is_enemy),
    )


def _cache(dist):
    return SimpleNamespace(cross_dist=[torch.tensor(dist, dtype=torch.float32)])


def _args(tgt_slot, send, eta, K=8, prod=(1.0, 1.0, 1.0), valid=True, is_def=True):
    dtype = torch.float32
    return dict(
        target_idx=torch.tensor([tgt_slot]),
        cand_tgt_slot=torch.tensor([tgt_slot]),
        cand_tgt_short=torch.tensor([0]),
        cand_send=torch.tensor([[send]], dtype=dtype),
        cand_eta=torch.tensor([[eta]], dtype=dtype),
        cand_valid=torch.tensor([valid]),
        cand_is_def=torch.tensor([is_def]),
        prod=torch.tensor(prod, dtype=dtype),
        K=K,
    )


def _stub_margins(pp_main, monkeypatch, threat, help_val):
    def threat_stub(obs, cache, target_idx, K, *, weight, lag=None):
        if threat is None: return None
        return torch.full((int(target_idx.shape[0]), K), float(threat) * float(weight))
    def help_stub(obs, cache, source_idx, K, *, lag=0.0):
        if help_val is None: return None
        return torch.full((int(source_idx.shape[0]), K), float(help_val))
    monkeypatch.setattr(pp_main, "_reactive_reinforcement_margin", threat_stub)
    monkeypatch.setattr(pp_main, "_friendly_support_margin", help_stub)


# planet 0 = ours g=20 prod=1 (the reinforce target); planet 2 = enemy
OBS = lambda: _obs(3, ["me", "me", "enemy"], [20.0, 60.0, 100.0])
DIST = [[0.0, 5.0, 30.0], [5.0, 0.0, 30.0], [30.0, 30.0, 0.0]]


def test_gate_default_off(monkeypatch, pp_main):
    monkeypatch.delenv("PRODUCER_PLUS_GARRISON_VALUE", raising=False)
    b = pp_main._garrison_value_bonus(obs=OBS(), cache=_cache(DIST),
                                      **_args(0, 80.0, 2.0))
    assert float(b.abs().sum()) == 0.0


def test_deficit_covered_earns_credit(monkeypatch, pp_main):
    # threat 100 (w=1), base = 20 + k + help 0 -> deficit ~ 72-79 > 0.
    # send 80 covers it: bonus = 12 * prod = 12.
    monkeypatch.setenv("PRODUCER_PLUS_GARRISON_VALUE", "12")
    monkeypatch.delenv("PRODUCER_PLUS_SOURCE_SAFETY", raising=False)
    _stub_margins(pp_main, monkeypatch, threat=100.0, help_val=0.0)
    b = pp_main._garrison_value_bonus(obs=OBS(), cache=_cache(DIST),
                                      **_args(0, 80.0, 2.0))
    assert float(b[0]) == pytest.approx(12.0)


def test_send_too_small_no_credit(monkeypatch, pp_main):
    # send 30 < deficit ~ 78: no credit (would not save the planet).
    monkeypatch.setenv("PRODUCER_PLUS_GARRISON_VALUE", "12")
    _stub_margins(pp_main, monkeypatch, threat=100.0, help_val=0.0)
    b = pp_main._garrison_value_bonus(obs=OBS(), cache=_cache(DIST),
                                      **_args(0, 30.0, 2.0))
    assert float(b[0]) == 0.0


def test_no_deficit_no_credit(monkeypatch, pp_main):
    # threat 10 < base: balance positive, reinforcement earns nothing.
    monkeypatch.setenv("PRODUCER_PLUS_GARRISON_VALUE", "12")
    _stub_margins(pp_main, monkeypatch, threat=10.0, help_val=0.0)
    b = pp_main._garrison_value_bonus(obs=OBS(), cache=_cache(DIST),
                                      **_args(0, 80.0, 2.0))
    assert float(b[0]) == 0.0


def test_help_cancels_deficit(monkeypatch, pp_main):
    # routable help 95 covers the threat: no deficit, no credit.
    monkeypatch.setenv("PRODUCER_PLUS_GARRISON_VALUE", "12")
    _stub_margins(pp_main, monkeypatch, threat=100.0, help_val=95.0)
    b = pp_main._garrison_value_bonus(obs=OBS(), cache=_cache(DIST),
                                      **_args(0, 80.0, 2.0))
    assert float(b[0]) == 0.0


def test_attack_candidate_excluded(monkeypatch, pp_main):
    monkeypatch.setenv("PRODUCER_PLUS_GARRISON_VALUE", "12")
    _stub_margins(pp_main, monkeypatch, threat=100.0, help_val=0.0)
    b = pp_main._garrison_value_bonus(obs=OBS(), cache=_cache(DIST),
                                      **_args(0, 80.0, 2.0, is_def=False))
    assert float(b[0]) == 0.0


def test_source_safety_weight_scales_threat(monkeypatch, pp_main):
    # w=0.5 halves threat to 50; base 20+k: deficit ~ 28; send 30 covers.
    monkeypatch.setenv("PRODUCER_PLUS_GARRISON_VALUE", "12")
    monkeypatch.setenv("PRODUCER_PLUS_SOURCE_SAFETY", "0.5")
    _stub_margins(pp_main, monkeypatch, threat=100.0, help_val=0.0)
    b = pp_main._garrison_value_bonus(obs=OBS(), cache=_cache(DIST),
                                      **_args(0, 30.0, 2.0))
    assert float(b[0]) == pytest.approx(12.0)
