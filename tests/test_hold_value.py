"""Tests for holding-time-priced capture credit (PRODUCER_PLUS_HOLD_VALUE).

Post-horizon production is credited per candidate ONLY when the projected
captured garrison (survivors + production) beats the enemy's full routable
mass at every later tick in the window. Default 0 = byte-identical.
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
        "producer_plus_main_test_holdval",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_holdval"] = module
    spec.loader.exec_module(module)
    return module


def test_gate_default_off(monkeypatch, pp_main):
    monkeypatch.delenv("PRODUCER_PLUS_HOLD_VALUE", raising=False)
    assert pp_main._hold_value() == 0.0


def _obs(P=3, neutral=(1,), enemy=(2,), ships=(30.0, 10.0, 20.0)):
    is_neutral = torch.zeros(P, dtype=torch.bool)
    is_enemy = torch.zeros(P, dtype=torch.bool)
    for i in neutral:
        is_neutral[i] = True
    for i in enemy:
        is_enemy[i] = True
    return SimpleNamespace(
        P=P,
        device=torch.device("cpu"),
        ships=torch.tensor(ships, dtype=torch.float32),
        alive=torch.ones(P, dtype=torch.bool),
        is_neutral=is_neutral,
        is_enemy=is_enemy,
    )


def _args(pp_main, *, tgt_slot, send, eta, K=8, prod=(1.0, 5.0, 2.0),
          floor_val=11.0, valid=True, is_def=False, T=1):
    dtype = torch.float32
    return dict(
        target_idx=torch.tensor([tgt_slot]),
        cand_tgt_slot=torch.tensor([tgt_slot]),
        cand_tgt_short=torch.tensor([0]),
        cand_send=torch.tensor([[send]], dtype=dtype),
        cand_eta=torch.tensor([[eta]], dtype=dtype),
        cand_valid=torch.tensor([valid]),
        cand_is_def=torch.tensor([is_def]),
        capture_floor_TK=torch.full((T, K), floor_val, dtype=dtype),
        prod=torch.tensor(prod, dtype=dtype),
        K=K,
    )


def _margin_stub(value):
    def stub(obs, cache, target_idx, K, *, weight, lag=None):
        if value is None:
            return None
        return torch.full((int(target_idx.shape[0]), K), float(value))
    return stub


def test_safe_capture_gets_full_credit(monkeypatch, pp_main):
    # No enemy can route anything (margin None): credit = lam * prod.
    monkeypatch.setenv("PRODUCER_PLUS_HOLD_VALUE", "12")
    monkeypatch.setattr(pp_main, "_reactive_reinforcement_margin", _margin_stub(None))
    b = pp_main._hold_value_bonus(obs=_obs(), cache=None,
                                  **_args(pp_main, tgt_slot=1, send=15.0, eta=4.0))
    assert float(b[0]) == pytest.approx(12.0 * 5.0)


def test_contested_capture_gets_zero(monkeypatch, pp_main):
    # Enemy routable mass 100 dwarfs survivors (15-11+1=5): no credit.
    monkeypatch.setenv("PRODUCER_PLUS_HOLD_VALUE", "12")
    monkeypatch.setattr(pp_main, "_reactive_reinforcement_margin", _margin_stub(100.0))
    b = pp_main._hold_value_bonus(obs=_obs(), cache=None,
                                  **_args(pp_main, tgt_slot=1, send=15.0, eta=4.0))
    assert float(b[0]) == 0.0


def test_strong_garrison_outgrows_threat(monkeypatch, pp_main):
    # Threat 20; survivors 40-11+1=30 ≥ 20 at every later tick: credited.
    monkeypatch.setenv("PRODUCER_PLUS_HOLD_VALUE", "12")
    monkeypatch.setattr(pp_main, "_reactive_reinforcement_margin", _margin_stub(20.0))
    b = pp_main._hold_value_bonus(obs=_obs(), cache=None,
                                  **_args(pp_main, tgt_slot=1, send=40.0, eta=4.0))
    assert float(b[0]) == pytest.approx(60.0)


def test_enemy_target_gets_no_credit(monkeypatch, pp_main):
    monkeypatch.setenv("PRODUCER_PLUS_HOLD_VALUE", "12")
    monkeypatch.setattr(pp_main, "_reactive_reinforcement_margin", _margin_stub(None))
    b = pp_main._hold_value_bonus(obs=_obs(), cache=None,
                                  **_args(pp_main, tgt_slot=2, send=30.0, eta=4.0))
    assert float(b[0]) == 0.0


def test_defensive_candidate_gets_no_credit(monkeypatch, pp_main):
    monkeypatch.setenv("PRODUCER_PLUS_HOLD_VALUE", "12")
    monkeypatch.setattr(pp_main, "_reactive_reinforcement_margin", _margin_stub(None))
    b = pp_main._hold_value_bonus(obs=_obs(), cache=None,
                                  **_args(pp_main, tgt_slot=1, send=15.0, eta=4.0,
                                          is_def=True))
    assert float(b[0]) == 0.0


def test_invalid_candidate_gets_no_credit(monkeypatch, pp_main):
    monkeypatch.setenv("PRODUCER_PLUS_HOLD_VALUE", "12")
    monkeypatch.setattr(pp_main, "_reactive_reinforcement_margin", _margin_stub(None))
    b = pp_main._hold_value_bonus(obs=_obs(), cache=None,
                                  **_args(pp_main, tgt_slot=1, send=15.0, eta=4.0,
                                          valid=False))
    assert float(b[0]) == 0.0


def test_zero_lambda_zero_everywhere(monkeypatch, pp_main):
    monkeypatch.delenv("PRODUCER_PLUS_HOLD_VALUE", raising=False)
    b = pp_main._hold_value_bonus(obs=_obs(), cache=None,
                                  **_args(pp_main, tgt_slot=1, send=15.0, eta=4.0))
    assert float(b.abs().sum()) == 0.0
