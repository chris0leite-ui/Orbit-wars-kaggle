"""Tests for the trust-weighted opponent model (PRODUCER_PLUS_REPLY_TRUST).

Each turn, last turn's predicted opponent launches are verified against the
fleets that actually appeared (matched by source planet + owner, ships
within 2x); an exponential moving recall scales the reply's priced ships
(certainty-equivalent). Producer-likes keep trust ~1 (behavior unchanged);
off-model opponents degrade the veto gracefully toward unconditioned play.
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
        "producer_plus_main_test_trust",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_trust"] = module
    spec.loader.exec_module(module)
    return module


def test_gate_default_off(monkeypatch, pp_main):
    monkeypatch.delenv("PRODUCER_PLUS_REPLY_TRUST", raising=False)
    assert pp_main._reply_trust_enabled() is False


def _obs_tensors(fleet_rows, planet_rows=None):
    """fleet_rows: [fleet_id, owner, x, y, angle, from_id, ships]."""
    if planet_rows is None:
        planet_rows = [[0, 0, 40, 50, 1, 30, 1], [7, 1, 60, 50, 1, 10, 2]]
    return {
        "planets": torch.tensor(planet_rows, dtype=torch.float32),
        "fleets": torch.tensor(
            fleet_rows if fleet_rows else [[-1.0] * 7], dtype=torch.float32),
    }


def _bg(pp_main, src_slot, owner, ships):
    from orbit_lite.garrison_launch import LaunchSet
    dtype = torch.float32
    return LaunchSet(
        source_slots=torch.tensor([src_slot]),
        target_slots=torch.tensor([0]),
        ships=torch.tensor([float(ships)], dtype=dtype),
        eta=torch.tensor([2.0], dtype=dtype),
        owner=torch.tensor([owner]),
        valid=torch.tensor([True]),
    )


def test_no_predictions_keeps_full_trust(pp_main):
    mem = SimpleNamespace()
    t = pp_main._update_reply_trust(mem, _obs_tensors([]), pid=0)
    assert t == 1.0


def test_fulfilled_prediction_keeps_trust(pp_main):
    # Predict: planet 7 (slot 1), owner 1, 20 ships. Next turn a new enemy
    # fleet appears from planet 7 with 18 ships -> match, trust stays 1.
    mem = SimpleNamespace()
    pp_main._record_reply_prediction(mem, _bg(pp_main, 1, 1, 20.0), _obs_tensors([]))
    t = pp_main._update_reply_trust(
        mem, _obs_tensors([[100, 1, 0, 0, 0, 7, 18.0]]), pid=0)
    assert t == pytest.approx(1.0)


def test_missed_prediction_decays_trust(pp_main):
    mem = SimpleNamespace()
    pp_main._record_reply_prediction(mem, _bg(pp_main, 1, 1, 20.0), _obs_tensors([]))
    t = pp_main._update_reply_trust(mem, _obs_tensors([]), pid=0)
    assert t == pytest.approx(0.8)        # 0.8*1.0 + 0.2*0.0


def test_old_fleets_dont_count_as_fulfillment(pp_main):
    # The fleet existed at prediction time (id known) -> not a new launch.
    mem = SimpleNamespace()
    existing = [[100, 1, 0, 0, 0, 7, 20.0]]
    pp_main._record_reply_prediction(
        mem, _bg(pp_main, 1, 1, 20.0), _obs_tensors(existing))
    t = pp_main._update_reply_trust(mem, _obs_tensors(existing), pid=0)
    assert t == pytest.approx(0.8)


def test_ships_mismatch_doesnt_match(pp_main):
    # Right source, wildly wrong size (3 vs predicted 20): no match.
    mem = SimpleNamespace()
    pp_main._record_reply_prediction(mem, _bg(pp_main, 1, 1, 20.0), _obs_tensors([]))
    t = pp_main._update_reply_trust(
        mem, _obs_tensors([[100, 1, 0, 0, 0, 7, 3.0]]), pid=0)
    assert t == pytest.approx(0.8)


def test_trust_floor(pp_main):
    mem = SimpleNamespace(trust_ema=0.01)
    t = pp_main._update_reply_trust(mem, _obs_tensors([]), pid=0)
    assert t == pp_main._REPLY_TRUST_FLOOR


def test_scaling_is_identity_at_full_trust(pp_main):
    ls = _bg(pp_main, 1, 1, 20.0)
    assert pp_main._scale_launch_set_ships(ls, 1.0) is ls
    scaled = pp_main._scale_launch_set_ships(ls, 0.5)
    assert scaled.ships.tolist() == [10.0]
