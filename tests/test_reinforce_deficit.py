"""Tests for the reinforcement deficit floor (PRODUCER_PLUS_REINFORCE_DEFICIT).

The mechanism: for an owned target whose do-nothing projection flips it at
tick k_f, pre-flip arrival cells get floor = ceil(post-flip survivor +
overhead) — the exact minimum send that holds the planet — instead of 1.
Default OFF leaves capture_floor untouched (byte-identical).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass

import pytest
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCER_PLUS_DIR = os.path.join(REPO_ROOT, "agents", "producer_plus")
PRODUCER_DIR = os.path.join(REPO_ROOT, "agents", "producer")


@pytest.fixture(scope="module")
def pp_main():
    for p in (PRODUCER_DIR, PRODUCER_PLUS_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        "producer_plus_main_test_reinf",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_reinf"] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class _Status:
    owner: torch.Tensor   # [P, H+1] long
    ships: torch.Tensor   # [P, H+1] float


def _status(owner_rows, ships_rows):
    return _Status(
        owner=torch.tensor(owner_rows, dtype=torch.long),
        ships=torch.tensor(ships_rows, dtype=torch.float32),
    )


def test_default_off(monkeypatch, pp_main):
    monkeypatch.delenv("PRODUCER_PLUS_REINFORCE_DEFICIT", raising=False)
    assert pp_main._reinforce_deficit_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_env_on(monkeypatch, pp_main, value):
    monkeypatch.setenv("PRODUCER_PLUS_REINFORCE_DEFICIT", value)
    assert pp_main._reinforce_deficit_enabled() is True


def test_preflip_cells_get_deficit(pp_main):
    # One planet, mine at k=0..2, flips to player 1 at k=3 with survivor 7.
    # H=5 horizon, K=5 floor window. Pre-flip cells (k=1,2) get 7+1=8;
    # cells at/after the flip keep their incoming (retake) floor.
    st = _status(
        [[0, 0, 0, 1, 1, 1]],
        [[10, 11, 12, 7, 9, 11]],
    )
    base = torch.tensor([[1.0, 1.0, 9.0, 11.0, 13.0]])   # as capture_floor built it
    out = pp_main._apply_reinforce_deficit_floor(
        base, garrison_status=st, target_idx=torch.tensor([0]), player_id=0,
    )
    assert out.tolist() == [[8.0, 8.0, 9.0, 11.0, 13.0]]


def test_no_flip_untouched(pp_main):
    st = _status([[0, 0, 0, 0]], [[10, 11, 12, 13]])
    base = torch.tensor([[1.0, 1.0, 1.0]])
    out = pp_main._apply_reinforce_deficit_floor(
        base, garrison_status=st, target_idx=torch.tensor([0]), player_id=0,
    )
    assert torch.equal(out, base)


def test_flip_at_k1_no_preflip_cells(pp_main):
    st = _status([[0, 1, 1, 1]], [[10, 5, 6, 7]])
    base = torch.tensor([[6.0, 7.0, 8.0]])
    out = pp_main._apply_reinforce_deficit_floor(
        base, garrison_status=st, target_idx=torch.tensor([0]), player_id=0,
    )
    assert torch.equal(out, base)


def test_enemy_target_untouched(pp_main):
    # Not ours now — capture_floor's not-mine branch owns these cells.
    st = _status([[2, 2, 0, 0]], [[10, 5, 6, 7]])
    base = torch.tensor([[6.0, 7.0, 8.0]])
    out = pp_main._apply_reinforce_deficit_floor(
        base, garrison_status=st, target_idx=torch.tensor([0]), player_id=0,
    )
    assert torch.equal(out, base)


def test_deficit_never_lowers_existing_floor(pp_main):
    # Survivor 2 -> deficit 3; an existing higher floor must win the max.
    st = _status([[0, 0, 1, 1]], [[10, 11, 2, 3]])
    base = torch.tensor([[5.0, 3.0, 4.0]])
    out = pp_main._apply_reinforce_deficit_floor(
        base, garrison_status=st, target_idx=torch.tensor([0]), player_id=0,
    )
    assert out.tolist() == [[5.0, 3.0, 4.0]]
