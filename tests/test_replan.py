"""Tests for the one-ply replan (PRODUCER_PLUS_REPLAN).

The mechanism: after pass-1 waves are chosen, predict the opponent's reply
(same mirror + roi normalization as the response veto) and run the WHOLE
planner a second time with that reply as background. Where the veto only
drops doomed waves, the replan redirects their ships and plans defenses
against the predicted counter.

The mirror and the second planner pass are monkeypatched here — these tests
cover the gating, the skip conditions, and the roi re-normalization contract.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCER_PLUS_DIR = os.path.join(REPO_ROOT, "agents", "producer_plus")
PRODUCER_DIR = os.path.join(REPO_ROOT, "agents", "producer")

for p in (PRODUCER_DIR, PRODUCER_PLUS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


@dataclass(frozen=True)
class _Cfg:
    roi_threshold: float = 1.5


@pytest.fixture(scope="module")
def pp_main():
    spec = importlib.util.spec_from_file_location(
        "producer_plus_main_test_replan",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_replan"] = module
    spec.loader.exec_module(module)
    return module


def test_env_default_off(monkeypatch, pp_main):
    monkeypatch.delenv("PRODUCER_PLUS_REPLAN", raising=False)
    assert pp_main._replan_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_env_on(monkeypatch, pp_main, value):
    monkeypatch.setenv("PRODUCER_PLUS_REPLAN", value)
    assert pp_main._replan_enabled() is True


def test_2p_gate(monkeypatch, pp_main):
    monkeypatch.setenv("PRODUCER_PLUS_REPLAN", "1")
    monkeypatch.setenv("PRODUCER_PLUS_REPLAN_2P_ONLY", "1")
    assert pp_main._replan_active(2) is True
    assert pp_main._replan_active(4) is False
    monkeypatch.delenv("PRODUCER_PLUS_REPLAN_2P_ONLY")
    assert pp_main._replan_active(4) is True


def _board(pp_main):
    """P=3, A=2, H=4. Planet 0 mine (20 ships), planets 1-2 enemy."""
    from orbit_lite.movement import PlanetGarrisonStatus

    H = 4
    dtype = torch.float32
    owner = torch.tensor(
        [[0] * (H + 1), [1] * (H + 1), [1] * (H + 1)], dtype=torch.long)
    ships = torch.tensor(
        [[20.0] * (H + 1), [5.0] * (H + 1), [30.0] * (H + 1)], dtype=dtype)
    status = PlanetGarrisonStatus(
        owner=owner, ships=ships,
        pre_combat_owner=owner.clone(), pre_combat_ships=ships.clone(),
        arrivals_by_owner=torch.zeros(3, H + 1, 2, dtype=dtype),
    )
    obs = SimpleNamespace(
        player_id=0, P=3, device=torch.device("cpu"),
        ships=torch.tensor([20.0, 5.0, 30.0], dtype=dtype),
        owned=torch.tensor([True, False, False]),
        is_enemy=torch.tensor([False, True, True]),
        alive=torch.tensor([True, True, True]),
    )
    prod = torch.tensor([1.0, 2.0, 1.0], dtype=dtype)
    alive_by_step = torch.ones(H + 1, 3, dtype=torch.bool)
    return status, obs, prod, alive_by_step, H


def _entries(pp_main, ships, tgt=1, valid=True):
    from orbit_lite.movement_step import LaunchEntries
    dtype = torch.float32
    return LaunchEntries(
        source_slots=torch.tensor([0]),
        target_slots=torch.tensor([tgt]),
        ships=torch.tensor([float(ships)], dtype=dtype),
        angle=torch.tensor([0.0], dtype=dtype),
        eta=torch.tensor([2.0], dtype=dtype),
        valid=torch.tensor([valid]),
    )


def _reply(pp_main, ships, eta=2.0, tgt=1):
    from orbit_lite.garrison_launch import LaunchSet
    dtype = torch.float32
    return LaunchSet(
        source_slots=torch.tensor([2]),
        target_slots=torch.tensor([tgt]),
        ships=torch.tensor([float(ships)], dtype=dtype),
        eta=torch.tensor([float(eta)], dtype=dtype),
        owner=torch.tensor([1]),
        valid=torch.tensor([ships > 0]),
    )


def _run_replan(pp_main, monkeypatch, entries, reply, plan_capture):
    status, obs, prod, alive_by_step, H = _board(pp_main)
    monkeypatch.setattr(pp_main, "_predict_reply", lambda mine, **kw: reply)

    def _fake_plan(**kw):
        plan_capture.append(kw)
        return _entries(pp_main, 7.0, tgt=2)

    monkeypatch.setattr(pp_main, "plan_lite_waves", _fake_plan)
    return pp_main._apply_replan(
        entries,
        movement=None, obs=obs, obs_tensors={}, cache=None,
        garrison_status=status, prod=prod, alive_by_step=alive_by_step,
        config=_Cfg(), player_count=2,
        K_eta_override=None, H=H, opp_weights=None,
    )


def test_skip_when_pass1_empty(monkeypatch, pp_main):
    # Pass 1 fired nothing: the reply to an empty plan is what pass 1
    # already planned against — return unchanged, never mirror or replan.
    captured = []
    entries = _entries(pp_main, 10.0, valid=False)
    out = _run_replan(pp_main, monkeypatch, entries, _reply(pp_main, 30.0), captured)
    assert out is entries
    assert captured == []


def test_skip_when_reply_empty(monkeypatch, pp_main):
    # Opponent's predicted reply is empty: nothing to adapt to.
    captured = []
    entries = _entries(pp_main, 10.0)
    out = _run_replan(pp_main, monkeypatch, entries, _reply(pp_main, 0.0), captured)
    assert out is entries
    assert captured == []


def test_replan_returns_second_pass(monkeypatch, pp_main):
    # With a live reply, the result is the SECOND planner pass's entries.
    captured = []
    entries = _entries(pp_main, 10.0)
    out = _run_replan(pp_main, monkeypatch, entries, _reply(pp_main, 30.0), captured)
    assert len(captured) == 1
    assert out.target_slots.tolist() == [2]          # the fake pass-2 plan
    assert out.ships.tolist() == [7.0]


def test_replan_renormalizes_roi_and_passes_reply(monkeypatch, pp_main):
    # The second pass must receive the reply as background and a roi
    # threshold shifted by the do-nothing-under-reply score.
    captured = []
    entries = _entries(pp_main, 10.0)
    reply = _reply(pp_main, 30.0)
    status, obs, prod, alive_by_step, H = _board(pp_main)
    _run_replan(pp_main, monkeypatch, entries, reply, captured)
    kw = captured[0]
    assert kw["background"] is reply
    dn = float(pp_main._score_do_nothing(
        status=status, prod=prod, alive_by_step=alive_by_step,
        player_count=2, background=reply, player_id=0, opp_weights=None,
    ))
    assert kw["config"].roi_threshold == pytest.approx(dn + 1.5)
    # Raw player/horizon plumbing intact.
    assert kw["player_count"] == 2
