"""Tests for the redirect (PRODUCER_PLUS_REDIRECT).

The mechanism: when the response veto drops waves, spend the freed budget
with ONE extra planner pass in which surviving waves are committed (sources
debited, effects + predicted reply in the scorer background). Pass-1
commitments are never reopened — the anti-oscillation answer to the full
replan's phantom-parry conservatism (tournament: replan split 2-2 with
under-aggression on the losing seeds).
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

for p in (PRODUCER_DIR, PRODUCER_PLUS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


@dataclass(frozen=True)
class _Cfg:
    roi_threshold: float = 1.5


@pytest.fixture(scope="module")
def pp_main():
    spec = importlib.util.spec_from_file_location(
        "producer_plus_main_test_redirect",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_redirect"] = module
    spec.loader.exec_module(module)
    return module


def test_env_default_off(monkeypatch, pp_main):
    monkeypatch.delenv("PRODUCER_PLUS_REDIRECT", raising=False)
    assert pp_main._redirect_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_env_on(monkeypatch, pp_main, value):
    monkeypatch.setenv("PRODUCER_PLUS_REDIRECT", value)
    assert pp_main._redirect_enabled() is True


def test_2p_gate(monkeypatch, pp_main):
    monkeypatch.setenv("PRODUCER_PLUS_REDIRECT", "1")
    monkeypatch.setenv("PRODUCER_PLUS_REDIRECT_2P_ONLY", "1")
    assert pp_main._redirect_active(2) is True
    assert pp_main._redirect_active(4) is False
    monkeypatch.delenv("PRODUCER_PLUS_REDIRECT_2P_ONLY")
    assert pp_main._redirect_active(4) is True


def _board(pp_main):
    """P=3, A=2, H=4. Planet 0 mine (20 ships), planets 1-2 enemy."""
    from orbit_lite.movement import PlanetGarrisonStatus
    from orbit_lite.obs import ParsedObs

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
    P = 3
    z = torch.zeros(P, dtype=dtype)
    obs = ParsedObs(
        alive=torch.tensor([True, True, True]),
        x=z.clone(), y=z.clone(), r=z.clone(),
        ships=torch.tensor([20.0, 5.0, 30.0], dtype=dtype),
        prod=torch.tensor([1.0, 2.0, 1.0], dtype=dtype),
        owner_abs=torch.tensor([0.0, 1.0, 1.0], dtype=dtype),
        owned=torch.tensor([True, False, False]),
        is_enemy=torch.tensor([False, True, True]),
        is_neutral=torch.tensor([False, False, False]),
        orb_r=z.clone(), orb_a0=z.clone(),
        is_orbiting=torch.tensor([False, False, False]),
        angvel=torch.tensor(0.0),
        step=torch.tensor(0.0),
        f_alive=torch.zeros(0, dtype=torch.bool),
        f_owner=torch.zeros(0, dtype=dtype),
        f_x=torch.zeros(0, dtype=dtype),
        f_y=torch.zeros(0, dtype=dtype),
        f_angle=torch.zeros(0, dtype=dtype),
        f_ships=torch.zeros(0, dtype=dtype),
        player_id=0, P=P, F=0, device=torch.device("cpu"),
    )
    prod = obs.prod
    alive_by_step = torch.ones(H + 1, 3, dtype=torch.bool)
    return status, obs, prod, alive_by_step, H


def _entries(pp_main, rows):
    """rows: list of (src, tgt, ships, valid)."""
    from orbit_lite.movement_step import LaunchEntries
    dtype = torch.float32
    return LaunchEntries(
        source_slots=torch.tensor([r[0] for r in rows]),
        target_slots=torch.tensor([r[1] for r in rows]),
        ships=torch.tensor([float(r[2]) for r in rows], dtype=dtype),
        angle=torch.zeros(len(rows), dtype=dtype),
        eta=torch.full((len(rows),), 2.0, dtype=dtype),
        valid=torch.tensor([r[3] for r in rows]),
    )


def _reply(pp_main, ships=8.0, tgt=1):
    from orbit_lite.garrison_launch import LaunchSet
    dtype = torch.float32
    return LaunchSet(
        source_slots=torch.tensor([2]),
        target_slots=torch.tensor([tgt]),
        ships=torch.tensor([float(ships)], dtype=dtype),
        eta=torch.tensor([2.0], dtype=dtype),
        owner=torch.tensor([1]),
        valid=torch.tensor([True]),
    )


def _run_redirect(pp_main, monkeypatch, entries, extra_rows, capture):
    status, obs, prod, alive_by_step, H = _board(pp_main)

    def _fake_plan(**kw):
        capture.append(kw)
        return _entries(pp_main, extra_rows)

    monkeypatch.setattr(pp_main, "plan_lite_waves", _fake_plan)
    return pp_main._apply_redirect(
        entries,
        reply=_reply(pp_main),
        movement=None, obs=obs, obs_tensors={}, cache=None,
        garrison_status=status, prod=prod, alive_by_step=alive_by_step,
        config=_Cfg(), player_count=2,
        K_eta_override=None, H=H, opp_weights=None,
    )


def test_committed_sources_debited_and_background_merged(monkeypatch, pp_main):
    # Surviving wave 0->1 (12 ships); a vetoed row is invalid. The extra
    # pass must see planet 0 at 20-12=8 ships and a background of
    # reply (L=1) + committed (L=1).
    capture = []
    entries = _entries(pp_main, [(0, 1, 12.0, True), (0, 2, 5.0, False)])
    _run_redirect(pp_main, monkeypatch, entries, [(0, 2, 3.0, True)], capture)
    kw = capture[0]
    assert kw["obs"].ships.tolist() == [8.0, 5.0, 30.0]
    bg = kw["background"]
    assert bg.valid.tolist() == [True, True]
    assert bg.owner.tolist() == [1, 0]            # reply (opp) + committed (mine)
    assert bg.ships.tolist() == [8.0, 12.0]


def test_roi_renormalized_under_combined_background(monkeypatch, pp_main):
    capture = []
    entries = _entries(pp_main, [(0, 1, 12.0, True), (0, 2, 5.0, False)])
    status, obs, prod, alive_by_step, H = _board(pp_main)
    _run_redirect(pp_main, monkeypatch, entries, [(0, 2, 3.0, True)], capture)
    kw = capture[0]
    dn = float(pp_main._score_do_nothing(
        status=status, prod=prod, alive_by_step=alive_by_step,
        player_count=2, background=kw["background"], player_id=0,
        opp_weights=None,
    ))
    assert kw["config"].roi_threshold == pytest.approx(dn + 1.5)


def test_extra_waves_appended(monkeypatch, pp_main):
    capture = []
    entries = _entries(pp_main, [(0, 1, 12.0, True), (0, 2, 5.0, False)])
    out = _run_redirect(pp_main, monkeypatch, entries, [(0, 2, 3.0, True)], capture)
    assert out.valid.tolist() == [True, False, True]
    assert out.ships.tolist() == [12.0, 5.0, 3.0]
    assert out.target_slots.tolist() == [1, 2, 2]


def test_no_extra_waves_returns_entries_unchanged(monkeypatch, pp_main):
    capture = []
    entries = _entries(pp_main, [(0, 1, 12.0, True), (0, 2, 5.0, False)])
    out = _run_redirect(pp_main, monkeypatch, entries, [(0, 2, 3.0, False)], capture)
    assert out is entries


def test_all_vetoed_plans_from_full_budget(monkeypatch, pp_main):
    # Every wave vetoed: nothing committed, full garrisons, background =
    # reply alone.
    capture = []
    entries = _entries(pp_main, [(0, 1, 12.0, False), (0, 2, 5.0, False)])
    _run_redirect(pp_main, monkeypatch, entries, [(0, 2, 3.0, True)], capture)
    kw = capture[0]
    assert kw["obs"].ships.tolist() == [20.0, 5.0, 30.0]
    assert kw["background"].valid.tolist() == [True]
    assert kw["background"].owner.tolist() == [1]
