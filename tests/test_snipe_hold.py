"""Tests for snipe-hold (PRODUCER_PLUS_SNIPE_HOLD): reserve idle ships with
a dated toll-snipe appointment so the regroup lane doesn't drain them."""
from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(REPO_ROOT, "agents", "producer"),
          os.path.join(REPO_ROOT, "agents", "producer_plus")):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="module")
def pp_main():
    spec = importlib.util.spec_from_file_location(
        "producer_plus_main_test_snipe",
        os.path.join(REPO_ROOT, "agents", "producer_plus", "main.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_snipe"] = m
    spec.loader.exec_module(m)
    return m


def _board(pp_main, flip_tick=3, survivor=4.0, my_ships=30.0):
    """P=3: planet 0 mine, planet 1 neutral flipping to opp at flip_tick,
    planet 2 enemy. H=6."""
    from orbit_lite.movement import PlanetGarrisonStatus
    H = 6
    dtype = torch.float32
    own_row = [0] * (H + 1)
    flip_row = [-1] * flip_tick + [1] * (H + 1 - flip_tick)
    enemy_row = [1] * (H + 1)
    owner = torch.tensor([own_row, flip_row, enemy_row], dtype=torch.long)
    ships = torch.tensor(
        [[my_ships] * (H + 1),
         [9.0] * flip_tick + [survivor] * (H + 1 - flip_tick),
         [20.0] * (H + 1)], dtype=dtype)
    status = PlanetGarrisonStatus(owner=owner, ships=ships)
    obs = SimpleNamespace(
        player_id=0, P=3, device=torch.device("cpu"),
        ships=torch.tensor([my_ships, 9.0, 20.0], dtype=dtype),
        owned=torch.tensor([True, False, False]),
        alive=torch.tensor([True, True, True]),
    )
    return status, obs, H


def _empty_waves(pp_main):
    from orbit_lite.movement_step import LaunchEntries
    z = torch.zeros(0)
    return LaunchEntries(
        source_slots=torch.zeros(0, dtype=torch.long),
        target_slots=torch.zeros(0, dtype=torch.long),
        ships=z.clone(), angle=z.clone(), eta=z.clone(),
        valid=torch.zeros(0, dtype=torch.bool))


def _aim(eta, viable=True):
    def f(movement, src, tgt, sizes, **kw):
        n = int(src.shape[0])
        return {"angle": torch.zeros(n), "eta": torch.full((n,), float(eta)),
                "viable": torch.full((n,), bool(viable), dtype=torch.bool)}
    return f


def test_gate_default_off(monkeypatch, pp_main):
    monkeypatch.delenv("PRODUCER_PLUS_SNIPE_HOLD", raising=False)
    assert pp_main._snipe_hold_enabled() is False


def test_reserves_early_source(monkeypatch, pp_main):
    # We could arrive at eta 2 < flip+1 = 4: waiting unlocks the toll snipe.
    status, obs, H = _board(pp_main)
    monkeypatch.setattr(pp_main, "intercept_angle", _aim(eta=2.0))
    r = pp_main._snipe_hold_reserved_sources(
        obs=obs, garrison_status=status, background=None,
        wave_entries=_empty_waves(pp_main), H=H, movement=object())
    assert r is not None and r.tolist() == [True, False, False]


def test_no_reserve_when_already_late(monkeypatch, pp_main):
    # eta 5 >= flip+1 = 4: launching now already lands post-flip — the
    # normal candidate handles it; no reservation needed.
    status, obs, H = _board(pp_main)
    monkeypatch.setattr(pp_main, "intercept_angle", _aim(eta=5.0))
    r = pp_main._snipe_hold_reserved_sources(
        obs=obs, garrison_status=status, background=None,
        wave_entries=_empty_waves(pp_main), H=H, movement=object())
    assert r is None


def test_no_reserve_when_poor(monkeypatch, pp_main):
    # Source cannot afford survivor+2 after commitments.
    status, obs, H = _board(pp_main, my_ships=3.0)
    monkeypatch.setattr(pp_main, "intercept_angle", _aim(eta=2.0))
    r = pp_main._snipe_hold_reserved_sources(
        obs=obs, garrison_status=status, background=None,
        wave_entries=_empty_waves(pp_main), H=H, movement=object())
    assert r is None


def test_no_flip_no_reserve(monkeypatch, pp_main):
    status, obs, H = _board(pp_main)
    # Overwrite planet 1 trajectory: stays neutral.
    status.owner[1, :] = -1
    monkeypatch.setattr(pp_main, "intercept_angle", _aim(eta=2.0))
    r = pp_main._snipe_hold_reserved_sources(
        obs=obs, garrison_status=status, background=None,
        wave_entries=_empty_waves(pp_main), H=H, movement=object())
    assert r is None
