"""Tests for background-aware floors (PRODUCER_PLUS_BG_FLOORS).

The mechanism: re-project garrison trajectories with the opponent's
predicted launches applied (exact engine recurrence) and let the SIZING
subsystem — capture_floor, defensive shortlist, safe_drain — read that,
while the scorer keeps the static baseline (it merges the background into
every candidate's flow diff itself).
"""
from __future__ import annotations

import importlib.util
import os
import sys

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
        "producer_plus_main_test_bgfloors",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_bgfloors"] = module
    spec.loader.exec_module(module)
    return module


def test_env_default_off(monkeypatch, pp_main):
    monkeypatch.delenv("PRODUCER_PLUS_BG_FLOORS", raising=False)
    assert pp_main._bg_floors_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_env_on(monkeypatch, pp_main, value):
    monkeypatch.setenv("PRODUCER_PLUS_BG_FLOORS", value)
    assert pp_main._bg_floors_enabled() is True


def _status(pp_main, owners, ships, prods, H=6):
    """Static do-nothing status for P planets with constant garrisons +
    per-step production for owned planets (matching the engine recurrence)."""
    from orbit_lite.movement import PlanetGarrisonStatus

    P = len(owners)
    dtype = torch.float32
    owner = torch.tensor([[o] * (H + 1) for o in owners], dtype=torch.long)
    traj = []
    for s, o, pr in zip(ships, owners, prods):
        row = [float(s) + (float(pr) * k if o >= 0 else 0.0) for k in range(H + 1)]
        traj.append(row)
    ships_t = torch.tensor(traj, dtype=dtype)
    status = PlanetGarrisonStatus(
        owner=owner, ships=ships_t,
        pre_combat_owner=owner.clone(), pre_combat_ships=ships_t.clone(),
        arrivals_by_owner=torch.zeros(P, H + 1, 2, dtype=dtype),
    )
    prod = torch.tensor([float(p) for p in prods], dtype=dtype)
    alive = torch.ones(H + 1, P, dtype=torch.bool)
    return status, prod, alive, H


def _bg(pp_main, src, tgt, ships, eta, owner=1):
    from orbit_lite.garrison_launch import LaunchSet
    dtype = torch.float32
    return LaunchSet(
        source_slots=torch.tensor([src]),
        target_slots=torch.tensor([tgt]),
        ships=torch.tensor([float(ships)], dtype=dtype),
        eta=torch.tensor([float(eta)], dtype=dtype),
        owner=torch.tensor([owner]),
        valid=torch.tensor([True]),
    )


def test_reinforcement_raises_target_trajectory(pp_main):
    # P0 mine(20), P1 enemy(5), P2 enemy(30); no production. Predicted
    # launch: P2 -> P1, 10 ships, eta 2. Adjusted: P1 holds 15 from k=2,
    # P2 debited to 20 from k=0.
    status, prod, alive, H = _status(pp_main, [0, 1, 1], [20, 5, 30], [0, 0, 0])
    adj = pp_main._background_adjusted_status(
        status, background=_bg(pp_main, 2, 1, 10, 2.0), prod=prod, alive_by_step=alive,
    )
    assert adj.ships[1, 1].item() == 5.0          # before arrival
    assert adj.ships[1, 2].item() == 15.0         # after arrival
    assert adj.ships[1, H].item() == 15.0
    assert adj.ships[2, 0].item() == 20.0         # source debited now
    assert int(adj.owner[1, 2].item()) == 1       # still enemy (reinforced)


def test_predicted_capture_flips_neutral_to_cheap_target(pp_main):
    # P1 NEUTRAL garrison 5; enemy launches 12 at it, eta 2. Survivor
    # 12-5=7, owner 1 from k=2 — a toll-snipe window the static floor
    # can't see (static: neutral 5 forever).
    status, prod, alive, H = _status(pp_main, [0, -1, 1], [20, 5, 30], [0, 0, 0])
    adj = pp_main._background_adjusted_status(
        status, background=_bg(pp_main, 2, 1, 12, 2.0), prod=prod, alive_by_step=alive,
    )
    assert int(adj.owner[1, 1].item()) == -1
    assert adj.ships[1, 1].item() == 5.0
    assert int(adj.owner[1, 2].item()) == 1
    assert adj.ships[1, 2].item() == 7.0


def test_capture_floor_sees_the_parry(pp_main):
    # capture_floor on the adjusted status prices the predicted
    # reinforcement: floor at k=2 is 5+10+1 = 16 instead of 6.
    from orbit_lite.planner_core import capture_floor

    status, prod, alive, H = _status(pp_main, [0, 1, 1], [20, 5, 30], [0, 0, 0])
    adj = pp_main._background_adjusted_status(
        status, background=_bg(pp_main, 2, 1, 10, 2.0), prod=prod, alive_by_step=alive,
    )
    tgt = torch.tensor([1])
    floor_static = capture_floor(
        status, target_idx=tgt, k_max=4, capture_overhead=1.0, player_id=0)
    floor_adj = capture_floor(
        adj, target_idx=tgt, k_max=4, capture_overhead=1.0, player_id=0)
    assert floor_static[0, 1].item() == 6.0       # k=2 static
    assert floor_adj[0, 1].item() == 16.0         # k=2 with the parry priced in
    assert floor_adj[0, 0].item() == 6.0          # k=1 (before arrival) unchanged


def test_safe_drain_holds_back_attacked_source(pp_main):
    # MY planet 0 (20 ships) is the target of a predicted 15-ship strike
    # at eta 2: adjusted trajectory dips to 5 — safe_drain drops 20 -> 5.
    from orbit_lite.planner_core import safe_drain

    status, prod, alive, H = _status(pp_main, [0, 1, 1], [20, 5, 30], [0, 0, 0])
    adj = pp_main._background_adjusted_status(
        status, background=_bg(pp_main, 2, 0, 15, 2.0), prod=prod, alive_by_step=alive,
    )
    src = torch.tensor([0])
    cur = torch.tensor([20.0])
    h_eff = torch.full((), float(H))
    assert safe_drain(status, source_idx=src, source_ships=cur,
                      H_eff=h_eff, player_id=0)[0].item() == 20.0
    assert safe_drain(adj, source_idx=src, source_ships=cur,
                      H_eff=h_eff, player_id=0)[0].item() == 5.0


def test_beyond_horizon_arrival_only_debits_source(pp_main):
    status, prod, alive, H = _status(pp_main, [0, 1, 1], [20, 5, 30], [0, 0, 0])
    adj = pp_main._background_adjusted_status(
        status, background=_bg(pp_main, 2, 1, 10, float(H + 3)),
        prod=prod, alive_by_step=alive,
    )
    assert adj.ships[2, 0].item() == 20.0         # debit applies
    assert adj.ships[1, H].item() == 5.0          # no in-horizon arrival
