"""Tests for the terminal production value (PRODUCER_PLUS_TERMINAL_PROD_VALUE).

The mechanism: the in-horizon flow terms truncate a captured planet's payoff
at H, so a neutral whose production only repays its garrison cost in-horizon
scores ~0 and never clears the roi threshold. The sparse flow diff now also
reports production owned at the horizon's final step (hypothetical − current,
per player); `competitive_score` credits it for `terminal_prod_weight`
post-horizon steps. Weight 0 (default) leaves every score bit-identical.
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
        "producer_plus_main_test_termval",
        os.path.join(PRODUCER_PLUS_DIR, "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["producer_plus_main_test_termval"] = module
    spec.loader.exec_module(module)
    return module


def test_env_default_off(monkeypatch, pp_main):
    monkeypatch.delenv("PRODUCER_PLUS_TERMINAL_PROD_VALUE", raising=False)
    assert pp_main._terminal_prod_value() == 0.0


def test_env_parses_float(monkeypatch, pp_main):
    monkeypatch.setenv("PRODUCER_PLUS_TERMINAL_PROD_VALUE", "12")
    assert pp_main._terminal_prod_value() == 12.0
    monkeypatch.setenv("PRODUCER_PLUS_TERMINAL_PROD_VALUE", "garbage")
    assert pp_main._terminal_prod_value() == 0.0
    monkeypatch.setenv("PRODUCER_PLUS_TERMINAL_PROD_VALUE", "-3")
    assert pp_main._terminal_prod_value() == 0.0


# ---------------------------------------------------------------------------
# Sparse flow diff: synthetic neutral capture
# ---------------------------------------------------------------------------


def _neutral_capture_fixture():
    """P=2, A=2, H=4. Planet 0 mine (prod 1, 20 ships), planet 1 neutral
    (garrison 5, prod 2). One candidate: send 6 from P0 to P1, eta 2 —
    flips the neutral at tick 2 and holds it through the horizon."""
    from orbit_lite.movement import PlanetGarrisonStatus
    from orbit_lite.planner_core import make_launch_set

    P, A, H = 2, 2, 4
    dtype = torch.float32
    owner = torch.tensor([[0] * (H + 1), [-1] * (H + 1)], dtype=torch.long)
    ships = torch.tensor([[20.0] * (H + 1), [5.0] * (H + 1)], dtype=dtype)
    status = PlanetGarrisonStatus(
        owner=owner,
        ships=ships,
        pre_combat_owner=owner.clone(),
        pre_combat_ships=ships.clone(),
        arrivals_by_owner=torch.zeros(P, H + 1, A, dtype=dtype),
    )
    prod = torch.tensor([1.0, 2.0], dtype=dtype)
    alive_by_step = torch.ones(H + 1, P, dtype=torch.bool)
    launches = make_launch_set(
        source_slots=torch.tensor([[0]]),
        target_slots=torch.tensor([[1]]),
        ships=torch.tensor([[6.0]], dtype=dtype),
        eta=torch.tensor([[2.0]], dtype=dtype),
        valid=torch.tensor([[True]]),
        player_id=0,
    )
    return status, prod, alive_by_step, launches


def test_sparse_diff_reports_terminal_prod_delta():
    from orbit_lite.garrison_launch import sparse_launch_flow_delta

    status, prod, alive_by_step, launches = _neutral_capture_fixture()
    diff = sparse_launch_flow_delta(
        status, prod=prod, alive_by_step=alive_by_step,
        player_count=2, launches=launches, player_id=0,
    )
    assert diff.terminal_prod_delta is not None
    # Player 0 owns the prod-2 neutral at the horizon's final step under the
    # hypothetical; the baseline owner is neutral, so nobody loses anything.
    assert diff.terminal_prod_delta.tolist() == [[2.0, 0.0]]


def test_competitive_score_weight_zero_is_legacy():
    from orbit_lite.garrison_launch import sparse_launch_flow_delta
    from orbit_lite.planner_core import competitive_score

    status, prod, alive_by_step, launches = _neutral_capture_fixture()
    diff = sparse_launch_flow_delta(
        status, prod=prod, alive_by_step=alive_by_step,
        player_count=2, launches=launches, player_id=0,
    )
    base = competitive_score(diff, player_id=0)
    gated = competitive_score(diff, player_id=0, terminal_prod_weight=0.0)
    assert torch.equal(base, gated)


def test_competitive_score_credits_post_horizon_production():
    from orbit_lite.garrison_launch import sparse_launch_flow_delta
    from orbit_lite.planner_core import competitive_score

    status, prod, alive_by_step, launches = _neutral_capture_fixture()
    diff = sparse_launch_flow_delta(
        status, prod=prod, alive_by_step=alive_by_step,
        player_count=2, launches=launches, player_id=0,
    )
    base = competitive_score(diff, player_id=0)
    lifted = competitive_score(diff, player_id=0, terminal_prod_weight=12.0)
    # Captured prod 2, credited for 12 post-horizon steps, no opponent term.
    assert torch.allclose(lifted - base, torch.tensor([24.0]))


def test_enemy_capture_counts_double():
    """Taking an ENEMY prod planet moves it from their column to ours, so the
    competitive (me − opp) terminal term is worth 2× the production."""
    from orbit_lite.movement import PlanetGarrisonStatus
    from orbit_lite.garrison_launch import sparse_launch_flow_delta
    from orbit_lite.planner_core import competitive_score, make_launch_set

    P, A, H = 2, 2, 4
    dtype = torch.float32
    owner = torch.tensor([[0] * (H + 1), [1] * (H + 1)], dtype=torch.long)
    ships = torch.tensor([[20.0] * (H + 1), [5.0] * (H + 1)], dtype=dtype)
    status = PlanetGarrisonStatus(
        owner=owner, ships=ships,
        pre_combat_owner=owner.clone(), pre_combat_ships=ships.clone(),
        arrivals_by_owner=torch.zeros(P, H + 1, A, dtype=dtype),
    )
    prod = torch.tensor([1.0, 2.0], dtype=dtype)
    alive_by_step = torch.ones(H + 1, P, dtype=torch.bool)
    launches = make_launch_set(
        source_slots=torch.tensor([[0]]),
        target_slots=torch.tensor([[1]]),
        ships=torch.tensor([[20.0]], dtype=dtype),
        eta=torch.tensor([[2.0]], dtype=dtype),
        valid=torch.tensor([[True]]),
        player_id=0,
    )
    diff = sparse_launch_flow_delta(
        status, prod=prod, alive_by_step=alive_by_step,
        player_count=2, launches=launches, player_id=0,
    )
    assert diff.terminal_prod_delta.tolist() == [[2.0, -2.0]]
    base = competitive_score(diff, player_id=0)
    lifted = competitive_score(diff, player_id=0, terminal_prod_weight=10.0)
    assert torch.allclose(lifted - base, torch.tensor([40.0]))


# ---------------------------------------------------------------------------
# Class-split overkill (PRODUCER_PLUS_OVERKILL_FACTOR_ENEMY)
# ---------------------------------------------------------------------------


def test_split_overkill_default_scalar(monkeypatch, pp_main):
    monkeypatch.delenv("PRODUCER_PLUS_OVERKILL_FACTOR_ENEMY", raising=False)
    monkeypatch.setenv("PRODUCER_PLUS_OVERKILL_FACTOR", "2.0")
    assert pp_main._overkill_factor_enemy() is None

    class _Obs:
        P = 3
        is_enemy = torch.tensor([False, True, False])
        ships = torch.tensor([1.0, 1.0, 1.0])

    out = pp_main._overkill_for_targets(_Obs(), torch.tensor([0, 1, 2]), 2, torch.float32)
    assert out == 2.0  # scalar legacy path


def test_split_overkill_per_target(monkeypatch, pp_main):
    monkeypatch.setenv("PRODUCER_PLUS_OVERKILL_FACTOR", "1.3")
    monkeypatch.setenv("PRODUCER_PLUS_OVERKILL_FACTOR_ENEMY", "4.0")
    monkeypatch.delenv("PRODUCER_PLUS_MASS_2P_ONLY", raising=False)

    class _Obs:
        P = 3
        is_enemy = torch.tensor([False, True, False])  # neutral, enemy, own
        ships = torch.tensor([1.0, 1.0, 1.0])

    out = pp_main._overkill_for_targets(_Obs(), torch.tensor([0, 1, 2]), 2, torch.float32)
    assert torch.allclose(out, torch.tensor([1.3, 4.0, 1.3]))
