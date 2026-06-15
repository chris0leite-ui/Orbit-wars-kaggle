"""Unit tests for the frontier / gateway value leaf-scorer term.

The term credits a capture for the NEW neutral frontier it unlocks as a launch
base. The decisive test: a *gateway* neutral (the only planet from which a
high-value back cluster becomes reachable) must score strictly higher than an
equal-income *dead-end* neutral that unlocks nothing new.

Spec: audit/2026-06-15-frontier-gateway-value-spec.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# orbit_lite uses intra-package relative imports; put its parent on the path so
# it imports as a package (mirrors producer_agent.py's sys.path injection).
_ORBIT_LITE_PARENT = Path(__file__).resolve().parents[1] / "agents" / "producer"
if str(_ORBIT_LITE_PARENT) not in sys.path:
    sys.path.insert(0, str(_ORBIT_LITE_PARENT))

from orbit_lite.distance_cache import DistanceCache  # noqa: E402
from orbit_lite.strategic_value import (  # noqa: E402
    _reach_eta_matrix,
    frontier_bonus,
)


# ---------------------------------------------------------------------------
# Synthetic board
# ---------------------------------------------------------------------------
#   P0 home  (10, 50)  owned by us
#   P1 gateway (35, 50) neutral, prod 1 — reaches the back cluster P3/P4
#   P2 dead-end (10, 75) neutral, prod 1 — reaches nothing the home doesn't
#   P3 backA  (60, 50)  neutral, prod 5 — too far from home, reachable via P1
#   P4 backB  (60, 40)  neutral, prod 5 — too far from home, reachable via P1
#
# Reach budget R=12 turns at nominal speed c=3 → max static reach 36 units.
#   home->P1 = 25 (reach, eta 9)   home->P3 = 50 (NO)   home->P4 = 51 (NO)
#   P1->P3   = 25 (reach, eta 9)   P1->P4   = 26.9 (reach, eta 9)
#   P2->P3   = 55.9 (NO)           P2->P4   = 61 (NO)

_POS = torch.tensor(
    [[10.0, 50.0], [35.0, 50.0], [10.0, 75.0], [60.0, 50.0], [60.0, 40.0]]
)
_P = 5
_K = 12


def _distance_cache() -> DistanceCache:
    """Static-board cache: cross_dist[k] is the same pairwise distance for all k."""
    d = torch.cdist(_POS, _POS)                      # [P, P]
    cross = d.unsqueeze(0).expand(_K + 1, _P, _P).contiguous()
    alive = torch.ones(_K + 1, _P, dtype=torch.bool)
    return DistanceCache(cross_dist=cross, alive_by_step=alive, K=_K)


def _obs():
    is_neutral = torch.tensor([False, True, True, True, True])
    return SimpleNamespace(
        device=torch.device("cpu"),
        ships=torch.zeros(_P),                       # only .dtype is read
        P=_P,
        is_neutral=is_neutral,
        is_enemy=torch.zeros(_P, dtype=torch.bool),
        alive=torch.ones(_P, dtype=torch.bool),
        owned=torch.tensor([True, False, False, False, False]),
    )


def _garrison_status_all_neutral():
    # _compute_captures reads only .owner; -1 everywhere => never owned by us,
    # so a sufficient send always counts as a capture.
    return SimpleNamespace(owner=torch.full((_P, _K + 1), -1.0))


_PROD = torch.tensor([1.0, 1.0, 1.0, 5.0, 5.0])


def _bonus(*, weight=0.05, reach=12, speed=3.0, contest=0.0,
           include_enemy=False, floor=1.0, send=10.0):
    """frontier_bonus for two candidates: capture P1 (gateway), capture P2 (dead-end)."""
    capture_floor_TK = torch.full((2, _K), float(floor))
    return frontier_bonus(
        obs=_obs(),
        cache=_distance_cache(),
        garrison_status=_garrison_status_all_neutral(),
        cand_tgt_slot=torch.tensor([1, 2]),
        cand_tgt_short=torch.tensor([0, 1]),
        cand_send=torch.tensor([[send], [send]]),
        cand_eta=torch.tensor([[2.0], [2.0]]),
        cand_valid=torch.tensor([True, True]),
        cand_is_def=torch.tensor([False, False]),
        capture_floor_TK=capture_floor_TK,
        prod=_PROD,
        H=18,
        current_step=0,
        game_length_est=200,
        weight=weight,
        reach_turns=reach,
        nominal_speed=speed,
        contest_weight=contest,
        include_enemy=include_enemy,
        player_id=0,
    )


# ---------------------------------------------------------------------------
# _reach_eta_matrix
# ---------------------------------------------------------------------------


def test_reach_matrix_reaches_near_blocks_far():
    eta, reach = _reach_eta_matrix(
        _distance_cache(), reach_turns=12, nominal_speed=3.0,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert bool(reach[0, 1]) and eta[0, 1].item() == 9.0      # home -> gateway
    assert not bool(reach[0, 3])                              # home -> backA (too far)
    assert bool(reach[1, 3]) and eta[1, 3].item() == 9.0      # gateway -> backA
    assert bool(reach[1, 4])                                  # gateway -> backB
    assert not bool(reach[2, 3])                              # dead-end -> backA (too far)


def test_reach_matrix_respects_budget():
    # Tighter budget (8 turns * speed 3 = 24 < 25): the gateway can no longer
    # reach the back cluster.
    _, reach = _reach_eta_matrix(
        _distance_cache(), reach_turns=8, nominal_speed=3.0,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert not bool(reach[1, 3])


# ---------------------------------------------------------------------------
# frontier_bonus — the gateway must beat the dead-end
# ---------------------------------------------------------------------------


def test_gateway_scores_strictly_above_dead_end():
    bonus = _bonus()
    assert bonus[0].item() > 0.0          # gateway unlocks the back cluster
    assert bonus[1].item() == 0.0         # dead-end unlocks nothing new
    assert bonus[0].item() > bonus[1].item()


def test_gateway_bonus_magnitude():
    # fv(P1) = prod[P3]*disc + prod[P4]*disc, disc = 1 - 9/13 = 40/130 per target
    # = 10 * (40/130) = 400/130; * future_h(182) * weight(0.05).
    # 182/13 = 14 exactly => 10 * 40/13 * 182 * 0.05 = 40 * 14 * 0.05 = 28.0.
    bonus = _bonus()
    assert bonus[0].item() == pytest.approx(28.0, abs=1e-3)


def test_disabled_when_weight_zero():
    bonus = _bonus(weight=0.0)
    assert torch.count_nonzero(bonus).item() == 0


def test_no_bonus_without_capture():
    # Send below the capture floor => not a capture => no frontier credit even
    # for the gateway.
    bonus = _bonus(floor=100.0, send=10.0)
    assert torch.count_nonzero(bonus).item() == 0


def test_tight_budget_zeros_gateway():
    # With reach budget 8 the gateway no longer unlocks the far cluster.
    bonus = _bonus(reach=8)
    assert bonus[0].item() == 0.0


def test_contest_is_noop_without_enemy():
    # With no enemy on the board, the contest knob can never trigger.
    assert _bonus(contest=0.0)[0].item() == pytest.approx(
        _bonus(contest=0.9)[0].item(), abs=1e-3
    )


# --- contest board: an enemy base reaches the unlocked cluster first ---------
#   Q0 home   (10, 50) owned
#   Q1 gateway(35, 50) neutral — unlocks Q2
#   Q2 cluster(60, 50) neutral, prod 5 — far from home, reached via gateway
#   Q3 enemy  (60, 55) enemy — 5 units from Q2, reaches it in eta 2 (< gateway's 9)
_QPOS = torch.tensor(
    [[10.0, 50.0], [35.0, 50.0], [60.0, 50.0], [60.0, 55.0]]
)
_QP = 4


def _contest_bonus(contest: float):
    d = torch.cdist(_QPOS, _QPOS)
    cross = d.unsqueeze(0).expand(_K + 1, _QP, _QP).contiguous()
    cache = DistanceCache(
        cross_dist=cross,
        alive_by_step=torch.ones(_K + 1, _QP, dtype=torch.bool),
        K=_K,
    )
    obs = SimpleNamespace(
        device=torch.device("cpu"),
        ships=torch.zeros(_QP),
        P=_QP,
        is_neutral=torch.tensor([False, True, True, False]),
        is_enemy=torch.tensor([False, False, False, True]),
        alive=torch.ones(_QP, dtype=torch.bool),
        owned=torch.tensor([True, False, False, False]),
    )
    return frontier_bonus(
        obs=obs, cache=cache,
        garrison_status=SimpleNamespace(owner=torch.full((_QP, _K + 1), -1.0)),
        cand_tgt_slot=torch.tensor([1]),          # capture the gateway Q1
        cand_tgt_short=torch.tensor([0]),
        cand_send=torch.tensor([[10.0]]),
        cand_eta=torch.tensor([[2.0]]),
        cand_valid=torch.tensor([True]),
        cand_is_def=torch.tensor([False]),
        capture_floor_TK=torch.full((1, _K), 1.0),
        prod=torch.tensor([1.0, 1.0, 5.0, 1.0]),
        H=18, current_step=0, game_length_est=200,
        weight=0.05, reach_turns=12, nominal_speed=3.0,
        contest_weight=contest, include_enemy=False, player_id=0,
    )


def test_contest_downweights_contested_cluster():
    base = _contest_bonus(0.0)[0].item()
    contested = _contest_bonus(0.9)[0].item()
    assert base > 0.0
    assert contested == pytest.approx(0.1 * base, abs=1e-3)   # (1 - 0.9) factor
