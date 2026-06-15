"""Unit tests for the tenure / durability leaf-scorer term.

Decisive test: an unholdable contested capture is penalised, but the SAME
capture with a friendly reinforcer in reach gets no penalty — reinforcement
closes the exposure. That reinforcement-reach is what makes this a *tenure*
("can we hold it") term rather than a pure enemy-threat penalty.

Spec: audit/2026-06-15-tenure-durability-spec.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_ORBIT_LITE_PARENT = Path(__file__).resolve().parents[1] / "agents" / "producer"
if str(_ORBIT_LITE_PARENT) not in sys.path:
    sys.path.insert(0, str(_ORBIT_LITE_PARENT))

from orbit_lite.distance_cache import DistanceCache  # noqa: E402
from orbit_lite.durability import tenure_penalty  # noqa: E402


# Board:
#   P0 home/source (90,90) ours, 50 ships — too far from T to reinforce it
#   P1 target T    (50,30) neutral, 10 ships
#   P2 enemy       (50,25) enemy, 100 ships — 5 units from T (reaches it fast)
#   P3 reinforcer  (50,35) ours OR neutral, 100 ships — 5 units from T
_POS = torch.tensor(
    [[90.0, 90.0], [50.0, 30.0], [50.0, 25.0], [50.0, 35.0]]
)
_P = 4
_K = 10


def _cache():
    d = torch.cdist(_POS, _POS)
    cross = d.unsqueeze(0).expand(_K + 1, _P, _P).contiguous()
    return DistanceCache(cross_dist=cross, alive_by_step=torch.ones(_K + 1, _P, dtype=torch.bool), K=_K)


def _ships():
    return torch.tensor([50.0, 10.0, 100.0, 100.0])


def _obs(*, reinforcer_ours: bool, enemy_present: bool = True):
    owned = torch.tensor([True, False, False, reinforcer_ours])
    is_enemy = torch.tensor([False, False, enemy_present, False])
    is_neutral = ~owned & ~is_enemy
    return SimpleNamespace(
        device=torch.device("cpu"), ships=_ships(), P=_P,
        alive=torch.ones(_P, dtype=torch.bool),
        owned=owned, is_enemy=is_enemy, is_neutral=is_neutral,
    )


def _pen(*, reinforcer_ours=False, enemy_present=True, weight=1.0, send=20.0,
         floor=10.0, W=8):
    cap_floor = torch.full((1, _K), float(floor))
    return tenure_penalty(
        obs=_obs(reinforcer_ours=reinforcer_ours, enemy_present=enemy_present),
        cache=_cache(),
        garrison_status=SimpleNamespace(owner=torch.full((_P, _K + 1), -1.0)),
        cand_tgt_slot=torch.tensor([1]),
        cand_tgt_short=torch.tensor([0]),
        cand_send=torch.tensor([[send]]),
        cand_eta=torch.tensor([[2.0]]),
        cand_valid=torch.tensor([True]),
        cand_is_def=torch.tensor([False]),
        capture_floor_TK=cap_floor,
        prod=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        H=18, W=W,
        hold_fraction=0.5, safety_reserve=0.5,
        weight=weight, player_id=0,
    )


def test_unholdable_capture_is_penalised():
    # Strong enemy nearby, no friendly reinforcement -> exposure > 0 -> penalty.
    assert _pen(reinforcer_ours=False)[0].item() > 0.0


def test_reinforcement_closes_exposure():
    # Same enemy threat, but a friendly reinforcer in reach -> exposure 0 -> no
    # penalty. This is the tenure novelty over a pure threat penalty.
    assert _pen(reinforcer_ours=True)[0].item() == 0.0
    assert _pen(reinforcer_ours=False)[0].item() > _pen(reinforcer_ours=True)[0].item()


def test_no_enemy_no_penalty():
    assert _pen(reinforcer_ours=False, enemy_present=False)[0].item() == 0.0


def test_disabled_when_weight_zero():
    assert _pen(reinforcer_ours=False, weight=0.0)[0].item() == 0.0


def test_no_penalty_without_capture():
    # Send below the capture floor -> not a capture -> no penalty.
    assert _pen(reinforcer_ours=False, send=5.0, floor=10.0)[0].item() == 0.0


def test_window_zero_disables():
    # A zero hold window leaves nothing reachable -> no exposure computed.
    assert _pen(reinforcer_ours=False, W=0)[0].item() == 0.0
