"""Smart-dropout scorer math (PRODUCER_PLUS_DROPOUT).

Dropout re-scores a candidate in a world where the planet it captures is
reflipped to the opponent a few turns later, injected as a CREDIT-ONLY enemy
arrival (source slot -1 so the scorer's source-validity gate drops the debit —
no friendly planet is drained). This test pins the two load-bearing claims:

1. The reflip leg is credit-only: appending it never debits our source planet.
2. A thin, recapturable capture scores STRICTLY LOWER once the reflip is
   applied — i.e. dropout penalises captures we cannot hold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "agents" / "producer"))

from orbit_lite.garrison_launch import (  # noqa: E402
    LaunchSet,
    _run_exact_recurrence,
    sparse_launch_flow_delta,
)
from orbit_lite.movement import PlanetGarrisonStatus  # noqa: E402
from orbit_lite.planner_core import competitive_score  # noqa: E402


def _do_nothing_status(*, init_owner, init_ships, prod, H, A):
    """Build a do-nothing PlanetGarrisonStatus (no new launches) for a board."""
    P = init_owner.shape[0]
    owner0 = init_owner.view(1, P)
    ships0 = init_ships.view(1, P)
    prod1 = prod.view(1, P)
    alive = torch.ones(1, P, H + 1, dtype=torch.bool)
    arrivals = torch.zeros(1, P, H, A)
    o, s, po, ps = _run_exact_recurrence(
        init_owner=owner0, init_ships=ships0, prod=prod1, alive=alive, arrivals=arrivals,
    )
    return PlanetGarrisonStatus(
        owner=o[0], ships=s[0], pre_combat_owner=po[0], pre_combat_ships=ps[0],
        arrivals_by_owner=torch.zeros(P, H + 1, A),
    ), alive[0].permute(1, 0).contiguous()  # alive_by_step [H+1, P]


def _score(status, alive_by_step, prod, launches, A):
    diff = sparse_launch_flow_delta(
        status, prod=prod, alive_by_step=alive_by_step, player_count=A,
        launches=launches, player_id=0,
    )
    return float(competitive_score(diff, player_id=0))


def test_reflip_is_credit_only_and_penalises_thin_capture():
    H, A = 10, 2
    # Planet 0 = mine (20 ships, prod 1); planet 1 = neutral (3 ships, prod 1).
    init_owner = torch.tensor([0, -1], dtype=torch.long)
    init_ships = torch.tensor([20.0, 3.0])
    prod = torch.tensor([1.0, 1.0])
    status, alive_by_step = _do_nothing_status(
        init_owner=init_owner, init_ships=init_ships, prod=prod, H=H, A=A,
    )

    # Capture leg: send 5 from planet 0 to neutral planet 1, arriving step 1.
    cap = LaunchSet(
        source_slots=torch.tensor([[0]]), target_slots=torch.tensor([[1]]),
        ships=torch.tensor([[5.0]]), eta=torch.tensor([[1.0]]),
        owner=torch.tensor([[0]]), valid=torch.tensor([[True]]),
    )
    score_clean = _score(status, alive_by_step, prod, cap, A)

    # Same capture + a credit-only enemy reflip of planet 1 at step 4 (source
    # slot -1 -> no friendly debit). Enough ships to flip the held garrison.
    cap_drop = LaunchSet(
        source_slots=torch.tensor([[0, -1]]), target_slots=torch.tensor([[1, 1]]),
        ships=torch.tensor([[5.0, 12.0]]), eta=torch.tensor([[1.0, 4.0]]),
        owner=torch.tensor([[0, 1]]), valid=torch.tensor([[True, True]]),
    )
    score_drop = _score(status, alive_by_step, prod, cap_drop, A)

    # Clean capture of a free neutral is worth something; the reflip makes it
    # worth strictly less (we lose the planet's later production + our garrison).
    assert score_clean > 0.0
    assert score_drop < score_clean

    # Credit-only check: a lone reflip leg (source -1) must not debit planet 0.
    # Compare our produced ships with the reflip-only world vs do-nothing: the
    # source planet's own production stream is untouched (no phantom drain).
    reflip_only = LaunchSet(
        source_slots=torch.tensor([[-1]]), target_slots=torch.tensor([[1]]),
        ships=torch.tensor([[12.0]]), eta=torch.tensor([[4.0]]),
        owner=torch.tensor([[1]]), valid=torch.tensor([[True]]),
    )
    diff = sparse_launch_flow_delta(
        status, prod=prod, alive_by_step=alive_by_step, player_count=A,
        launches=reflip_only, player_id=0,
    )
    # Planet 0 is never an affected (debited) planet, so our PRODUCED ships are
    # unchanged by the reflip (only the neutral->enemy flip on planet 1 moves
    # the combat/opponent terms). produced_delta for player 0 must be ~0.
    assert abs(float(diff.ships_produced_delta[0, 0])) < 1e-5
