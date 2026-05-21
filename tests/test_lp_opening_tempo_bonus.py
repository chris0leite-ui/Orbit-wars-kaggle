"""Pin tests for `_opening_tempo_bonus` — Phase ε option 3.

Rule 38: each behaviour is exercised both with the feature OFF (no
bonus, no LP behaviour change) and ON (bonus fires only on captures of
currently-neutral planets at step < OPENING_TEMPO_HORIZON).

Plan: /root/.knowledge-base/plans/do-it-thoroughly-consider-tingly-fox.md
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from lib.joint_solver.lp_outcome import (
    OPENING_TEMPO_FACTOR,
    OPENING_TEMPO_HORIZON,
    _opening_tempo_bonus,
)
from lib.joint_solver.outcome_table import OutcomeRow


@pytest.fixture(autouse=True)
def _clear_opening_tempo_env(monkeypatch):
    """Ensure each test starts from a known env-var state. The
    `_opening_tempo_enabled()` helper reads the env var on every call,
    so tests can safely flip it via monkeypatch without reloading
    modules (which previously polluted test_lp_endgame_predicate)."""
    monkeypatch.delenv("LP_OPENING_TEMPO", raising=False)
    yield


# ---------------------------------------------------------------------------
# Mock helpers — minimal world / row pair to exercise the bonus.
# ---------------------------------------------------------------------------


def _mock_world(planet_owner: int):
    """Tiny mock world: one planet with id=0, the requested owner."""
    p = MagicMock()
    p.id = 0
    p.owner = int(planet_owner)
    p.production = 3
    world = MagicMock()
    world.planets_by_id = {0: p}
    return world


def _mock_row(owner_T: int, my_prod: int = 100):
    """OutcomeRow with our prod_stream set; opp shown but irrelevant."""
    return OutcomeRow(
        subset=(0,),
        owner_T=int(owner_T),
        ships_T=10.0,
        prod_stream={0: int(my_prod), 1: 50},
        prod_stream_discounted={0: float(my_prod), 1: 50.0},
    )


# ---------------------------------------------------------------------------
# OFF state: no bonus regardless of inputs (Rule 38: feature OFF
# returns the pre-fix behaviour byte-identical).
# ---------------------------------------------------------------------------


def test_off_returns_zero_for_neutral_capture(monkeypatch):
    """With LP_OPENING_TEMPO unset, even an opening-tempo-qualifying
    subset returns 0.0 — no LP cost-vector perturbation."""
    monkeypatch.delenv("LP_OPENING_TEMPO", raising=False)
    # Re-import to refresh the module-level _OPENING_TEMPO_ENABLED.
    world = _mock_world(planet_owner=-1)  # currently neutral
    row = _mock_row(owner_T=0, my_prod=100)  # we end up owning
    assert _opening_tempo_bonus(
        planet_id=0, row=row, world=world, my_id=0, step_now=10,
    ) == 0.0


# ---------------------------------------------------------------------------
# ON state — bonus fires for the qualifying case.
# ---------------------------------------------------------------------------


def test_on_fires_on_opening_neutral_capture(monkeypatch):
    monkeypatch.setenv("LP_OPENING_TEMPO", "1")
    world = _mock_world(planet_owner=-1)
    row = _mock_row(owner_T=0, my_prod=100)
    got = _opening_tempo_bonus(
        planet_id=0, row=row, world=world, my_id=0, step_now=10,
    )
    expected = (OPENING_TEMPO_FACTOR - 1.0) * 100.0
    assert got == pytest.approx(expected)


# ---------------------------------------------------------------------------
# ON state — guards: each gate independently zeros the bonus.
# ---------------------------------------------------------------------------


def test_on_zero_when_past_opening_horizon(monkeypatch):
    monkeypatch.setenv("LP_OPENING_TEMPO", "1")
    world = _mock_world(planet_owner=-1)
    row = _mock_row(owner_T=0, my_prod=100)
    # step at exactly the horizon → no bonus.
    assert _opening_tempo_bonus(
        planet_id=0, row=row, world=world, my_id=0,
        step_now=OPENING_TEMPO_HORIZON,
    ) == 0.0
    # And well past it.
    assert _opening_tempo_bonus(
        planet_id=0, row=row, world=world, my_id=0,
        step_now=300,
    ) == 0.0


def test_on_zero_when_subset_doesnt_capture_for_me(monkeypatch):
    monkeypatch.setenv("LP_OPENING_TEMPO", "1")
    world = _mock_world(planet_owner=-1)
    # owner_T = opp (player 1), not me — no credit even though planet is neutral.
    row = _mock_row(owner_T=1, my_prod=100)
    assert _opening_tempo_bonus(
        planet_id=0, row=row, world=world, my_id=0, step_now=10,
    ) == 0.0


def test_on_zero_when_planet_not_currently_neutral(monkeypatch):
    """Captures of already-owned planets (ours or opp's) get no opening
    bonus — only the "snowball neutrals fast" axis is rewarded."""
    monkeypatch.setenv("LP_OPENING_TEMPO", "1")
    row = _mock_row(owner_T=0, my_prod=100)
    # Currently owned by opp → no bonus.
    world_opp = _mock_world(planet_owner=1)
    assert _opening_tempo_bonus(
        planet_id=0, row=row, world=world_opp, my_id=0, step_now=10,
    ) == 0.0
    # Currently owned by me → no bonus (reinforcement isn't tempo).
    world_mine = _mock_world(planet_owner=0)
    assert _opening_tempo_bonus(
        planet_id=0, row=row, world=world_mine, my_id=0, step_now=10,
    ) == 0.0


def test_on_zero_when_planet_missing_from_world(monkeypatch):
    """Defensive: planet_id not in world.planets_by_id → 0, no exception."""
    monkeypatch.setenv("LP_OPENING_TEMPO", "1")
    world = _mock_world(planet_owner=-1)
    row = _mock_row(owner_T=0, my_prod=100)
    # planet_id=99 not in mock world.
    assert _opening_tempo_bonus(
        planet_id=99, row=row, world=world, my_id=0, step_now=10,
    ) == 0.0


# ---------------------------------------------------------------------------
# Multiplicative-effect math sanity.
# ---------------------------------------------------------------------------


def test_on_bonus_proportional_to_my_prod_stream(monkeypatch):
    """The bonus IS (factor - 1) × my_prod_stream — so summed with the
    existing `_value_for_outcome` me_prod term, the effective me_prod
    contribution is `OPENING_TEMPO_FACTOR × prod`."""
    monkeypatch.setenv("LP_OPENING_TEMPO", "1")
    world = _mock_world(planet_owner=-1)
    factor = OPENING_TEMPO_FACTOR
    for my_prod in (0, 50, 250, 1000):
        row = _mock_row(owner_T=0, my_prod=my_prod)
        got = _opening_tempo_bonus(
            planet_id=0, row=row, world=world, my_id=0, step_now=10,
        )
        assert got == pytest.approx((factor - 1.0) * my_prod)
