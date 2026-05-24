"""Layer R — reliability multiplier oracle tests.

Verifies the closed-form reliability function decays correctly with eta,
wait_N, and pessimistic-opp-counter geometry.
"""
import math
from types import SimpleNamespace

import pytest

from lib.reliability import RELIABILITY_PRICING_ENABLED, reliability


def _world_no_opp():
    """World with no opp planets reachable → opp_reach=0 → no landing penalty."""
    return SimpleNamespace(planets_by_id={})


def _world_with_opp(opp_x: float, opp_y: float, opp_ships: int):
    """World containing one enemy planet (owner=1) at (x, y) with `opp_ships`."""
    opp = SimpleNamespace(id=99, x=opp_x, y=opp_y, owner=1, ships=opp_ships)
    return SimpleNamespace(planets_by_id={99: opp})


def _tgt(production: float = 3.0, x: float = 0.0, y: float = 0.0):
    return SimpleNamespace(id=1, x=x, y=y, production=production, owner=-1)


def test_default_no_opp_high_eta_small_fleet():
    """2-ship eta=25 vs prod=3, no opp counter → reliability dominated by
    eta_rel * landing_rel where landing = 2 - 75 = -73 → landing_rel=0."""
    rel = reliability(
        tgt=_tgt(production=3.0),
        ships=2,
        eta=25,
        wait_N=0,
        world=_world_no_opp(),
        my_id=0,
    )
    assert rel == 0.0, f"expected 0.0 (negative pessimistic), got {rel}"


def test_eta_decay_short_haul_large_fleet():
    """30-ship eta=8 vs prod=3, no opp counter:
    eta_rel = exp(-8/20) ≈ 0.670
    landing = 30 - 24 = 6; landing_rel = 6/30 = 0.2
    → reliability ≈ 0.670 * 1.0 * 0.2 ≈ 0.134."""
    rel = reliability(
        tgt=_tgt(production=3.0),
        ships=30,
        eta=8,
        wait_N=0,
        world=_world_no_opp(),
        my_id=0,
    )
    expected = math.exp(-8.0 / 20.0) * 1.0 * (6.0 / 30.0)
    assert abs(rel - expected) < 1e-6, f"got {rel}, expected {expected}"


def test_wait_decay_compounds_with_eta():
    """wait_N=15 eta=10 vs prod=0 (no production-bleed), no opp:
    eta_rel = exp(-10/20) ≈ 0.607
    wait_rel = exp(-15/10) ≈ 0.223
    landing = 10 - 0 = 10; landing_rel = 1.0
    → ≈ 0.607 * 0.223 * 1.0 ≈ 0.135."""
    rel = reliability(
        tgt=_tgt(production=0.0),
        ships=10,
        eta=10,
        wait_N=15,
        world=_world_no_opp(),
        my_id=0,
    )
    expected = math.exp(-10.0 / 20.0) * math.exp(-15.0 / 10.0) * 1.0
    assert abs(rel - expected) < 1e-6, f"got {rel}, expected {expected}"


def test_high_prod_target_kills_landing_residual():
    """20-ship eta=8 vs prod=4, no opp:
    landing = 20 - 32 = -12 → landing_rel = 0 → reliability = 0."""
    rel = reliability(
        tgt=_tgt(production=4.0),
        ships=20,
        eta=8,
        wait_N=0,
        world=_world_no_opp(),
        my_id=0,
    )
    assert rel == 0.0, f"expected 0.0 (prod eats fleet), got {rel}"


def test_opp_counter_reduces_landing_residual():
    """50-ship eta=8 vs prod=0, with opp 30 ships at (20, 0):
    Distance src→tgt = 20; fleet_speed(30) ≈ √30 ≈ 5.48; eta_opp ≈ 4.
    Opp reachable (within 8+4=12 ticks): YES → opp_reach=30.
    landing = 50 - 0 - 0.4*30 = 38; landing_rel = 38/50 = 0.76
    eta_rel = exp(-8/20) ≈ 0.670 → reliability ≈ 0.510."""
    world = _world_with_opp(opp_x=20.0, opp_y=0.0, opp_ships=30)
    rel = reliability(
        tgt=_tgt(production=0.0, x=0.0, y=0.0),
        ships=50,
        eta=8,
        wait_N=0,
        world=world,
        my_id=0,
    )
    eta_rel = math.exp(-8.0 / 20.0)
    landing_rel = (50.0 - 0.0 - 0.4 * 30.0) / 50.0
    expected = eta_rel * 1.0 * landing_rel
    assert abs(rel - expected) < 1e-6, f"got {rel}, expected {expected}"


def test_reliability_pricing_default_off():
    """Without the env var set, RELIABILITY_PRICING_ENABLED is False.
    This is the gate the proposer / opening_planner check before calling
    reliability(...)."""
    # NOTE: module-level constant reads at import time, so this test
    # asserts the *default* import state (no env var set in test env).
    # Tests that set the env var explicitly need to reload the module.
    # Here we just check the symbol exists and the default is False.
    assert RELIABILITY_PRICING_ENABLED is False


def test_reliability_returns_in_unit_interval():
    """Sanity: reliability output is always in [0, 1]."""
    cases = [
        (10, 5, 0, 2.0),
        (1, 50, 30, 5.0),
        (100, 2, 0, 1.0),
        (50, 8, 8, 0.0),
    ]
    for ships, eta, wait_N, prod in cases:
        rel = reliability(
            tgt=_tgt(production=prod),
            ships=ships,
            eta=eta,
            wait_N=wait_N,
            world=_world_no_opp(),
            my_id=0,
        )
        assert 0.0 <= rel <= 1.0, (
            f"reliability out of unit interval: {rel} "
            f"(ships={ships}, eta={eta}, wait_N={wait_N}, prod={prod})"
        )
