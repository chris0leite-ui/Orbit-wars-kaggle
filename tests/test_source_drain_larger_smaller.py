"""Larger-to-smaller source-drain protection (Fix 3, 2026-05-27 plan).

Pre-fix: `_source_survives_launch` returned True whenever there was
no in-flight enemy threat in the ledger — even for a high-prod home
draining itself to capture a tiny low-prod neutral, with a fully-
loaded opp parked at medium distance ready to launch.

Post-fix: when `src.production > tgt.production`, three extra
clauses fire:
  A. Stockpile floor (residue >= `STOCKPILE_PROD_MULT × src.prod`).
  B. Stricter survival margin (`SAFETY_MARGIN_DRAIN × threat_force`).
  C. Potential-launch coverage (folds biggest opp's garrison into
     threat estimate).

The companion "smaller-to-larger" case (src.prod < tgt.prod) does
NOT trigger the hardening — verifies the gate doesn't over-restrict.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.baseline.proposer import _source_survives_launch
from lib.intent import World
from lib.world_model import WorldModel


@pytest.fixture(autouse=True)
def _opt_in_to_drain_harden(monkeypatch):
    # `_source_survives_launch` defaults to legacy (2026-05-27 revert
    # after sub 53083109 panel FAIL). These tests exercise the
    # larger→smaller hardened variant — opt-in via
    # BASELINE_DRAIN_HARDEN=1.
    monkeypatch.setenv("BASELINE_DRAIN_HARDEN", "1")


def _planet(pid, owner, x, y, *, ships=20, production=2, radius=1.5):
    return SimpleNamespace(
        id=pid, owner=owner, x=x, y=y, radius=radius,
        ships=ships, production=production,
    )


def _world(planets, my_id=0):
    obs = {
        "player": my_id,
        "planets": [
            [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
            for p in planets
        ],
        "fleets": [],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": 0,
    }
    return World.from_obs(obs)


def test_larger_to_smaller_with_strong_opp_rejects_drain():
    """Failure state pre-fix: high-prod source S (prod=5, 30 ships)
    launches at a tiny prod=1 neutral, no in-flight threat in the
    ledger — `_source_survives_launch` returns True. But there's a
    400-ship opp at distance 40 that could trivially capture our
    drained S.

    Post-fix: src.prod (5) > tgt.prod (1) → larger→smaller gate fires.
    Residue after launching 28 ships = 2; stockpile floor = 5*5 = 25.
    Clause A rejects.
    """
    src = _planet(0, 0, 50.0, 50.0, ships=30, production=5)
    tgt = _planet(1, -1, 70.0, 50.0, ships=5, production=1)
    strong_opp = _planet(2, 1, 10.0, 50.0, ships=400, production=3)

    world = _world([src, tgt, strong_opp])
    model = WorldModel.from_world(world)

    survives = _source_survives_launch(
        src, ships=28, wait_N=0,
        world=world, model=model, me=0,
        tgt=tgt,
    )
    assert survives is False, (
        "Larger→smaller drain with strong opp present should reject "
        "via Clause A (stockpile floor)."
    )


def test_smaller_to_larger_does_not_trigger_hardening():
    """Sanity-check the gate: when src.prod < tgt.prod, the extra
    clauses do NOT fire. Launch passes (no in-flight ledger threat)
    via the original return-True path. Verifies we don't
    over-restrict the common 'feeder reinforces front line' pattern.
    """
    src = _planet(0, 0, 50.0, 50.0, ships=30, production=1)
    tgt = _planet(1, -1, 70.0, 50.0, ships=5, production=5)
    strong_opp = _planet(2, 1, 10.0, 50.0, ships=400, production=3)

    world = _world([src, tgt, strong_opp])
    model = WorldModel.from_world(world)

    # Without tgt, behaves like pre-fix.
    survives_no_tgt = _source_survives_launch(
        src, ships=28, wait_N=0,
        world=world, model=model, me=0,
    )
    # With tgt, the gate is FALSE (src.prod 1 < tgt.prod 5), so
    # hardening doesn't fire — same answer.
    survives_with_tgt = _source_survives_launch(
        src, ships=28, wait_N=0,
        world=world, model=model, me=0,
        tgt=tgt,
    )
    assert survives_no_tgt == survives_with_tgt


def test_larger_to_smaller_low_drain_survives_stockpile_floor():
    """Counterpart: a *small* launch from a larger source clears the
    stockpile floor and is allowed."""
    src = _planet(0, 0, 50.0, 50.0, ships=80, production=5)
    tgt = _planet(1, -1, 70.0, 50.0, ships=5, production=1)
    # No opp in this World — no potential threat to flag.
    world = _world([src, tgt])
    model = WorldModel.from_world(world)

    # residue = 80 - 10 = 70; stockpile_floor = 5*5 = 25 → OK
    survives = _source_survives_launch(
        src, ships=10, wait_N=0,
        world=world, model=model, me=0,
        tgt=tgt,
    )
    assert survives is True


def test_tgt_none_preserves_pre_fix_behavior():
    """`_source_survives_launch(src, ships, wait_N, world, model, me)`
    without `tgt` keeps the original semantics (test fixtures that
    pre-date the signature change shouldn't break).
    """
    src = _planet(0, 0, 50.0, 50.0, ships=30, production=5)
    strong_opp = _planet(2, 1, 10.0, 50.0, ships=400, production=3)
    world = _world([src, strong_opp])
    model = WorldModel.from_world(world)

    # tgt=None → no larger→smaller hardening; potential-launch threat
    # has no in-flight ledger entry, so original behavior returns True.
    survives = _source_survives_launch(
        src, ships=28, wait_N=0,
        world=world, model=model, me=0,
    )
    assert survives is True
