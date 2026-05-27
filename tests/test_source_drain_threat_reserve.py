"""Garrison-reserve clause inside `_source_survives_launch_legacy`
(2026-05-27 plan, Fix B).

When `BASELINE_THREAT_RESERVE_ALPHA > 0`, the legacy predicate adds a
reserve check: `residue_after_launch >= ceil(ALPHA * predicted_threat_force(
src, WINDOW))`. Default ALPHA=0.0 → block skipped → identical to legacy
decision.

The reserve catches rotating-opp waves the legacy in-flight-ledger
`threat_force` sum misses.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.baseline import proposer as _proposer
from agents.baseline.proposer import _source_survives_launch_legacy
from lib.intent import World
from lib.world_model import WorldModel


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


def test_reserve_default_off_no_op(monkeypatch):
    """ALPHA=0.0 (default) → clause skipped, decision identical to
    pre-fix legacy. The setup is a launch with no inbound threat —
    legacy returns True; new code must also return True."""
    monkeypatch.setattr(_proposer, "THREAT_RESERVE_ALPHA", 0.0)
    src = _planet(0, 0, 50.0, 50.0, ships=30, production=3)
    # A nearby strong opp exists but ALPHA=0.0 → ignored.
    strong_opp = _planet(1, 1, 60.0, 50.0, ships=200, production=2)
    world = _world([src, strong_opp])
    model = WorldModel.from_world(world)

    survives = _source_survives_launch_legacy(
        src, ships=20, wait_N=0, world=world, model=model, me=0,
    )
    # Legacy verdict: no in-flight ledger force, threat_eta might be
    # set, threat_force from ledger is 0 → return True at the
    # `if threat_force <= 0: return True` line.
    assert survives is True


def test_reserve_rejects_when_residue_below_alpha_predicted(monkeypatch):
    """ALPHA=1.0, predicted=200 (strong opp within window), launch
    leaves residue=10 → 10 < 200 → reject."""
    monkeypatch.setattr(_proposer, "THREAT_RESERVE_ALPHA", 1.0)
    monkeypatch.setattr(_proposer, "THREAT_RESERVE_WINDOW", 30)
    src = _planet(0, 0, 50.0, 50.0, ships=30, production=3)
    strong_opp = _planet(1, 1, 60.0, 50.0, ships=200, production=2)
    world = _world([src, strong_opp])
    model = WorldModel.from_world(world)

    # Launch 20 → residue = 30 - 20 = 10. Predicted = 200. 10 < 200 → reject.
    survives = _source_survives_launch_legacy(
        src, ships=20, wait_N=0, world=world, model=model, me=0,
    )
    assert survives is False, (
        "ALPHA=1.0 with predicted=200 and residue=10 should reject; "
        "legacy would have returned True (no in-flight ledger threat)."
    )

    # Sanity: at ALPHA=0.0 the same setup returns True.
    monkeypatch.setattr(_proposer, "THREAT_RESERVE_ALPHA", 0.0)
    survives_noop = _source_survives_launch_legacy(
        src, ships=20, wait_N=0, world=world, model=model, me=0,
    )
    assert survives_noop is True


def test_reserve_admits_when_residue_above_alpha_predicted(monkeypatch):
    """ALPHA=0.5, predicted=40, residue=80 → 80 >= ceil(0.5*40)=20 → accept."""
    monkeypatch.setattr(_proposer, "THREAT_RESERVE_ALPHA", 0.5)
    monkeypatch.setattr(_proposer, "THREAT_RESERVE_WINDOW", 30)
    src = _planet(0, 0, 50.0, 50.0, ships=100, production=3)
    weak_opp = _planet(1, 1, 60.0, 50.0, ships=40, production=2)
    world = _world([src, weak_opp])
    model = WorldModel.from_world(world)

    # Launch 20 → residue = 100 - 20 = 80. Predicted = 40. ceil(0.5*40)=20.
    # 80 >= 20 → reserve clause passes. Legacy then runs and accepts.
    survives = _source_survives_launch_legacy(
        src, ships=20, wait_N=0, world=world, model=model, me=0,
    )
    assert survives is True
