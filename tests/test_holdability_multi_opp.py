"""Multi-opp + recapture-cost fixed-point tests for
`_target_holdable_after_capture` (Fix 2a, 2026-05-27 plan).

The pre-fix filter picked `nearest_opp` purely by distance among opps
with `ships >= MIN_COUNTER_SHIPS`. A stronger but slightly-further opp
was silently ignored — so we'd capture targets the stronger opp could
immediately recapture.

The fix iterates every threatening opp and computes each one's true
recapture cost via a 3-iteration fixed-point on `(opp_needed,
opp_speed, t_op)`. Rejects when ANY opp can both afford and overwhelm.

Test shape mirrors `tests/test_proposer_orbital.py` — synthetic World
built via `World.from_obs`, filter called directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.baseline.proposer import _target_holdable_after_capture
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


@pytest.fixture(autouse=True)
def _orbital_safety_off(monkeypatch):
    # Keep these tests focused on the multi-opp + speed-fixed-point
    # fix; orbital safety is covered separately.
    monkeypatch.setenv("BASELINE_ORBITAL_SAFETY", "0")


def test_strong_opp_at_medium_distance_rejects_capture():
    """Reproduces the failure state of the pre-2026-05-27 nearest-only
    bug: a small nearby opp lets the filter pass, but a stronger opp
    slightly further away can actually mount a successful recapture.

    Setup:
      - src S (mine), 50 ships at (0, 50). Far from target.
      - tgt T (neutral), 5 ships at (40, 50).
      - ally A (mine), 100 ships at (20, 50).
      - opp O1, 25 ships at (45, 50) — closest opp passing
        `MIN_COUNTER_SHIPS=20` filter.
      - opp O2, 400 ships at (60, 50) — farther but vastly stronger.

    We launch 50 ships at T to capture. After arrival we'd hold a
    small garrison. The fix detects O2's recapture even though O1 is
    closer.

    Pre-fix: passes (picks O1 — too weak to recapture).
    Post-fix: rejects (O2 can recapture).
    """
    src = _planet(0, 0, 0.0, 50.0, ships=50, production=1)
    tgt = _planet(1, -1, 40.0, 50.0, ships=5, production=2)
    ally = _planet(2, 0, 20.0, 50.0, ships=100, production=1)
    opp1 = _planet(3, 1, 45.0, 50.0, ships=25, production=1)
    opp2 = _planet(4, 1, 60.0, 50.0, ships=400, production=2)

    world = _world([src, tgt, ally, opp1, opp2])
    model = WorldModel.from_world(world)

    holdable = _target_holdable_after_capture(
        src, tgt, ships=50, wait_N=0, eta=10,
        world=world, model=model, me=0,
    )
    assert holdable is False, (
        "Stronger opp O2 should have been detected by the new all-opp "
        "iteration; pre-fix nearest-only logic would have picked weak O1."
    )


def test_no_threatening_opps_allows_capture():
    """When no opp has `ships >= MIN_COUNTER_SHIPS=20`, the filter
    returns True regardless of distance."""
    src = _planet(0, 0, 0.0, 50.0, ships=50, production=2)
    tgt = _planet(1, -1, 40.0, 50.0, ships=5, production=2)
    weak_opp = _planet(2, 1, 45.0, 50.0, ships=5, production=1)

    world = _world([src, tgt, weak_opp])
    model = WorldModel.from_world(world)

    holdable = _target_holdable_after_capture(
        src, tgt, ships=50, wait_N=0, eta=10,
        world=world, model=model, me=0,
    )
    assert holdable is True


def test_ally_closer_than_all_opps_allows_capture():
    """The 'ally closer than every threatening opp' global accept
    clause survives the rewrite."""
    src = _planet(0, 0, 0.0, 50.0, ships=80, production=2)
    tgt = _planet(1, -1, 50.0, 50.0, ships=5, production=2)
    nearby_ally = _planet(2, 0, 48.0, 50.0, ships=200, production=2)
    distant_opp = _planet(3, 1, 95.0, 50.0, ships=300, production=2)

    world = _world([src, tgt, nearby_ally, distant_opp])
    model = WorldModel.from_world(world)

    holdable = _target_holdable_after_capture(
        src, tgt, ships=80, wait_N=0, eta=10,
        world=world, model=model, me=0,
    )
    assert holdable is True


def test_reinforce_own_planet_always_holdable():
    """Filter no-op for `tgt.owner == me` (reinforce path)."""
    src = _planet(0, 0, 0.0, 50.0, ships=50, production=2)
    tgt = _planet(1, 0, 40.0, 50.0, ships=10, production=2)
    opp = _planet(2, 1, 45.0, 50.0, ships=400, production=2)

    world = _world([src, tgt, opp])
    model = WorldModel.from_world(world)

    holdable = _target_holdable_after_capture(
        src, tgt, ships=20, wait_N=0, eta=10,
        world=world, model=model, me=0,
    )
    assert holdable is True
