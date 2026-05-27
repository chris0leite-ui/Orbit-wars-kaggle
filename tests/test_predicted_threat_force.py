"""Tests for `WorldModel.predicted_threat_force` (2026-05-27 plan).

Companion to `time_to_enemy_threat` — same two-layer logic (in-flight
ledger + stationary opp potential launches) but returns the FORCE
(sum of ship counts) instead of the earliest ETA. Used by
`_source_survives_launch_legacy` to size a quantitative garrison
reserve when `BASELINE_THREAT_RESERVE_ALPHA > 0`.
"""

from __future__ import annotations

from types import SimpleNamespace

from lib.intent import World
from lib.world_model import WorldModel


def _planet(pid, owner, x, y, *, ships=20, production=2, radius=1.5):
    return SimpleNamespace(
        id=pid, owner=owner, x=x, y=y, radius=radius,
        ships=ships, production=production,
    )


def _world(planets, my_id=0, fleets=None):
    obs = {
        "player": my_id,
        "planets": [
            [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
            for p in planets
        ],
        "fleets": list(fleets or []),
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": 0,
    }
    return World.from_obs(obs)


def test_no_inflight_no_stationary_returns_zero():
    """Only my own planet and a neutral on the board → predicted
    threat force is 0."""
    src = _planet(0, 0, 50.0, 50.0, ships=30, production=2)
    neutral = _planet(1, -1, 60.0, 50.0, ships=5, production=1)
    world = _world([src, neutral])
    model = WorldModel.from_world(world)

    force = model.predicted_threat_force(0, 0, world, lookahead=30)
    assert force == 0


def test_sums_inflight_within_window():
    """Two enemy fleets in the ledger at eta=5 and eta=40, window=30.
    Only the eta=5 fleet contributes."""
    src = _planet(0, 0, 50.0, 50.0, ships=30, production=2)
    world = _world([src])
    model = WorldModel.from_world(world)
    # Manually seed the ledger — eta=5 (in-window), eta=40 (out).
    model.ledger.setdefault(0, []).extend([
        (5, 1, 15),     # in-window, force=15
        (40, 1, 100),   # out-of-window, ignored
        (3, 1, 0),      # ships=0, ignored
    ])

    force = model.predicted_threat_force(0, 0, world, lookahead=30)
    assert force == 15


def test_includes_stationary_within_eta():
    """A near opp (distance ~10, default fleet_speed) is reachable
    within window → its garrison contributes (as MAX single threat).
    A far opp (distance ~60) is not → ignored. Big window: both
    reachable but only the MAX-garrison opp counts (mental model: at
    most one opp mounts a coordinated wave in the window)."""
    src = _planet(0, 0, 50.0, 50.0, ships=30, production=2)
    near_opp = _planet(1, 1, 60.0, 50.0, ships=50, production=2)
    far_opp = _planet(2, 1, 90.0, 99.0, ships=20, production=2)
    world = _world([src, near_opp, far_opp])
    model = WorldModel.from_world(world)

    # Small window: near opp included, far opp excluded.
    force_small = model.predicted_threat_force(0, 0, world, lookahead=5)
    assert force_small == 50, (
        "near_opp at distance 10 should be reachable within window=5, "
        "but far_opp at distance ~60 should not."
    )
    # Big window: both reachable → MAX single = 50 (near_opp), not 70.
    force_big = model.predicted_threat_force(0, 0, world, lookahead=500)
    assert force_big == 50


def test_excludes_own_and_neutral():
    """`predicted_threat_force` ignores own planets and neutrals even
    when they have large ship counts and are physically near."""
    src = _planet(0, 0, 50.0, 50.0, ships=30, production=2)
    ally = _planet(1, 0, 60.0, 50.0, ships=999, production=5)
    neutral = _planet(2, -1, 55.0, 50.0, ships=999, production=5)
    world = _world([src, ally, neutral])
    model = WorldModel.from_world(world)

    force = model.predicted_threat_force(0, 0, world, lookahead=100)
    assert force == 0
