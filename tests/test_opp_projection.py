"""Phase 5D tests: multi-launch opp projection.

Four fixtures exercise the core behaviors of `predict_opp_multi_launch`:

  1. test_multi_launch_count_scales_with_horizon — high-prod opp source
     projects multiple launches over the horizon (more than 1).
  2. test_ship_budget_honored — low-ship opp source can't project more
     launches than its accumulated ships allow.
  3. test_target_diversity — opp doesn't project every launch at the
     same target; the `already_targeted` set spreads them out.
  4. test_handles_neutral_only_world — graceful empty return when there
     are no opp planets.
"""

from __future__ import annotations

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.intent import World
from lib.joint_solver.opp_projection import (
    HORIZON,
    OPP_MIN_SHIPS,
    predict_opp_multi_launch,
)


def _planet(pid, owner, *, ships=10, production=2, x=50.0, y=50.0, radius=1.5):
    return Planet(pid, owner, x, y, radius, ships, production)


def _world(my_id, planets, *, step=0):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": [],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": step,
    }
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# Fixture 1: multi-launch count scales with horizon.
# ---------------------------------------------------------------------------


def test_multi_launch_count_scales_with_horizon():
    """High-prod opp source with multiple cheap neutrals around it
    projects several launches over the 15-tick horizon. Specifically:
    more than 1 (vs Phase 5C's 1-shot)."""
    # Opp source with prod=5, surrounded by 5 weakly-defended neutrals.
    opp = [_planet(0, 1, ships=20, production=5, x=10.0, y=10.0)]
    neutrals = [
        _planet(10, -1, ships=2, production=1, x=20.0, y=10.0),
        _planet(11, -1, ships=2, production=1, x=30.0, y=10.0),
        _planet(12, -1, ships=2, production=1, x=10.0, y=20.0),
        _planet(13, -1, ships=2, production=1, x=10.0, y=30.0),
        _planet(14, -1, ships=3, production=1, x=20.0, y=20.0),
    ]
    me = [_planet(20, 0, ships=10, production=1, x=90.0, y=90.0)]
    world = _world(my_id=0, planets=opp + neutrals + me)

    arrivals = predict_opp_multi_launch(world, my_id=0, num_seats=2)
    # The opp source has 5 nearby neutrals to grab and grows by 5 ships/tick.
    # Expect at least 2-3 projected launches over 15 ticks.
    assert len(arrivals) >= 2, \
        f"expected ≥2 multi-launch arrivals, got {len(arrivals)}: {arrivals}"
    # All targets opp launches at should be non-self, non-mine.
    for tgt_pid, eta, owner, ships in arrivals:
        assert int(owner) == 1
        assert int(ships) >= OPP_MIN_SHIPS
        assert int(tgt_pid) != 0  # not opp's own source


# ---------------------------------------------------------------------------
# Fixture 2: ship budget honored.
# ---------------------------------------------------------------------------


def test_ship_budget_honored():
    """Opp source with very few ships and low production can only project
    a small number of launches. Cumulative ships projected ≤ initial + prod·horizon."""
    opp = [_planet(0, 1, ships=10, production=1, x=10.0, y=10.0)]
    neutrals = [_planet(10, -1, ships=2, production=1, x=20.0, y=10.0)]
    me = [_planet(20, 0, ships=10, production=1, x=90.0, y=90.0)]
    world = _world(my_id=0, planets=opp + neutrals + me)

    arrivals = predict_opp_multi_launch(world, my_id=0, num_seats=2)
    # Total ships from source 0 across all projected launches must not exceed
    # initial + prod * HORIZON = 10 + 1*15 = 25.
    total_ships = sum(int(s) for _t, _e, _o, s in arrivals)
    assert total_ships <= 10 + 1 * HORIZON, \
        f"projected ships {total_ships} exceeds budget {10 + HORIZON}"


# ---------------------------------------------------------------------------
# Fixture 3: target diversity (already_targeted set works).
# ---------------------------------------------------------------------------


def test_target_diversity_no_repeat_targets():
    """When opp projects multiple launches, each goes to a DIFFERENT target
    (the already_targeted set prevents repeats from the same source)."""
    opp = [_planet(0, 1, ships=100, production=5, x=10.0, y=10.0)]
    neutrals = [
        _planet(10, -1, ships=2, production=1, x=20.0, y=10.0),
        _planet(11, -1, ships=2, production=1, x=30.0, y=10.0),
        _planet(12, -1, ships=2, production=1, x=10.0, y=20.0),
    ]
    me = [_planet(20, 0, ships=10, production=1, x=90.0, y=90.0)]
    world = _world(my_id=0, planets=opp + neutrals + me)

    arrivals = predict_opp_multi_launch(world, my_id=0, num_seats=2)
    # Each (source, target) pair should appear at most once because the
    # already_targeted set is per-source-projection.
    targets_seen = [int(t) for t, _e, _o, _s in arrivals]
    assert len(targets_seen) == len(set(targets_seen)), \
        f"targets repeated: {targets_seen}"


# ---------------------------------------------------------------------------
# Fixture 4: no opp planets → empty projection.
# ---------------------------------------------------------------------------


def test_no_opp_planets_returns_empty():
    """If the world has no opp planets (we eliminated them), projection
    returns an empty list — no exception, no spurious arrivals."""
    me = [_planet(0, 0, ships=20, production=2)]
    neutrals = [_planet(10, -1, ships=2, production=1, x=20.0, y=10.0)]
    world = _world(my_id=0, planets=me + neutrals)

    arrivals = predict_opp_multi_launch(world, my_id=0, num_seats=2)
    assert arrivals == []
