"""Unit tests for agents/baseline/migration_solver — Slice 9.

Plan reference: /root/.claude/plans/take-the-lens-of-magical-shore.md §15.

Covers `_best_capture_ev_for_planet`, `compute_capture_ev_per_planet`,
`_threat_reserve`, and `propose_migrations`.
"""

from __future__ import annotations

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from agents.baseline.migration_solver import (
    MIN_MIGRATION_SHIPS,
    _best_capture_ev_for_planet,
    _threat_reserve,
    compute_capture_ev_per_planet,
    propose_migrations,
)
from lib.intent import World
from lib.world_model import WorldModel


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    return Planet(pid, owner, x, y, radius, ships, production)


def _fleet(fid, owner, x, y, angle, ships, from_planet_id=0):
    return Fleet(fid, owner, x, y, angle, from_planet_id, ships)


def _world(my_id, planets, *, fleets=None, step=0, omega=0.0):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": [
            (f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships)
            for f in (fleets or [])
        ],
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "comets": [],
        "step": step,
    }
    return obs, World.from_obs(obs)


# ---------------------------------------------------------------------------
# _best_capture_ev_for_planet
# ---------------------------------------------------------------------------


def test_capture_ev_zero_when_no_targets():
    """No non-mine planets → EV = 0."""
    P = _planet(0, 0, 10.0, 50.0, ships=100)
    obs, world = _world(0, [P])
    model = WorldModel.from_world(world)
    ev = _best_capture_ev_for_planet(P, 100, world, model, me=0, step=0)
    assert ev == 0.0


def test_capture_ev_zero_when_ship_count_too_low():
    """Ship count below MIN_MIGRATION_SHIPS → EV = 0."""
    P = _planet(0, 0, 10.0, 50.0, ships=2)
    T = _planet(1, -1, 30.0, 50.0, ships=5, production=2)
    obs, world = _world(0, [P, T])
    model = WorldModel.from_world(world)
    ev = _best_capture_ev_for_planet(P, 2, world, model, me=0, step=0)
    assert ev == 0.0


def test_capture_ev_zero_when_infeasible():
    """Ship count too low to overcome target garrison → EV = 0."""
    P = _planet(0, 0, 10.0, 50.0, ships=10, production=1)
    T = _planet(1, -1, 30.0, 50.0, ships=200, production=2)
    obs, world = _world(0, [P, T])
    model = WorldModel.from_world(world)
    ev = _best_capture_ev_for_planet(P, 10, world, model, me=0, step=0)
    assert ev == 0.0


def test_capture_ev_positive_when_feasible():
    """Strong source, weak target → EV > 0."""
    P = _planet(0, 0, 10.0, 50.0, ships=120, production=2)
    T = _planet(1, -1, 30.0, 50.0, ships=5, production=3)
    obs, world = _world(0, [P, T])
    model = WorldModel.from_world(world)
    ev = _best_capture_ev_for_planet(P, 120, world, model, me=0, step=0)
    assert ev > 0.0


def test_capture_ev_picks_highest_production_target():
    """Among feasible targets, EV picks the max production × pv."""
    P = _planet(0, 0, 10.0, 50.0, ships=200, production=2)
    T_low = _planet(1, -1, 25.0, 50.0, ships=5, production=1)
    T_high = _planet(2, -1, 30.0, 50.0, ships=5, production=5)
    obs_both, world_both = _world(0, [P, T_low, T_high])
    model_both = WorldModel.from_world(world_both)
    ev_both = _best_capture_ev_for_planet(
        P, 200, world_both, model_both, me=0, step=0,
    )
    # EV should reflect the high-production target.
    obs_low, world_low = _world(0, [P, T_low])
    model_low = WorldModel.from_world(world_low)
    ev_low_only = _best_capture_ev_for_planet(
        P, 200, world_low, model_low, me=0, step=0,
    )
    assert ev_both > ev_low_only


# ---------------------------------------------------------------------------
# compute_capture_ev_per_planet
# ---------------------------------------------------------------------------


def test_per_planet_ev_dict_shape():
    """Returns one entry per my-planet."""
    p0 = _planet(0, 0, 10.0, 50.0, ships=100)
    p1 = _planet(1, 0, 80.0, 50.0, ships=80)
    enemy = _planet(2, 1, 50.0, 50.0, ships=20)
    obs, world = _world(0, [p0, p1, enemy])
    model = WorldModel.from_world(world)
    ev = compute_capture_ev_per_planet(world, model, me=0)
    assert set(ev.keys()) == {0, 1}


def test_per_planet_ev_zero_for_isolated():
    """A my-planet with no feasible captures → EV = 0."""
    isolated = _planet(0, 0, 10.0, 50.0, ships=2)  # below MIN_MIGRATION_SHIPS
    target = _planet(1, -1, 50.0, 50.0, ships=5, production=2)
    obs, world = _world(0, [isolated, target])
    model = WorldModel.from_world(world)
    ev = compute_capture_ev_per_planet(world, model, me=0)
    assert ev[0] == 0.0


# ---------------------------------------------------------------------------
# _threat_reserve
# ---------------------------------------------------------------------------


def test_threat_reserve_zero_when_no_threat():
    """No inbound enemy fleets → reserve = 0."""
    P = _planet(0, 0, 10.0, 50.0, ships=100)
    obs, world = _world(0, [P])
    model = WorldModel.from_world(world)
    assert _threat_reserve(P, world, model, me=0) == 0


def test_threat_reserve_counts_inbound_enemy():
    """Inbound enemy → reserve matches threat force."""
    P = _planet(0, 0, 50.0, 50.0, ships=100)
    enemy = _planet(1, 1, 80.0, 50.0, ships=200)
    inbound = _fleet(0, 1, 70.0, 50.0, angle=3.141592, ships=30)
    obs, world = _world(0, [P, enemy], fleets=[inbound])
    model = WorldModel.from_world(world)
    reserve = _threat_reserve(P, world, model, me=0)
    assert reserve > 0


# ---------------------------------------------------------------------------
# propose_migrations — end-to-end
# ---------------------------------------------------------------------------


def test_propose_migrations_empty_when_single_planet():
    """Need at least 2 own planets to have a migration."""
    P = _planet(0, 0, 10.0, 50.0, ships=100)
    obs, world = _world(0, [P])
    model = WorldModel.from_world(world)
    assert propose_migrations(world, model, me=0) == []


def test_propose_migrations_emits_when_dst_unlocks_capture():
    """src has many ships but no nearby targets; dst is near a target
    but lacks ships → migration emits.
    """
    # src is in the corner, no nearby targets, with 200 ships.
    src = _planet(0, 0, 5.0, 5.0, ships=200, production=1)
    # dst is near the target but has only 10 ships (not enough).
    dst = _planet(1, 0, 80.0, 50.0, ships=10, production=1)
    # Target near dst, requires ~25 ships to capture, dst alone can't.
    tgt = _planet(2, -1, 85.0, 50.0, ships=20, production=3)
    obs, world = _world(0, [src, dst, tgt])
    model = WorldModel.from_world(world)
    migrations = propose_migrations(world, model, me=0)
    # At least one migration: src → dst (so dst can capture tgt).
    assert len(migrations) >= 1
    # The migration's tuple shape matches proposer's prerank format.
    m = migrations[0]
    assert len(m) == 8
    value, m_src, m_tgt, ships, angle, eta, horizon, wait_N = m
    assert value > 0.0
    assert int(m_src.id) == 0  # from src (the rear one)
    assert int(m_tgt.id) == 1  # to dst (the front-line)
    assert int(wait_N) == 0    # migrations are fire-now
    assert int(ships) >= MIN_MIGRATION_SHIPS


def test_propose_migrations_dedups_per_source():
    """Each source can have at most one migration per turn."""
    src = _planet(0, 0, 5.0, 5.0, ships=200, production=1)
    dst_a = _planet(1, 0, 80.0, 50.0, ships=10, production=1)
    dst_b = _planet(2, 0, 80.0, 90.0, ships=10, production=1)
    tgt_a = _planet(3, -1, 85.0, 50.0, ships=20, production=3)
    tgt_b = _planet(4, -1, 85.0, 90.0, ships=20, production=4)
    obs, world = _world(0, [src, dst_a, dst_b, tgt_a, tgt_b])
    model = WorldModel.from_world(world)
    migrations = propose_migrations(world, model, me=0)
    # At most 1 migration from src.
    src_migrations = [m for m in migrations if int(m[1].id) == 0]
    assert len(src_migrations) <= 1


def test_propose_migrations_skips_threatened_destination():
    """Destinations under inbound enemy threat are NOT migration targets
    (defensive reinforces handle them via W2 / proposer).
    """
    src = _planet(0, 0, 5.0, 5.0, ships=200, production=1)
    dst = _planet(1, 0, 80.0, 50.0, ships=10, production=1)
    enemy_close = _planet(2, 1, 90.0, 50.0, ships=200)
    tgt = _planet(3, -1, 85.0, 50.0, ships=20, production=3)
    # Inbound enemy fleet targeting dst → dst is under threat.
    inbound = _fleet(0, 1, 85.0, 50.0, angle=3.141592, ships=40)
    obs, world = _world(0, [src, dst, enemy_close, tgt], fleets=[inbound])
    model = WorldModel.from_world(world)
    migrations = propose_migrations(world, model, me=0)
    # No migration with dst as destination (it's a defensive scenario).
    dst_migrations = [m for m in migrations if int(m[2].id) == 1]
    assert dst_migrations == []


def test_propose_migrations_empty_when_no_value_unlock():
    """If both planets have similar EV, migration has no value → empty."""
    a = _planet(0, 0, 40.0, 50.0, ships=100, production=2)
    b = _planet(1, 0, 60.0, 50.0, ships=100, production=2)
    # No targets — both planets have EV=0, no migration helps.
    obs, world = _world(0, [a, b])
    model = WorldModel.from_world(world)
    migrations = propose_migrations(world, model, me=0)
    assert migrations == []
