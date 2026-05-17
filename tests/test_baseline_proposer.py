"""Unit tests for agents/baseline/proposer."""

from __future__ import annotations

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from agents.baseline.proposer import (
    MIN_FLEET_SIZE,
    WAIT_EXTRA_SURPLUS,
    capture_size,
    enumerate_ship_counts,
    nearest_k,
    propose,
    wait_band,
    wait_then_fire_variants,
)
from lib.intent import World
from lib.world_model import WorldModel


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    """Planet is a NamedTuple: (id, owner, x, y, radius, ships, production)."""
    return Planet(pid, owner, x, y, radius, ships, production)


def _world(my_id, planets, *, step=0, omega=0.0):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": [],
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "step": step,
    }
    return World.from_obs(obs)


def test_wait_band_buckets():
    assert wait_band(0) == 0
    assert wait_band(1) == 1
    assert wait_band(7) == 1
    assert wait_band(8) == 2
    assert wait_band(100) == 2


def test_nearest_k_sorts_by_distance():
    src = _planet(0, 0, 10.0, 50.0)
    targets = [
        _planet(1, 1, 90.0, 50.0),  # far
        _planet(2, 1, 20.0, 50.0),  # near
        _planet(3, 1, 50.0, 50.0),  # mid
    ]
    out = nearest_k(targets, src, 2)
    assert [t.id for t in out] == [2, 3]


def test_capture_size_neutral_is_at_least_min_plus_one():
    """For a non-mine target, capture size >= max(MIN_FLEET_SIZE, garrison+1)."""
    src = _planet(0, 0, 10.0, 50.0, ships=100, production=2)
    tgt = _planet(1, -1, 12.0, 50.0, ships=5, production=1)  # neutral with 5 ships
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    sz = capture_size(src, tgt, model, omega=0.0, me=0, world=world)
    assert sz >= 6  # at minimum need garrison+1 = 6


def test_capture_size_own_no_threat_returns_zero():
    """Reinforce for my own planet with no enemy threat → 0 (skip)."""
    src = _planet(0, 0, 10.0, 50.0, ships=20)
    mine_tgt = _planet(1, 0, 12.0, 50.0, ships=5)  # also mine, no threat
    world = _world(0, [src, mine_tgt])
    model = WorldModel.from_world(world)
    sz = capture_size(src, mine_tgt, model, omega=0.0, me=0, world=world)
    assert sz == 0


def test_enumerate_ship_counts_returns_sizes_at_or_under_budget():
    src = _planet(0, 0, 10.0, 50.0, ships=50, production=1)
    tgt = _planet(1, -1, 12.0, 50.0, ships=3, production=1)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    sizes = enumerate_ship_counts(src, tgt, model, omega=0.0, me=0, world=world)
    assert sizes  # non-empty
    assert all(MIN_FLEET_SIZE <= s <= 50 for s in sizes)
    assert sizes == sorted(sizes)  # sorted ascending


def test_wait_then_fire_variants_skips_mine_targets():
    src = _planet(0, 0, 10.0, 50.0, ships=10, production=1)
    mine = _planet(1, 0, 12.0, 50.0, ships=5)
    world = _world(0, [src, mine])
    model = WorldModel.from_world(world)
    assert wait_then_fire_variants(src, mine, model, omega=0.0, me=0) == []


def test_wait_then_fire_variants_skips_zero_production():
    src = _planet(0, 0, 10.0, 50.0, ships=10, production=0)
    tgt = _planet(1, -1, 12.0, 50.0, ships=20)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    # src can't accumulate so no wait variants
    assert wait_then_fire_variants(src, tgt, model, omega=0.0, me=0) == []


def test_wait_then_fire_variants_emits_multi_grid_with_dedup():
    """A feasible-now pair still gets wait>=1 variants for surplus targets,
    deduped by (wait_N, ships)."""
    src = _planet(0, 0, 10.0, 50.0, ships=50, production=2)
    tgt = _planet(1, -1, 12.0, 50.0, ships=3, production=1)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    variants = wait_then_fire_variants(src, tgt, model, omega=0.0, me=0)
    assert variants, "expected at least one wait variant"
    # wait_N>=1 for every variant
    assert all(w >= 1 for _ships, w, _angle, _eta in variants)
    # Up to len(WAIT_EXTRA_SURPLUS) distinct (wait_N, ships) variants
    assert len(variants) <= len(WAIT_EXTRA_SURPLUS)
    # All distinct (wait_N, ships) keys
    keys = {(w, s) for s, w, _angle, _eta in variants}
    assert len(keys) == len(variants)


def test_propose_dedups_per_src_tgt_band():
    """propose() returns at most one entry per (src, tgt, wait_band).
    Bands are {0, 1..7, >=8}, so at most 3 entries per (src, tgt) pair.
    """
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=3)
    tgt = _planet(1, -1, 12.0, 50.0, ships=5, production=2)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    out = propose(
        my_planets=[src], target_pool=[tgt],
        world=world, model=model, me=0, omega=0.0, baseline_len=50,
    )
    # All entries are for (src, tgt)
    assert all(int(e[1].id) == 0 and int(e[2].id) == 1 for e in out)
    # Unique (band) per (src, tgt)
    bands = [wait_band(int(e[7])) for e in out]
    assert len(bands) == len(set(bands))


def test_propose_sorts_descending_by_cheap_delta():
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=3)
    tgt = _planet(1, -1, 12.0, 50.0, ships=5, production=2)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    out = propose(
        my_planets=[src], target_pool=[tgt],
        world=world, model=model, me=0, omega=0.0, baseline_len=50,
    )
    cheap_vals = [e[0] for e in out]
    assert cheap_vals == sorted(cheap_vals, reverse=True)


def test_propose_returns_empty_when_no_targets():
    src = _planet(0, 0, 10.0, 50.0, ships=20)
    world = _world(0, [src])
    model = WorldModel.from_world(world)
    out = propose(
        my_planets=[src], target_pool=[],
        world=world, model=model, me=0, omega=0.0, baseline_len=50,
    )
    assert out == []
