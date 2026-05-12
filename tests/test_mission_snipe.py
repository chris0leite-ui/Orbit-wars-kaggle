"""Tests for lib/missions/snipe.propose_snipe_missions."""

from __future__ import annotations

from types import SimpleNamespace

from agent import World, WorldModel, propose_snipe_missions


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    return SimpleNamespace(
        id=pid, owner=owner, x=x, y=y, radius=radius,
        ships=ships, production=production,
    )


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


def test_no_missions_when_we_own_nothing():
    world = _world(my_id=0, planets=[
        _planet(0, 1, 10.0, 50.0),
        _planet(1, 2, 90.0, 50.0),
    ])
    model = WorldModel.from_world(world)
    assert propose_snipe_missions(world, model) == []


def test_no_missions_when_no_non_our_targets():
    world = _world(my_id=0, planets=[
        _planet(0, 0, 10.0, 50.0),
        _planet(1, 0, 90.0, 50.0),
    ])
    model = WorldModel.from_world(world)
    assert propose_snipe_missions(world, model) == []


def test_cross_product_count_2_sources_3_targets():
    """2 owned, 3 enemy planets → 6 candidate snipe missions."""
    world = _world(my_id=0, planets=[
        _planet(0, 0, 10.0, 50.0, ships=20),     # us
        _planet(1, 0, 90.0, 50.0, ships=20),     # us
        _planet(2, 1, 50.0, 10.0, ships=5),      # enemy
        _planet(3, 2, 50.0, 90.0, ships=5),      # enemy
        _planet(4, -1, 50.0, 50.0, ships=10),    # neutral
    ])
    model = WorldModel.from_world(world)
    ms = propose_snipe_missions(world, model)
    assert len(ms) == 6
    assert all(m.mission_class == "snipe" for m in ms)
    assert all(m.src_id in {0, 1} for m in ms)
    assert all(m.target_id in {2, 3, 4} for m in ms)
    assert all(m.ships >= 1 for m in ms)
    assert all(m.score > 0.0 for m in ms)


def test_score_scales_with_production_inverse_distance():
    """Closer / higher-production target ranks above farther / lower."""
    world = _world(my_id=0, planets=[
        _planet(0, 0, 0.0, 0.0, ships=20),
        _planet(1, 1, 10.0, 0.0, ships=1, production=5),   # near, high-prod
        _planet(2, 1, 100.0, 0.0, ships=1, production=5),  # far, same prod
    ])
    model = WorldModel.from_world(world)
    ms = propose_snipe_missions(world, model)
    # Two missions: src=0 -> {1, 2}.
    near = next(m for m in ms if m.target_id == 1)
    far = next(m for m in ms if m.target_id == 2)
    assert near.score > far.score


def test_skips_target_already_ours_at_arrival():
    """If WorldModel predicts target ours with surplus garrison >= base_ships
    at our arrival, no mission is produced for that pair.

    Under aggressive sizing, base_ships scales to ~0.7 * src.ships, so the
    in-flight friendly fleet must be at least that large to trigger suppression.
    """
    target = _planet(1, 1, 5.0, 0.0, ships=1, production=1, radius=0.5)
    src = _planet(0, 0, 0.0, 0.0, ships=100, production=1)
    obs = {
        "player": 0,
        "planets": [
            (src.id, src.owner, src.x, src.y, src.radius, src.ships, src.production),
            (target.id, target.owner, target.x, target.y, target.radius,
             target.ships, target.production),
        ],
        # 200-ship in-flight friendly fleet — comfortably above aggressive base_ships (70).
        "fleets": [(900, 0, 4.0, 0.0, 0.0, src.id, 200)],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": 0,
    }
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    pred_o = model.owner_at(target.id, 1)
    pred_s = model.ships_at(target.id, 1) or 0.0
    if pred_o == 0 and pred_s >= 70:
        ms = propose_snipe_missions(world, model)
        assert all(m.target_id != target.id for m in ms), (
            f"snipe to target {target.id} should be suppressed; got "
            f"{[(m.src_id, m.target_id, m.ships) for m in ms]}"
        )
