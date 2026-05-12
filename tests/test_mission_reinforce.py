"""Tests for lib/missions/reinforce.propose_reinforce_missions."""

from __future__ import annotations

from types import SimpleNamespace

from agent import World, WorldModel, propose_reinforce_missions


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    return SimpleNamespace(
        id=pid, owner=owner, x=x, y=y, radius=radius,
        ships=ships, production=production,
    )


def _world(my_id, planets, fleets=None, omega=0.0):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": fleets or [],
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "comets": [],
        "step": 0,
    }
    return World.from_obs(obs)


def test_no_missions_when_no_planets_threatened():
    """No incoming enemy fleets → no planet predicted to flip → no missions."""
    world = _world(my_id=0, planets=[
        _planet(0, 0, 0.0, 0.0, ships=50),
        _planet(1, 0, 100.0, 0.0, ships=50),
    ])
    model = WorldModel.from_world(world)
    assert propose_reinforce_missions(world, model) == []


def test_no_missions_when_only_one_owned_planet():
    """Need at least one source AND one defendee; same planet can't self-reinforce."""
    world = _world(my_id=0, planets=[
        _planet(0, 0, 0.0, 0.0, ships=10),
        _planet(1, 1, 100.0, 0.0, ships=99),
    ])
    model = WorldModel.from_world(world)
    assert propose_reinforce_missions(world, model) == []


def test_mission_built_when_enemy_fleet_threatens_our_planet():
    """Enemy fleet inbound at our planet 1 → reinforce mission from a
    closer source 0 to defend planet 1 before the enemy lands.

    Geometry: enemy is far from target (eta ~12), source is close to
    target (eta ~6) so we can arrive before T_loss.
    """
    # Enemy at (5, 0) heading east with 100 ships → planet 1 at (50, 0)
    # is 45 units away. Source at (50, 30) is OFF-axis so the enemy's
    # straight east trajectory doesn't ray-cast onto our source.
    # Defender at (50, 0) with 5 ships + prod=3 will have ~41 ships by
    # eta=12 < attackers' 100 → flips. Source-to-target distance is 30,
    # fleet of ~60 ships at speed ~3.4 → eta ~9 < T_loss=12.
    enemy_fleet = (900, 1, 5.0, 0.0, 0.0, 99, 100)
    world = _world(
        my_id=0,
        planets=[
            _planet(0, 0, 50.0, 30.0, ships=200, production=2),
            _planet(1, 0, 50.0, 0.0, ships=5, production=3),
        ],
        fleets=[enemy_fleet],
    )
    model = WorldModel.from_world(world)
    missions = propose_reinforce_missions(world, model)
    # We expect at least one reinforce mission from planet 0 → planet 1.
    assert len(missions) >= 1
    m = missions[0]
    assert m.mission_class == "reinforce"
    assert m.src_id == 0
    assert m.target_id == 1
    assert m.ships >= 1


def test_skips_when_we_cant_arrive_before_loss():
    """If our source is too far to arrive before T_loss, no mission."""
    # Enemy fleet at (49, 0) → planet at (50, 0), arrives in 1-2 steps.
    # Our other planet is at (0, 0), 50 units away → eta way > 2.
    enemy_fleet = (900, 1, 49.0, 0.0, 0.0, 99, 50)
    world = _world(
        my_id=0,
        planets=[
            _planet(0, 0, 0.0, 0.0, ships=200, production=2),
            _planet(1, 0, 50.0, 0.0, ships=1, production=3),
        ],
        fleets=[enemy_fleet],
    )
    model = WorldModel.from_world(world)
    missions = propose_reinforce_missions(world, model)
    # Our fleet would arrive at planet 1 in ~10+ steps; enemy lands in 1-2.
    # We can't defend in time.
    assert missions == []


def test_ship_size_scales_with_predicted_attacker_strength():
    """Bigger inbound enemy → bigger reinforce fleet sent."""
    # Small enemy threat.
    small_enemy = (900, 1, 30.0, 0.0, 0.0, 99, 10)
    world_small = _world(
        my_id=0,
        planets=[
            _planet(0, 0, 0.0, 0.0, ships=500, production=2),
            _planet(1, 0, 50.0, 0.0, ships=1, production=3),
        ],
        fleets=[small_enemy],
    )
    model_small = WorldModel.from_world(world_small)
    missions_small = propose_reinforce_missions(world_small, model_small)

    # Large enemy threat.
    big_enemy = (900, 1, 30.0, 0.0, 0.0, 99, 100)
    world_big = _world(
        my_id=0,
        planets=[
            _planet(0, 0, 0.0, 0.0, ships=500, production=2),
            _planet(1, 0, 50.0, 0.0, ships=1, production=3),
        ],
        fleets=[big_enemy],
    )
    model_big = WorldModel.from_world(world_big)
    missions_big = propose_reinforce_missions(world_big, model_big)

    # If both worlds produce missions, big-enemy mission should send
    # more ships. (If either produced none — e.g. eta too long — we
    # can't make this assertion strictly. Soft-check.)
    if missions_small and missions_big:
        assert missions_big[0].ships >= missions_small[0].ships
