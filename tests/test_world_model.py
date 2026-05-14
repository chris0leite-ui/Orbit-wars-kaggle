"""Tests for lib/world_model — arrival ledger + per-planet timeline.

Coverage:
- `build_arrival_ledger`: attribute in-flight fleets to their target
  planet (ray-cast); fleets not hitting any planet within horizon are
  dropped.
- `simulate_planet_timeline`: production accrual + same-step combat
  resolution + ownership tracking over `horizon` steps.
- `WorldModel.from_world` + lookups: predicted owner / ships at a
  future step.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from lib.intent import World
from lib.world_model import (
    WorldModel,
    build_arrival_ledger,
    fleet_target_planet,
    simulate_planet_timeline,
    state_at_timeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    return SimpleNamespace(
        id=pid, owner=owner, x=x, y=y, radius=radius, ships=ships, production=production,
    )


def _fleet(fid, owner, x, y, angle, from_id=0, ships=10):
    return SimpleNamespace(
        id=fid, owner=owner, x=x, y=y, angle=angle, from_planet_id=from_id, ships=ships,
    )


# ---------------------------------------------------------------------------
# fleet_target_planet — ray-cast attribution
# ---------------------------------------------------------------------------


def test_fleet_aimed_at_planet_attributes_correctly():
    """Fleet heading east at (10, 50) → planet at (90, 50) → hits."""
    target = _planet(1, -1, 90.0, 50.0, radius=2.0)
    other = _planet(2, 0, 50.0, 5.0)
    fleet = _fleet(100, 0, 10.0, 50.0, angle=0.0, ships=100)
    p, eta = fleet_target_planet(fleet, [target, other])
    assert p is target
    assert eta is not None and eta > 0


def test_fleet_aimed_away_from_all_returns_none():
    p1 = _planet(1, -1, 10.0, 10.0)
    fleet = _fleet(100, 0, 50.0, 50.0, angle=math.pi, ships=10)  # heading west
    p, eta = fleet_target_planet(fleet, [p1])
    assert p is None and eta is None


def test_fleet_picks_nearest_in_path():
    """Two planets along the fleet's path; nearer one is the target."""
    near = _planet(1, -1, 60.0, 50.0, radius=2.0)
    far = _planet(2, -1, 90.0, 50.0, radius=2.0)
    fleet = _fleet(100, 0, 10.0, 50.0, angle=0.0, ships=100)
    p, _ = fleet_target_planet(fleet, [far, near])
    assert p is near


# ---------------------------------------------------------------------------
# build_arrival_ledger
# ---------------------------------------------------------------------------


def test_arrival_ledger_groups_by_target_planet():
    p1 = _planet(1, -1, 90.0, 50.0)
    p2 = _planet(2, -1, 50.0, 90.0)
    fleets = [
        _fleet(100, 0, 10.0, 50.0, angle=0.0, ships=20),     # → p1
        _fleet(101, 1, 50.0, 10.0, angle=math.pi/2, ships=15),  # → p2
        _fleet(102, 0, 10.0, 50.0, angle=0.0, ships=10),     # → p1 again
    ]
    ledger = build_arrival_ledger(fleets, [p1, p2])
    assert len(ledger[1]) == 2
    assert len(ledger[2]) == 1
    # Each entry is (eta, owner, ships).
    assert {l[1] for l in ledger[1]} == {0}
    assert {l[1] for l in ledger[2]} == {1}


# ---------------------------------------------------------------------------
# simulate_planet_timeline
# ---------------------------------------------------------------------------


def test_neutral_planet_no_arrivals_stays_neutral():
    p = _planet(1, -1, 50.0, 50.0, ships=0, production=0)
    tl = simulate_planet_timeline(p, [], horizon=10)
    assert tl["owner_at"][10] == -1
    assert tl["ships_at"][10] == 0.0


def test_owned_planet_accrues_production():
    p = _planet(1, 0, 50.0, 50.0, ships=10, production=2)
    tl = simulate_planet_timeline(p, [], horizon=5)
    # Step 0 = 10. Step 5 = 10 + 5*2 = 20.
    assert tl["ships_at"][0] == 10.0
    assert tl["ships_at"][5] == 20.0
    assert tl["owner_at"][5] == 0


def test_neutral_planet_does_not_accrue():
    """Neutrals don't produce per env spec."""
    p = _planet(1, -1, 50.0, 50.0, ships=10, production=3)
    tl = simulate_planet_timeline(p, [], horizon=5)
    assert tl["ships_at"][5] == 10.0  # unchanged


def test_enemy_arrival_flips_ownership():
    """At eta=2, P1 sends 100 ships against our 5+production*2=9. Flip."""
    p = _planet(1, 0, 50.0, 50.0, ships=5, production=2)
    arrivals = [(2, 1, 100)]
    tl = simulate_planet_timeline(p, arrivals, horizon=5)
    assert tl["owner_at"][2] == 1
    # 100 - 9 = 91 survivors after combat.
    assert tl["ships_at"][2] == 91.0
    # After step 2, P1 starts producing too — step 3-5 = 91 + 3*2 = 97.
    assert tl["ships_at"][5] == 97.0


def test_state_at_timeline_clamps_to_horizon():
    p = _planet(1, 0, 50.0, 50.0, ships=10, production=2)
    tl = simulate_planet_timeline(p, [], horizon=5)
    owner, ships = state_at_timeline(tl, arrival_turn=999)
    assert owner == 0
    assert ships == 20.0


# ---------------------------------------------------------------------------
# WorldModel snapshot
# ---------------------------------------------------------------------------


def test_world_model_lookups_match_timelines():
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 2.0, 100, 1],   # mine
            [1, -1, 90.0, 50.0, 2.0,  20, 2],  # neutral target
        ],
        "fleets": [
            [200, 0, 10.0, 50.0, 0.0, 0, 50],  # our fleet heading east at p1
        ],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": 1,
    }
    world = World.from_obs(obs)
    wm = WorldModel.from_world(world)
    # The fleet is in flight; at some future step it arrives and flips p1.
    assert wm.owner_at(1, 0) == -1  # currently neutral
    # Pick a step well past the fleet's arrival.
    owner_late = wm.owner_at(1, 80)
    # The flight is ~80 units at speed v(50) ≈ 3.0 → eta ≈ 27 steps. By
    # step 80 the planet should be ours.
    assert owner_late == 0


# ---------------------------------------------------------------------------
# time_to_enemy_threat — HAV helper (2026-05-14 plan)
# ---------------------------------------------------------------------------


def _world_for_threat(planets, fleets=(), my_id=0):
    obs = {
        "player": my_id,
        "planets": [
            [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
            for p in planets
        ],
        "fleets": list(fleets),
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": 0,
    }
    return World.from_obs(obs)


def test_time_to_enemy_threat_returns_none_when_no_enemy_exists():
    # Only us + a neutral. No threat to the neutral.
    mine = _planet(0, 0, 10.0, 10.0, ships=100)
    neutral = _planet(1, -1, 90.0, 90.0, ships=5)
    world = _world_for_threat([mine, neutral])
    wm = WorldModel.from_world(world)
    assert wm.time_to_enemy_threat(neutral.id, my_id=0, world=world) is None


def test_time_to_enemy_threat_uses_inflight_enemy_fleet():
    # Enemy fleet inbound at target's planet id.
    mine = _planet(0, 0, 10.0, 10.0, ships=100)
    target = _planet(1, -1, 90.0, 50.0, ships=20, radius=2.0)
    # Enemy fleet at (10, 50) flying east toward target.
    obs = {
        "player": 0,
        "planets": [
            [mine.id, mine.owner, mine.x, mine.y, mine.radius, mine.ships, mine.production],
            [target.id, target.owner, target.x, target.y, target.radius, target.ships, target.production],
        ],
        "fleets": [[200, 1, 10.0, 50.0, 0.0, 99, 50]],  # enemy id=200, ships=50
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": 0,
    }
    world = World.from_obs(obs)
    wm = WorldModel.from_world(world)
    threat = wm.time_to_enemy_threat(target.id, my_id=0, world=world)
    assert threat is not None
    assert threat > 0


def test_time_to_enemy_threat_uses_potential_enemy_launch():
    # No in-flight fleets but a nearby enemy planet could launch.
    mine = _planet(0, 0, 5.0, 50.0, ships=100)
    target = _planet(1, -1, 50.0, 50.0, ships=10)
    enemy = _planet(2, 1, 95.0, 50.0, ships=80, radius=2.0)  # to the east of target
    world = _world_for_threat([mine, target, enemy])
    wm = WorldModel.from_world(world)
    threat = wm.time_to_enemy_threat(target.id, my_id=0, world=world)
    # The enemy is 45 units away from the target; with 80 ships its
    # fleet_speed is well above 1, so ETA must be > 0 and finite.
    assert threat is not None
    assert threat > 0


def test_time_to_enemy_threat_picks_minimum_across_sources():
    # Two enemy planets — one close, one far. We want the closer one.
    mine = _planet(0, 0, 5.0, 50.0, ships=100)
    target = _planet(1, -1, 50.0, 50.0, ships=10)
    near = _planet(2, 1, 60.0, 50.0, ships=20)
    far = _planet(3, 1, 95.0, 95.0, ships=20)
    world = _world_for_threat([mine, target, near, far])
    wm = WorldModel.from_world(world)
    threat = wm.time_to_enemy_threat(target.id, my_id=0, world=world)
    # Sanity: threat ETA should be small (near planet is 10 units off).
    assert threat is not None
    assert threat < 30


def test_time_to_enemy_threat_skips_neutral_and_self_planets():
    # Only neutrals + our own planets — no enemy threat.
    mine1 = _planet(0, 0, 10.0, 10.0, ships=100)
    mine2 = _planet(1, 0, 20.0, 20.0, ships=50)
    target = _planet(2, -1, 50.0, 50.0, ships=10)
    neutral2 = _planet(3, -1, 90.0, 90.0, ships=5)
    world = _world_for_threat([mine1, mine2, target, neutral2])
    wm = WorldModel.from_world(world)
    assert wm.time_to_enemy_threat(target.id, my_id=0, world=world) is None
