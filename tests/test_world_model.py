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


# ---------------------------------------------------------------------------
# predict_garrison_at — parity with simulate_planet_timeline at one tick
# ---------------------------------------------------------------------------


def test_predict_garrison_at_parity_with_timeline_neutral_no_arrivals():
    """Neutral planet, no arrivals — predict_garrison_at must match
    simulate_planet_timeline at every eta."""
    from lib.world_model import predict_garrison_at
    p = _planet(0, -1, 50.0, 50.0, ships=10, production=2)
    tl = simulate_planet_timeline(p, [], horizon=15)
    for eta in [0, 1, 5, 10, 15]:
        owner_tl, ships_tl = state_at_timeline(tl, eta)
        owner_p, ships_p = predict_garrison_at(p, eta, [])
        assert owner_p == owner_tl, f"eta={eta}: owner {owner_p} vs {owner_tl}"
        assert abs(ships_p - ships_tl) < 1e-9, (
            f"eta={eta}: ships {ships_p} vs {ships_tl}"
        )


def test_predict_garrison_at_parity_owned_with_production():
    """Owned planet accrues production each tick — must match timeline."""
    from lib.world_model import predict_garrison_at
    p = _planet(0, 0, 50.0, 50.0, ships=5, production=3)
    tl = simulate_planet_timeline(p, [], horizon=10)
    for eta in [0, 1, 3, 10]:
        owner_tl, ships_tl = state_at_timeline(tl, eta)
        owner_p, ships_p = predict_garrison_at(p, eta, [])
        assert (owner_p, ships_p) == (owner_tl, ships_tl), (
            f"eta={eta}: ({owner_p},{ships_p}) vs ({owner_tl},{ships_tl})"
        )


def test_predict_garrison_at_parity_with_combat():
    """Enemy arrival flips ownership — predict + timeline must agree."""
    from lib.world_model import predict_garrison_at
    p = _planet(0, 0, 50.0, 50.0, ships=10, production=1)
    arrivals = [(3, 1, 20)]  # enemy arrives at t=3 with 20 ships
    tl = simulate_planet_timeline(p, arrivals, horizon=10)
    for eta in [0, 1, 2, 3, 4, 5, 10]:
        owner_tl, ships_tl = state_at_timeline(tl, eta)
        owner_p, ships_p = predict_garrison_at(p, eta, arrivals)
        assert (owner_p, ships_p) == (owner_tl, ships_tl), (
            f"eta={eta}: ({owner_p},{ships_p}) vs ({owner_tl},{ships_tl})"
        )


def test_predict_garrison_at_skips_arrivals_past_eta():
    """An arrival scheduled AFTER eta must not influence the prediction."""
    from lib.world_model import predict_garrison_at
    p = _planet(0, -1, 50.0, 50.0, ships=10)
    # Enemy arrival at t=20; we predict at eta=5 — should be neutral+10.
    owner, ships = predict_garrison_at(p, 5, [(20, 1, 100)])
    assert owner == -1 and ships == 10.0


def test_predict_garrison_at_eta_zero_returns_current_state():
    """eta=0 → exactly current owner + garrison, regardless of arrivals."""
    from lib.world_model import predict_garrison_at
    p = _planet(0, 0, 50.0, 50.0, ships=15, production=5)
    owner, ships = predict_garrison_at(p, 0, [(1, 1, 100)])
    assert owner == 0 and ships == 15.0


# ---------------------------------------------------------------------------
# comet_position_at — path-indexed lookup (Part C, 2026-05-19 PM)
# ---------------------------------------------------------------------------


def _world_with_comet(comet_id, path, path_index):
    """Minimal obs that includes a comet at path[path_index]."""
    cur_x, cur_y = path[path_index]
    obs = {
        "player": 0,
        "planets": [
            (0, 0, 5.0, 5.0, 1.0, 100, 1),
            (comet_id, -1, float(cur_x), float(cur_y), 1.0, 30, 1),
        ],
        "fleets": [],
        "angular_velocity": 0.04,
        "comet_planet_ids": [comet_id],
        "comets": [
            {
                "planet_ids": [comet_id],
                "paths": [path],
                "path_index": path_index,
            },
        ],
        "step": 50,
    }
    return World.from_obs(obs)


def test_comet_position_at_returns_path_indexed_point():
    """comet_position_at(comet, world, lead) returns path[path_index+lead]."""
    from lib.world_model import comet_position_at
    # Linear path moving east at 4 units/step.
    path = [[20.0 + i * 4.0, 50.0] for i in range(10)]
    world = _world_with_comet(comet_id=42, path=path, path_index=2)
    # lead=0 → current position = path[2] = (28.0, 50.0)
    pos0 = comet_position_at(42, world, 0)
    assert pos0 == (28.0, 50.0), f"lead=0: got {pos0}"
    # lead=3 → path[5] = (40.0, 50.0)
    pos3 = comet_position_at(42, world, 3)
    assert pos3 == (40.0, 50.0), f"lead=3: got {pos3}"


def test_comet_position_at_returns_none_past_path_end():
    """When path_index + lead >= len(path), comet has exited → None."""
    from lib.world_model import comet_position_at
    path = [[20.0 + i * 4.0, 50.0] for i in range(5)]
    world = _world_with_comet(comet_id=42, path=path, path_index=2)
    # path_index=2 + lead=3 = 5 == len(path) → exited
    assert comet_position_at(42, world, 3) is None
    # path_index=2 + lead=10 → way past
    assert comet_position_at(42, world, 10) is None


def test_comet_position_at_returns_none_for_non_comet():
    """Non-comet planet_id returns None (no path data)."""
    from lib.world_model import comet_position_at
    path = [[20.0 + i * 4.0, 50.0] for i in range(5)]
    world = _world_with_comet(comet_id=42, path=path, path_index=0)
    # Planet 0 (the source) is not a comet.
    assert comet_position_at(0, world, 1) is None
    # Some random id not in the obs.
    assert comet_position_at(999, world, 0) is None


# ---------------------------------------------------------------------------
# planet_position_at — comet-aware dispatcher (2026-05-23 KT-parity fix)
# ---------------------------------------------------------------------------


def test_planet_position_at_comet_uses_path_not_rotation():
    """Comet target: planet_position_at routes through comet_position_at,
    NOT through predict_relative's orbital rotation.

    Regression pin for the KT bit-parity bug (40% turn divergence)
    where reinforce paths called predict_relative(comet, omega, eta)
    and got rotated orbital math instead of the comet's discrete path.
    """
    from lib.world_model import planet_position_at
    from lib.orbit import predict_relative
    path = [[20.0 + i * 4.0, 50.0] for i in range(10)]
    world = _world_with_comet(comet_id=42, path=path, path_index=2)
    # Tuple form (id, owner, x, y, radius, ships, prod).
    comet_tup = (42, -1, 28.0, 50.0, 1.0, 30, 1)
    # lead=3 → path[5] = (40.0, 50.0). NOT the rotated value.
    pos = planet_position_at(comet_tup, world, 3)
    assert pos == (40.0, 50.0), f"comet at lead=3: got {pos}"
    # Confirm raw predict_relative would have rotated (the bug).
    rotated = predict_relative(comet_tup, world.omega, 3)
    assert rotated != (40.0, 50.0), (
        "Test invalid: raw predict_relative happens to match path here; "
        "pick a path point that differs from the rotated value."
    )


def test_planet_position_at_expired_comet_returns_off_board():
    """Comet whose path has expired returns the OFF_BOARD sentinel,
    matching lib.kinematic_table.lookup_relative semantics."""
    from lib.world_model import planet_position_at
    path = [[20.0 + i * 4.0, 50.0] for i in range(5)]
    world = _world_with_comet(comet_id=42, path=path, path_index=2)
    comet_tup = (42, -1, 28.0, 50.0, 1.0, 30, 1)
    # path_index=2 + lead=10 → past end of path → expired.
    pos = planet_position_at(comet_tup, world, 10)
    assert pos == (-1e6, -1e6), f"expired comet: got {pos}"


def test_planet_position_at_orbital_falls_through_to_predict_relative():
    """Non-comet orbital planet: behavior matches predict_relative."""
    from lib.world_model import planet_position_at
    from lib.orbit import predict_relative
    path = [[20.0 + i * 4.0, 50.0] for i in range(5)]
    world = _world_with_comet(comet_id=42, path=path, path_index=0)
    # Planet 0 is the (orbital) source at (5,5) — not a comet.
    p0_tup = (0, 0, 5.0, 5.0, 1.0, 100, 1)
    for lead in (0, 5, 50):
        got = planet_position_at(p0_tup, world, lead)
        exp = predict_relative(p0_tup, world.omega, lead)
        assert got == exp, f"orbital lead={lead}: got {got} vs {exp}"


def test_planet_position_at_accepts_planet_namedtuple():
    """Accepts both tuple-shape and `.id`-attribute Planet objects."""
    from lib.world_model import planet_position_at
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
    path = [[20.0 + i * 4.0, 50.0] for i in range(10)]
    world = _world_with_comet(comet_id=42, path=path, path_index=2)
    comet_planet = Planet(42, -1, 28.0, 50.0, 1.0, 30, 1)
    pos = planet_position_at(comet_planet, world, 3)
    assert pos == (40.0, 50.0), f"Planet form: got {pos}"
