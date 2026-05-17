"""Phase 3 tests for lib.trajectory_layer — hypothetical launch overlay.

`World.with_candidate(spec)` returns a child World with a synthetic
fleet appended and the source planet's ships decremented. The
invariant under test: for every spec, the overlay's predicted
arrivals match what fast_sim produces when the launch is COMMITTED.

`overlay` is at step S with the synthetic fleet added. `committed`
is at step S+1 (fast_sim has been stepped once). Their arrival ETAs
differ by exactly 1: `overlay.eta == committed.eta + 1`.
"""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

pytestmark = pytest.mark.slow

from kaggle_environments import make

from lib.fast_sim import Snapshot, clone as fs_clone
from lib.fast_sim import from_obs as fs_from_obs
from lib.fast_sim import step as fs_step
from lib.trajectory_layer import (
    Arrival,
    LaunchSpec,
    World,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step_env_to_obs(seed: int, warmup: int, num_seats: int,
                     ) -> tuple[Any, int]:
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=num_seats)
    rng = random.Random(seed * 11 + 3)
    for _ in range(warmup):
        obs0 = env.state[0].observation
        planets = (obs0["planets"] if isinstance(obs0, dict)
                   else obs0.planets)
        actions: list[list] = [[] for _ in range(num_seats)]
        for p in planets:
            owner = p[1]
            if 0 <= owner < num_seats and p[5] > 5 and rng.random() < 0.3:
                actions[owner].append([p[0], rng.uniform(0.0, 6.283),
                                       int(p[5] // 2)])
        env.step(actions)
    return env.state[0].observation, int(env.info.get("seed", seed))


def _our_planets_with_ships(world: World, my_id: int) -> list:
    return [p for p in world.planets
            if p.owner == my_id and p.ships >= 2 and not p.is_comet]


def _pick_aim_angle_at_target(src, target) -> float:
    """Aim from src toward target's current position."""
    dx = target.current_x - src.current_x
    dy = target.current_y - src.current_y
    return math.atan2(dy, dx)


# ---------------------------------------------------------------------------
# Basic mutation semantics
# ---------------------------------------------------------------------------


def test_overlay_decrements_source_ships():
    """The synthetic launch deducts ships from the source planet."""
    obs, ep_seed = _step_env_to_obs(seed=42, warmup=10, num_seats=2)
    world = World.from_obs(obs, episode_seed=ep_seed)
    src = _our_planets_with_ships(world, my_id=0)[0]
    spec = LaunchSpec(src_id=src.id, aim_angle=0.0, ships=3, owner=0)
    overlay = world.with_candidate(spec)
    new_src = overlay.planet_by_id(src.id)
    assert new_src.ships == src.ships - 3
    # Parent unchanged.
    assert world.planet_by_id(src.id).ships == src.ships


def test_overlay_adds_synthetic_fleet():
    """The overlay's fleets tuple has exactly one more entry, at the
    env-faithful spawn position."""
    obs, ep_seed = _step_env_to_obs(seed=42, warmup=10, num_seats=2)
    world = World.from_obs(obs, episode_seed=ep_seed)
    src = _our_planets_with_ships(world, my_id=0)[0]
    n_before = len(world.fleets)

    spec = LaunchSpec(src_id=src.id, aim_angle=1.5, ships=5, owner=0)
    overlay = world.with_candidate(spec)
    assert len(overlay.fleets) == n_before + 1
    assert len(world.fleets) == n_before  # parent unchanged

    # New fleet's spawn = src.center + (radius + 0.1) * direction
    new_fleet = overlay.fleets[-1]
    expected_x = src.current_x + math.cos(1.5) * (src.radius + 0.1)
    expected_y = src.current_y + math.sin(1.5) * (src.radius + 0.1)
    assert math.isclose(new_fleet.current_x, expected_x, abs_tol=1e-12)
    assert math.isclose(new_fleet.current_y, expected_y, abs_tol=1e-12)
    assert new_fleet.angle == 1.5
    assert new_fleet.ships == 5
    assert new_fleet.from_planet_id == src.id
    # Synthetic fleets get negative ids to avoid collision with real fleets.
    assert new_fleet.id < 0


def test_overlay_parent_caches_intact():
    """Materialising the parent's ledger BEFORE the overlay leaves
    the parent's caches alive; the overlay has FRESH caches."""
    obs, ep_seed = _step_env_to_obs(seed=42, warmup=30, num_seats=2)
    world = World.from_obs(obs, episode_seed=ep_seed)
    _ = world.ledger_all(horizon=50)
    assert 50 in world._ledger_cache

    src = _our_planets_with_ships(world, my_id=0)[0]
    spec = LaunchSpec(src_id=src.id, aim_angle=0.0, ships=2, owner=0)
    overlay = world.with_candidate(spec)
    # Parent still cached.
    assert 50 in world._ledger_cache
    # Overlay has fresh, empty cache.
    assert overlay._ledger_cache == {}
    assert overlay._timeline_cache == {}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_overlay_raises_on_unknown_src():
    obs, ep_seed = _step_env_to_obs(seed=42, warmup=10, num_seats=2)
    world = World.from_obs(obs, episode_seed=ep_seed)
    with pytest.raises(ValueError, match="unknown src_id"):
        world.with_candidate(LaunchSpec(
            src_id=99999, aim_angle=0.0, ships=1, owner=0,
        ))


def test_overlay_raises_on_insufficient_ships():
    obs, ep_seed = _step_env_to_obs(seed=42, warmup=10, num_seats=2)
    world = World.from_obs(obs, episode_seed=ep_seed)
    src = _our_planets_with_ships(world, my_id=0)[0]
    with pytest.raises(ValueError, match="cannot launch"):
        world.with_candidate(LaunchSpec(
            src_id=src.id, aim_angle=0.0,
            ships=int(src.ships) + 999, owner=0,
        ))


def test_overlay_raises_on_zero_or_negative_ships():
    obs, ep_seed = _step_env_to_obs(seed=42, warmup=10, num_seats=2)
    world = World.from_obs(obs, episode_seed=ep_seed)
    src = _our_planets_with_ships(world, my_id=0)[0]
    with pytest.raises(ValueError, match="must be > 0"):
        world.with_candidate(LaunchSpec(
            src_id=src.id, aim_angle=0.0, ships=0, owner=0,
        ))


def test_overlay_raises_on_future_launch_turn():
    obs, ep_seed = _step_env_to_obs(seed=42, warmup=10, num_seats=2)
    world = World.from_obs(obs, episode_seed=ep_seed)
    src = _our_planets_with_ships(world, my_id=0)[0]
    with pytest.raises(NotImplementedError, match="launch_turn"):
        world.with_candidate(LaunchSpec(
            src_id=src.id, aim_angle=0.0, ships=1, owner=0,
            launch_turn=3,
        ))


# ---------------------------------------------------------------------------
# THE Phase 3 gate: overlay matches committed launch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [7, 42, 100, 314, 2026])
def test_overlay_ledger_matches_committed_launch(seed: int):
    """For each seed, pick a launch from our strongest planet
    toward the nearest enemy/neutral. Compute overlay's predicted
    arrivals; commit the launch via fast_sim and re-build a World
    from the resulting state. For every arrival from the synthetic
    fleet:

      overlay.arrival.eta == committed.arrival.eta + 1
      overlay.arrival.owner == committed.arrival.owner
      overlay.arrival.ships == committed.arrival.ships

    The synthetic fleet is identified by `from_planet_id` (the
    only fleet inbound from our chosen source on this step).
    """
    num_seats = 2
    my_id = 0
    obs_S, ep_seed = _step_env_to_obs(seed, warmup=30, num_seats=num_seats)
    world_S = World.from_obs(obs_S, episode_seed=ep_seed)

    # Pick a launch: from our strongest non-comet planet aimed at the
    # nearest non-ours target.
    owned = _our_planets_with_ships(world_S, my_id=my_id)
    if not owned:
        pytest.skip(f"seed={seed}: no usable source planet")
    src = max(owned, key=lambda p: p.ships)
    targets = [p for p in world_S.planets
               if p.owner != my_id and not p.is_comet]
    if not targets:
        pytest.skip(f"seed={seed}: no non-ours target")
    target = min(targets,
                 key=lambda p: math.hypot(p.current_x - src.current_x,
                                          p.current_y - src.current_y))
    aim = _pick_aim_angle_at_target(src, target)
    ships_to_send = int(src.ships) // 2 + 1
    spec = LaunchSpec(src_id=src.id, aim_angle=aim,
                       ships=ships_to_send, owner=my_id)

    # Overlay.
    overlay = world_S.with_candidate(spec)
    overlay_ledger = overlay.ledger_all(horizon=100)

    # Commit: run fast_sim with the action; build a fresh World.
    snap_S = fs_from_obs(obs_S, episode_seed=ep_seed, num_seats=num_seats)
    action = [[src.id, aim, ships_to_send]]
    snap_S_plus_1 = fs_step(snap_S, [action] + [[] for _ in range(num_seats - 1)])
    world_S_plus_1 = World.from_obs(snap_S_plus_1.obs, episode_seed=ep_seed)
    committed_ledger = world_S_plus_1.ledger_all(horizon=100)

    # Find the synthetic fleet's expected target (if any) in each.
    overlay_synth = [
        (pid, a) for pid, arrs in overlay_ledger.items()
        for a in arrs if a.fleet_id < 0  # synthetic ids are negative
    ]
    # In the committed state, the just-launched fleet is the one
    # with from_planet_id == src.id that wasn't there before.
    pre_existing_fleet_ids = {f.id for f in world_S.fleets}
    just_launched = [f for f in world_S_plus_1.fleets
                     if f.id not in pre_existing_fleet_ids
                     and f.from_planet_id == src.id]
    if not just_launched:
        # The fleet died in sun/OOB the very first step — overlay
        # should also have no synthetic arrival.
        assert not overlay_synth, \
            f"seed={seed}: overlay predicted arrival but env killed " \
            f"the fleet immediately"
        return

    assert len(just_launched) == 1, \
        f"seed={seed}: ambiguous just-launched fleets"
    real_synth_id = just_launched[0].id
    committed_synth = [
        (pid, a) for pid, arrs in committed_ledger.items()
        for a in arrs if a.fleet_id == real_synth_id
    ]

    # Either both predictors agree the fleet arrives somewhere, or
    # both agree it doesn't.
    if not overlay_synth and not committed_synth:
        return
    assert overlay_synth and committed_synth, \
        f"seed={seed}: divergence — overlay={overlay_synth} " \
        f"committed={committed_synth}"

    o_pid, o_arr = overlay_synth[0]
    c_pid, c_arr = committed_synth[0]
    assert o_pid == c_pid, \
        f"seed={seed}: target mismatch overlay={o_pid} committed={c_pid}"
    assert o_arr.eta == c_arr.eta + 1, \
        f"seed={seed}: eta mismatch overlay={o_arr.eta} " \
        f"committed_eta+1={c_arr.eta + 1}"
    assert o_arr.owner == c_arr.owner
    assert o_arr.ships == c_arr.ships


# ---------------------------------------------------------------------------
# Chained overlays
# ---------------------------------------------------------------------------


def test_with_candidates_chains():
    """Two sequential candidates from the SAME source decrement
    ships cumulatively."""
    obs, ep_seed = _step_env_to_obs(seed=42, warmup=10, num_seats=2)
    world = World.from_obs(obs, episode_seed=ep_seed)
    src = max(_our_planets_with_ships(world, my_id=0),
              key=lambda p: p.ships)
    initial_ships = int(src.ships)
    if initial_ships < 6:
        pytest.skip("source planet too small for the test")

    specs = [
        LaunchSpec(src_id=src.id, aim_angle=0.0, ships=2, owner=0),
        LaunchSpec(src_id=src.id, aim_angle=1.5, ships=3, owner=0),
    ]
    overlay = world.with_candidates(specs)
    assert overlay.planet_by_id(src.id).ships == initial_ships - 5
    assert len(overlay.fleets) == len(world.fleets) + 2
    # Synthetic fleet ids are unique negatives.
    synth_ids = [f.id for f in overlay.fleets if f.id < 0]
    assert len(set(synth_ids)) == len(synth_ids)
    # Parent unchanged.
    assert world.planet_by_id(src.id).ships == initial_ships
    assert len(world.fleets) == len(world.fleets)  # tautology, but pin


def test_with_candidates_empty_is_identity():
    """`with_candidates([])` returns the SAME world (no copy)."""
    obs, ep_seed = _step_env_to_obs(seed=42, warmup=10, num_seats=2)
    world = World.from_obs(obs, episode_seed=ep_seed)
    out = world.with_candidates([])
    assert out is world


# ---------------------------------------------------------------------------
# Spawn-step source-skip
# ---------------------------------------------------------------------------


def test_synthetic_fleet_does_not_self_hit_source():
    """A 1-ship fleet launched from a source planet must NOT have
    eta=1 hit-back on its source (env's spawn-step source skip).

    Geometry chosen to avoid the sun: source and target both at
    x=20 (column 30 units west of CENTER's x=50, so the fleet's
    vertical path stays clear of the sun's radius=10 disc).
    """
    obs = {
        "step": 0,
        "player": 0,
        "angular_velocity": 0.0,
        "planets": [
            # Source: big planet at (20, 20). Launch south along x=20.
            [0, 0, 20.0, 20.0, 3.0, 50, 2],
            # Target south at x=20 — no sun in the way.
            [1, -1, 20.0, 80.0, 1.0, 10, 1],
        ],
        "initial_planets": [
            [0, 0, 20.0, 20.0, 3.0, 50, 2],
            [1, -1, 20.0, 80.0, 1.0, 10, 1],
        ],
        "fleets": [],
        "comet_planet_ids": [],
        "comets": [],
        "next_fleet_id": 0,
    }
    world = World.from_obs(obs)
    # angle=pi/2 means "down" per the env convention (0=right, pi/2=down).
    spec = LaunchSpec(src_id=0, aim_angle=math.pi / 2,
                       ships=1, owner=0)
    overlay = world.with_candidate(spec)
    arrivals = overlay.ledger_all(horizon=200)
    # Source must not appear in arrivals.
    assert 0 not in arrivals, \
        f"synthetic fleet self-hit source: {arrivals.get(0)}"
    # Target must appear (the fleet should reach it eventually).
    assert 1 in arrivals, \
        f"target unreachable; arrivals={dict(arrivals)}"
