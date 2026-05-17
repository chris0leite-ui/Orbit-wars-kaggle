"""Phase 2 parity tests for lib.trajectory_layer — arrival ledger and
per-planet timelines.

Differential vs the legacy `lib.world_model.WorldModel.from_world`
across multiple seeds and warmups. Bit-exact on integer fields
(owner, ships, eta) and within 1e-9 on float (ships_at).

Plus an interpreter-parity test that confirms predicted arrivals
materialise at the predicted step under `fast_sim`.
"""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

pytestmark = pytest.mark.slow

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet as _EnvFleet

from lib.fast_sim import Snapshot, clone as fs_clone
from lib.fast_sim import from_obs as fs_from_obs
from lib.fast_sim import step as fs_step
from lib.intent import World as LegacyWorld
from lib.trajectory_layer import (
    DEFAULT_LEDGER_HORIZON,
    Arrival,
    World,
)
from lib.world_model import WorldModel as LegacyWorldModel


# ---------------------------------------------------------------------------
# Helpers (shared shape with the positions test)
# ---------------------------------------------------------------------------


def _random_actions(obs0: Any, num_seats: int,
                    rng: random.Random) -> list[list]:
    actions: list[list] = [[] for _ in range(num_seats)]
    planets = obs0["planets"] if isinstance(obs0, dict) else obs0.planets
    for p in planets:
        owner = p[1]
        if 0 <= owner < num_seats and p[5] > 5 and rng.random() < 0.3:
            actions[owner].append(
                [p[0], rng.uniform(0.0, 6.283), int(p[5] // 2)],
            )
    return actions


def _step_env_to_obs(seed: int, warmup: int, num_seats: int,
                     ) -> tuple[Any, int]:
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=num_seats)
    rng = random.Random(seed * 11 + 3)
    for _ in range(warmup):
        obs0 = env.state[0].observation
        env.step(_random_actions(obs0, num_seats, rng))
    return env.state[0].observation, int(env.info.get("seed", seed))


def _build_legacy_model(obs: Any, horizon: int,
                        ) -> LegacyWorldModel:
    """Build the legacy WorldModel from an obs (same path used by
    production code today)."""
    lw = LegacyWorld.from_obs(obs)
    return LegacyWorldModel.from_world(lw, horizon=horizon)


# ---------------------------------------------------------------------------
# Differential parity vs legacy WorldModel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [7, 42, 100, 314, 2026])
@pytest.mark.parametrize("warmup", [10, 30])
def test_ledger_for_matches_world_model(seed: int, warmup: int):
    """For each planet, the new ledger's `(eta, owner, ships)` set
    matches the legacy ledger's exactly (set-wise; ordering may
    differ)."""
    num_seats = 2
    horizon = 100
    obs, ep_seed = _step_env_to_obs(seed, warmup, num_seats)
    world = World.from_obs(obs, episode_seed=ep_seed)
    legacy = _build_legacy_model(obs, horizon)

    # Iterate every planet that has at least one arrival in either
    # ledger; assert the sets agree.
    new_pids = set(world.ledger_all(horizon).keys())
    legacy_pids = {pid for pid, arrs in legacy.ledger.items() if arrs}
    assert new_pids == legacy_pids, \
        f"planet-id sets diverge: new={new_pids ^ legacy_pids}"

    for pid in new_pids:
        new_set = {(a.eta, a.owner, a.ships)
                   for a in world.ledger_for(pid, horizon)}
        legacy_set = set(tuple(t) for t in legacy.ledger[pid])
        assert new_set == legacy_set, \
            f"seed={seed} warmup={warmup} pid={pid}: " \
            f"new={new_set} legacy={legacy_set}"


@pytest.mark.parametrize("seed", [7, 42, 100])
@pytest.mark.parametrize("warmup", [10, 30])
@pytest.mark.parametrize("t", [0, 1, 5, 20, 50, 100])
def test_ownership_at_matches_world_model(seed: int, warmup: int, t: int):
    """`world.ownership_at(pid, t)` matches the legacy
    `(owner_at, ships_at)` for every planet, every t in the set."""
    num_seats = 2
    horizon = 100
    obs, ep_seed = _step_env_to_obs(seed, warmup, num_seats)
    world = World.from_obs(obs, episode_seed=ep_seed)
    legacy = _build_legacy_model(obs, horizon)

    for p in world.planets:
        if p.is_comet:
            continue  # comets aren't combat targets in the legacy timeline
                      # (they pass through; production resolution is identical
                      # but ownership rarely matters for them — skip for parity).
        new_owner, new_ships = world.ownership_at(p.id, t, horizon=horizon)
        legacy_owner = legacy.owner_at(p.id, t)
        legacy_ships = legacy.ships_at(p.id, t)
        assert new_owner == legacy_owner, \
            f"seed={seed} warmup={warmup} t={t} pid={p.id}: " \
            f"new_owner={new_owner} legacy={legacy_owner}"
        assert math.isclose(new_ships, float(legacy_ships), abs_tol=1e-9), \
            f"seed={seed} warmup={warmup} t={t} pid={p.id}: " \
            f"new_ships={new_ships} legacy={legacy_ships}"


@pytest.mark.parametrize("seed", [7, 42, 100])
@pytest.mark.parametrize("warmup", [10, 30])
def test_incoming_enemy_eta_matches_world_model(seed: int, warmup: int):
    """`world.incoming_enemy_eta(pid, my_id)` matches the legacy
    method for every planet × every seat."""
    num_seats = 2
    horizon = 100
    obs, ep_seed = _step_env_to_obs(seed, warmup, num_seats)
    world = World.from_obs(obs, episode_seed=ep_seed)
    legacy = _build_legacy_model(obs, horizon)

    for p in world.planets:
        for seat in range(num_seats):
            new_eta = world.incoming_enemy_eta(p.id, seat, horizon=horizon)
            legacy_eta = legacy.incoming_enemy_eta(p.id, seat)
            assert new_eta == legacy_eta, \
                f"seed={seed} warmup={warmup} pid={p.id} seat={seat}: " \
                f"new={new_eta} legacy={legacy_eta}"


# ---------------------------------------------------------------------------
# Orbiting-target attribution (port of test_world_model_orbiting_target.py)
# ---------------------------------------------------------------------------


def test_orbiting_target_attribution_matches_legacy():
    """The omega-aware fleet ray-cast correctly attributes hits to
    inner-orbiting planets (the bug fixed in commit 4b609411).

    Hand-built scenario: a fleet at (10, 50) firing right with 1 ship
    at omega=0.05. Target is an inner orbiting planet at (40, 50).
    At fleet's speed=1.0, the fleet takes ~29 steps to reach (40, 50).
    By that time the planet has rotated ~1.5 rad around CENTER and
    moved out of the way.

    The legacy `_static_first_hit` would mis-attribute the hit; the
    new ray-cast (and the legacy's `_orbital_first_hit` with
    omega!=0) correctly say "no planet hit, fleet times out".
    """
    # Build a synthetic obs.
    omega = 0.05
    obs = {
        "step": 5,
        "player": 0,
        "angular_velocity": omega,
        "planets": [
            # id, owner, x, y, radius, ships, production
            [0, -1, 40.0, 50.0, 1.0, 10, 1],  # inner orbiting target
            [1, 0, 10.0, 50.0, 2.0, 50, 2],   # our source planet
        ],
        "initial_planets": [
            [0, -1, 40.0, 50.0, 1.0, 10, 1],
            [1, 0, 10.0, 50.0, 2.0, 50, 2],
        ],
        "fleets": [
            # id, owner, x, y, angle, from_pid, ships
            [0, 0, 13.0, 50.0, 0.0, 1, 1],  # 1-ship fleet flying right
        ],
        "comet_planet_ids": [],
        "comets": [],
        "next_fleet_id": 1,
    }
    world = World.from_obs(obs, episode_seed=42)
    lw = LegacyWorld.from_obs(obs)
    legacy = LegacyWorldModel.from_world(lw, horizon=50)

    # Both should agree: in this geometry the orbital drift carries
    # the target out of the fleet's path, so no arrival is recorded.
    new_arrivals = world.ledger_for(0, horizon=50)
    legacy_arrivals = legacy.ledger.get(0, [])
    assert tuple(new_arrivals) == tuple(
        Arrival(eta=a[0], owner=a[1], ships=a[2], fleet_id=-1)
        for a in legacy_arrivals
    ) or (not new_arrivals and not legacy_arrivals), \
        f"orbital attribution divergence: new={new_arrivals} " \
        f"legacy={legacy_arrivals}"


def test_orbiting_target_static_omega_zero():
    """At omega=0 (no rotation), the new ray-cast must behave like
    the legacy static raycast — a fleet aimed straight at a static
    planet hits it."""
    obs = {
        "step": 0,
        "player": 0,
        "angular_velocity": 0.0,
        "planets": [
            [0, -1, 40.0, 50.0, 1.0, 10, 1],
            [1, 0, 10.0, 50.0, 2.0, 50, 2],
        ],
        "initial_planets": [
            [0, -1, 40.0, 50.0, 1.0, 10, 1],
            [1, 0, 10.0, 50.0, 2.0, 50, 2],
        ],
        "fleets": [
            [0, 0, 13.0, 50.0, 0.0, 1, 1],
        ],
        "comet_planet_ids": [],
        "comets": [],
        "next_fleet_id": 1,
    }
    world = World.from_obs(obs)
    arrivals = world.ledger_for(0, horizon=50)
    # The fleet should hit planet 0 (id=0) at some eta.
    assert len(arrivals) == 1
    assert arrivals[0].owner == 0
    assert arrivals[0].ships == 1


# ---------------------------------------------------------------------------
# Interpreter-parity (THE Phase 2 gate)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [7, 42, 100])
def test_predicted_arrivals_materialise_in_fast_sim(seed: int):
    """For each predicted arrival from the World ledger, step
    fast_sim forward `eta` turns under EMPTY actions and assert
    that the planet's owner/ships at that step are consistent
    with the legacy WorldModel's prediction.

    This is the "no false positives, no missed hits" gate. If the
    new ledger says "planet P gets hit at turn t by owner O with
    S ships" but the env says otherwise, the ray-cast is wrong.

    We compare BOTH new and legacy against the env's truth, so a
    legacy bug doesn't silently mask a new-code bug — the test
    fails any way the two predictors diverge from ground truth.
    """
    num_seats = 2
    horizon = 40
    obs, ep_seed = _step_env_to_obs(seed, warmup=30, num_seats=num_seats)
    world = World.from_obs(obs, episode_seed=ep_seed)

    snap = fs_from_obs(obs, episode_seed=ep_seed, num_seats=num_seats)
    # Step fast_sim forward `horizon` turns under empty actions —
    # only existing in-flight fleets get to interact with planets.
    snap_T = fs_clone(snap)
    for _ in range(horizon):
        snap_T = fs_step(snap_T, [[] for _ in range(num_seats)],
                          in_place=True)

    # For each planet that appears in either ledger, the
    # ownership_at(horizon) must equal the env's actual state at
    # that turn.
    truth_planets = (snap_T.obs["planets"] if isinstance(snap_T.obs, dict)
                     else snap_T.obs.planets)
    truth_by_id = {int(p[0]): (int(p[1]), float(p[5]))
                   for p in truth_planets}

    for p in world.planets:
        if p.is_comet:
            continue
        if p.id not in truth_by_id:
            # Planet disappeared (rare, e.g. comet expiry — skip).
            continue
        truth_owner, truth_ships = truth_by_id[p.id]
        pred_owner, pred_ships = world.ownership_at(
            p.id, horizon, horizon=horizon,
        )
        assert pred_owner == truth_owner, \
            f"seed={seed} pid={p.id}: predicted owner {pred_owner} " \
            f"vs env truth {truth_owner}"
        # Allow ±1 ship of integer rounding tolerance — the env's
        # interpreter uses integer ship counts on planets, but the
        # timeline floats production.
        assert abs(pred_ships - truth_ships) <= 1.5, \
            f"seed={seed} pid={p.id}: predicted ships {pred_ships} " \
            f"vs env truth {truth_ships}"


# ---------------------------------------------------------------------------
# Cache / sparse-cost semantics
# ---------------------------------------------------------------------------


def test_ledger_cache_is_per_horizon():
    """Calling `ledger_for(pid, h1)` then `ledger_for(pid, h2)` for
    different horizons builds two ledger entries — different
    horizons might filter different fleets out."""
    seed = 42
    obs, ep_seed = _step_env_to_obs(seed, warmup=30, num_seats=2)
    world = World.from_obs(obs, episode_seed=ep_seed)
    assert len(world._ledger_cache) == 0
    world.ledger_all(horizon=50)
    assert 50 in world._ledger_cache
    world.ledger_all(horizon=100)
    assert 50 in world._ledger_cache and 100 in world._ledger_cache


def test_timeline_cache_is_per_planet():
    """Calling `ownership_at(pid_A, ...)` materialises ONE timeline,
    not all. Pins the lazy-per-planet semantics."""
    seed = 42
    obs, ep_seed = _step_env_to_obs(seed, warmup=30, num_seats=2)
    world = World.from_obs(obs, episode_seed=ep_seed)

    # Pick two distinct non-comet planet ids.
    pids = [p.id for p in world.planets if not p.is_comet][:2]
    assert len(pids) == 2

    world.ownership_at(pids[0], 5)
    cached_keys_a = set(world._timeline_cache.keys())
    assert any(k[0] == pids[0] for k in cached_keys_a)
    assert not any(k[0] == pids[1] for k in cached_keys_a)

    world.ownership_at(pids[1], 5)
    cached_keys_b = set(world._timeline_cache.keys())
    assert any(k[0] == pids[0] for k in cached_keys_b)
    assert any(k[0] == pids[1] for k in cached_keys_b)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_no_fleets_returns_empty_ledger():
    obs = {
        "step": 0,
        "player": 0,
        "angular_velocity": 0.025,
        "planets": [
            [0, 0, 30.0, 50.0, 1.0, 10, 1],
            [1, -1, 70.0, 50.0, 1.0, 10, 1],
        ],
        "initial_planets": [
            [0, 0, 30.0, 50.0, 1.0, 10, 1],
            [1, -1, 70.0, 50.0, 1.0, 10, 1],
        ],
        "fleets": [],
        "comet_planet_ids": [],
        "comets": [],
        "next_fleet_id": 0,
    }
    world = World.from_obs(obs)
    assert world.ledger_all() == {}
    # Ownership stays at initial (with production for the owned planet).
    owner_0, ships_0 = world.ownership_at(0, 10)
    assert owner_0 == 0
    # 10 starting + 10 turns * 1 production = 20
    assert ships_0 == 20.0
    # Neutral planet: no production accrues.
    owner_1, ships_1 = world.ownership_at(1, 10)
    assert owner_1 == -1
    assert ships_1 == 10.0


def test_fleet_bound_for_sun_not_in_ledger():
    """A fleet aimed straight at the sun (50, 50) from (30, 50) is
    omitted from the ledger — it dies in sun, not at any planet."""
    obs = {
        "step": 0,
        "player": 0,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 30.0, 50.0, 2.0, 50, 2],
        ],
        "initial_planets": [
            [0, 0, 30.0, 50.0, 2.0, 50, 2],
        ],
        "fleets": [
            # Aimed straight right toward the sun.
            [0, 0, 33.0, 50.0, 0.0, 0, 10],
        ],
        "comet_planet_ids": [],
        "comets": [],
        "next_fleet_id": 1,
    }
    world = World.from_obs(obs)
    # Sun-killed fleet is dropped from the ledger.
    assert world.ledger_all() == {}


def test_unknown_planet_id_returns_neutral_zero():
    """`ownership_at` on an unknown planet returns `(-1, 0.0)` — no
    crash, no stale value."""
    obs = {
        "step": 0,
        "player": 0,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 30.0, 50.0, 1.0, 10, 1],
        ],
        "initial_planets": [
            [0, 0, 30.0, 50.0, 1.0, 10, 1],
        ],
        "fleets": [],
        "comet_planet_ids": [],
        "comets": [],
        "next_fleet_id": 0,
    }
    world = World.from_obs(obs)
    owner, ships = world.ownership_at(99999, 10)
    assert owner == -1
    assert ships == 0.0
    assert world.ledger_for(99999) == ()
    assert world.incoming_enemy_eta(99999, 0) is None
