"""Phase 1 parity tests for lib.trajectory_layer — positions only.

For each seed × non-trivial start step, build a World and assert
that planet/fleet/comet positions at relative turn t match what
fast_sim produces when stepped forward t times. Tolerance: 1e-9
on floats; bit-exact on integer fields.

These tests cover the env's step-0 off-by-one rotation quirk
(`_effective_t_for_orbital`) — failing them means the trajectory
layer disagrees with the interpreter on basic kinematics.
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
from lib.trajectory_layer import World


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_actions(obs0: Any, num_seats: int, rng: random.Random) -> list[list]:
    """Same random-launch policy as test_fast_sim_parity.py — produces a
    non-trivial in-flight fleet population by the time we sample."""
    actions: list[list] = [[] for _ in range(num_seats)]
    planets = obs0["planets"] if isinstance(obs0, dict) else obs0.planets
    for p in planets:
        owner = p[1]
        if 0 <= owner < num_seats and p[5] > 5 and rng.random() < 0.3:
            actions[owner].append([p[0], rng.uniform(0.0, 6.283), int(p[5] // 2)])
    return actions


def _step_env_to_obs(seed: int, warmup: int, num_seats: int) -> tuple[Any, int]:
    """Drive a real env forward `warmup` ticks under random play; return
    (obs at that step, episode_seed). The obs is what World.from_obs
    consumes."""
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=num_seats)
    rng = random.Random(seed * 11 + 3)
    for _ in range(warmup):
        obs0 = env.state[0].observation
        env.step(_random_actions(obs0, num_seats, rng))
    return env.state[0].observation, int(env.info.get("seed", seed))


def _step_fast_sim_t_times(snap: Snapshot, t: int,
                            rng: random.Random, num_seats: int) -> Snapshot:
    """Step a fast_sim snapshot forward t times under EMPTY actions.
    No new launches — only motion / production / rotation / combat /
    spawn happen. That isolates the position-prediction question."""
    snap = fs_clone(snap)
    for _ in range(t):
        snap = fs_step(snap, [[] for _ in range(num_seats)], in_place=True)
    return snap


def _planet_xy_by_id(obs: Any) -> dict[int, tuple[float, float]]:
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    return {int(p[0]): (float(p[2]), float(p[3])) for p in planets}


def _fleet_xy_by_id(obs: Any) -> dict[int, tuple[float, float]]:
    fleets = obs["fleets"] if isinstance(obs, dict) else obs.fleets
    return {int(f[0]): (float(f[2]), float(f[3])) for f in fleets}


def _assert_close(a: tuple[float, float], b: tuple[float, float],
                  *, tol: float = 1e-9, label: str = "") -> None:
    assert math.isclose(a[0], b[0], abs_tol=tol), \
        f"{label}: x mismatch {a[0]} vs {b[0]} (Δ={a[0]-b[0]:.3e})"
    assert math.isclose(a[1], b[1], abs_tol=tol), \
        f"{label}: y mismatch {a[1]} vs {b[1]} (Δ={a[1]-b[1]:.3e})"


# ---------------------------------------------------------------------------
# Static + orbiting planet position parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [7, 42, 100])
@pytest.mark.parametrize("warmup", [0, 5, 25])
@pytest.mark.parametrize("t", [0, 1, 5, 20, 50])
def test_planet_position_parity_seeded(seed: int, warmup: int, t: int):
    """Every alive planet's position at relative t matches what
    fast_sim produces after stepping forward t times under empty
    actions. Tolerance 1e-9.

    Covers static (outer ring) AND orbiting (inner ring) planets in
    the same test — the matrix of (seed × warmup) gives enough
    variation to exercise both populations.
    """
    num_seats = 2
    obs_S, ep_seed = _step_env_to_obs(seed, warmup, num_seats)
    snap_S = fs_from_obs(obs_S, configuration=None,
                          episode_seed=ep_seed, num_seats=num_seats)
    world_S = World.from_obs(obs_S, configuration=None,
                              episode_seed=ep_seed)

    # Step fast_sim forward t under empty actions; this is "ground
    # truth" for planet positions.
    rng = random.Random((seed * 31 + warmup) % 2**31)
    snap_T = _step_fast_sim_t_times(snap_S, t, rng, num_seats)
    truth_xy = _planet_xy_by_id(snap_T.obs)

    # Only assert on planets that existed in the original World — comets
    # spawned mid-window (steps 50/150/250/350/450 cross) can't be
    # predicted by the layer without re-simulating, and the contract is
    # that they're surfaced as UNCERTAIN in Phase 4. For Phase 1, skip.
    sample_pids = {p.id for p in world_S.planets}
    checked = 0
    for pid, expected in truth_xy.items():
        if pid not in sample_pids:
            continue
        predicted = world_S.planet_position(pid, t)
        assert predicted is not None, \
            f"World returned None for planet {pid} (in sample) at t={t}"
        _assert_close(predicted, expected, tol=1e-9,
                      label=f"seed={seed} warmup={warmup} t={t} pid={pid}")
        checked += 1
    assert checked > 0, "no planets checked — sample population empty"


# ---------------------------------------------------------------------------
# Off-by-one pin (the parity-critical step=0 quirk)
# ---------------------------------------------------------------------------


def test_planet_position_off_by_one_step_zero():
    """At obs.step=0, the env hasn't rotated planets yet. So:
      - planet_position(pid, 0) returns the INITIAL position.
      - planet_position(pid, 1) ALSO returns the initial position
        (the rotation for step=0's interpretation produces the
        SAME position because angle = init + omega*0).
      - planet_position(pid, 2) returns init + omega*1.

    This is the off-by-one accounting documented in
    `lib/foundation/predictor.py:208-220` and the trajectory_layer
    module docstring. If this test fails the layer is silently
    one-step-ahead of the env on every orbital query.
    """
    seed = 42
    num_seats = 2
    obs0, ep_seed = _step_env_to_obs(seed, warmup=0, num_seats=num_seats)
    snap0 = fs_from_obs(obs0, episode_seed=ep_seed, num_seats=num_seats)
    world = World.from_obs(obs0, episode_seed=ep_seed)
    assert world.step == 0

    # Find an orbiting (rotating) planet.
    rotating = [p for p in world.planets if p.is_rotating and not p.is_comet]
    assert rotating, "no rotating planet in this seed; pick another seed"
    p0 = rotating[0]

    # t=0 → current (== initial for step=0).
    pos_0 = world.planet_position(p0.id, 0)
    assert pos_0 == (p0.current_x, p0.current_y)
    assert math.isclose(pos_0[0], p0.init_x, abs_tol=1e-12)
    assert math.isclose(pos_0[1], p0.init_y, abs_tol=1e-12)

    # t=1 — fast_sim ground truth.
    rng = random.Random(seed)
    snap_t1 = _step_fast_sim_t_times(snap0, 1, rng, num_seats)
    truth_t1 = _planet_xy_by_id(snap_t1.obs)[p0.id]
    predicted_t1 = world.planet_position(p0.id, 1)
    _assert_close(predicted_t1, truth_t1, tol=1e-9,
                  label="step=0, t=1")

    # t=2 — fast_sim ground truth (this one DOES move from init).
    snap_t2 = _step_fast_sim_t_times(snap0, 2, rng, num_seats)
    truth_t2 = _planet_xy_by_id(snap_t2.obs)[p0.id]
    predicted_t2 = world.planet_position(p0.id, 2)
    _assert_close(predicted_t2, truth_t2, tol=1e-9,
                  label="step=0, t=2")

    # Sanity: t=2 differs from t=0 (the planet moved).
    assert not math.isclose(predicted_t2[0], pos_0[0], abs_tol=1e-6) \
        or not math.isclose(predicted_t2[1], pos_0[1], abs_tol=1e-6), \
        "planet didn't move from t=0 to t=2 — orbital math broken"


# ---------------------------------------------------------------------------
# Fleet straight-line position parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [11, 99, 314])
@pytest.mark.parametrize("t", [1, 3, 10])
def test_fleet_position_straight_line_parity(seed: int, t: int):
    """For each in-flight fleet at the sample step, World's straight-
    line projection at relative t matches fast_sim's position after
    t turns of EMPTY actions — when the fleet doesn't collide.

    We filter to fleets that are STILL alive at t (no collision):
    those are the cases where the straight-line projection is the
    correct answer. A future Phase 2 ledger test will cover the
    collision case.
    """
    num_seats = 2
    obs_S, ep_seed = _step_env_to_obs(seed, warmup=20, num_seats=num_seats)
    if not (obs_S["fleets"] if isinstance(obs_S, dict) else obs_S.fleets):
        pytest.skip("no in-flight fleets at sample step")

    snap_S = fs_from_obs(obs_S, episode_seed=ep_seed, num_seats=num_seats)
    world_S = World.from_obs(obs_S, episode_seed=ep_seed)

    # Step fast_sim forward t with empty actions.
    rng = random.Random(seed)
    snap_T = _step_fast_sim_t_times(snap_S, t, rng, num_seats)
    truth_xy = _fleet_xy_by_id(snap_T.obs)

    # For each ORIGINALLY in-flight fleet, if it survived to step
    # S+t, its position should match World's projection.
    for f in world_S.fleets:
        if f.id not in truth_xy:
            # Fleet collided / OOB'd in the interim — skip; Phase 2's
            # ledger will cover this.
            continue
        predicted = world_S.fleet_position(f.id, t)
        assert predicted is not None
        _assert_close(predicted, truth_xy[f.id], tol=1e-9,
                      label=f"seed={seed} t={t} fid={f.id}")


# ---------------------------------------------------------------------------
# Comet position + expiry
# ---------------------------------------------------------------------------


def test_comet_position_in_bounds():
    """At seed=42, warmup=60 we're past the step-50 comet spawn so
    there are 4 active comets. World.comet_position(pid, 0) returns
    the obs's current position; positive t reads forward in the path."""
    seed = 42
    num_seats = 2
    obs_S, ep_seed = _step_env_to_obs(seed, warmup=60, num_seats=num_seats)
    world_S = World.from_obs(obs_S, episode_seed=ep_seed)
    if not world_S.comet_paths:
        pytest.skip("no comets at warmup=60 for this seed")

    snap_S = fs_from_obs(obs_S, episode_seed=ep_seed, num_seats=num_seats)

    # For each comet, query t in {0, 1, 5} and compare to fast_sim
    # ground truth (planet position of that comet pid after stepping).
    for cpath in world_S.comet_paths:
        for t in (0, 1, 5):
            rng = random.Random(seed)
            snap_T = _step_fast_sim_t_times(snap_S, t, rng, num_seats)
            truth = _planet_xy_by_id(snap_T.obs).get(cpath.planet_id)
            predicted = world_S.comet_position(cpath.planet_id, t)
            if truth is None:
                # Comet expired in the interim — predictor should also
                # return None at that t.
                assert predicted is None, \
                    f"comet {cpath.planet_id} expired at t={t} but " \
                    f"World predicted {predicted}"
                continue
            assert predicted is not None, \
                f"comet {cpath.planet_id} predicted None at t={t} but " \
                f"truth is {truth}"
            _assert_close(predicted, truth, tol=1e-9,
                          label=f"comet pid={cpath.planet_id} t={t}")


def test_comet_position_expiry_returns_none():
    """Past the end of the path, the comet has left the board; the
    predictor returns None."""
    # Construct a synthetic comet path with 3 entries; query t > 2.
    from lib.trajectory_layer import CometPathView
    c = CometPathView(
        planet_id=999,
        path=((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)),
        path_index=2,
    )
    assert c.position_at(0) == (5.0, 6.0)
    assert c.position_at(1) is None  # would be index 3, out of range
    assert c.position_at(100) is None
    # Negative t past the start is also None.
    assert c.position_at(-3) is None


# ---------------------------------------------------------------------------
# Lookup sanity
# ---------------------------------------------------------------------------


def test_unknown_ids_return_none():
    """Querying a planet/fleet/comet id that doesn't exist returns
    None (not crash, not a stale value)."""
    seed = 42
    obs_S, ep_seed = _step_env_to_obs(seed, warmup=5, num_seats=2)
    world = World.from_obs(obs_S, episode_seed=ep_seed)
    assert world.planet_position(99999, 0) is None
    assert world.planet_position(99999, 10) is None
    assert world.fleet_position(99999, 1) is None
    assert world.comet_position(99999, 0) is None
    assert world.planet_by_id(99999) is None
    assert world.fleet_by_id(99999) is None
    assert world.comet_by_planet_id(99999) is None
    assert not world.is_comet(99999)
