"""Brute-force parity gate for the Phase γ predict_fleet_fate swap.

For 100 fixed + ~1000 randomized `(src, tgt, angle, ships, wait_N)`
tuples sampled across several seed-built worlds, assert that
`predict_fleet_fate(...)` returns the SAME `FleetFate` with the
kinematic table enabled vs. disabled.

`FleetFate` is a frozen dataclass — `==` compares `(outcome,
hit_planet_id, step)` exactly. Any divergence is a Phase γ bug that
must be fixed before agent wiring escalates.

Plan: /root/.claude/plans/do-it-thoroughly-consider-tingly-fox.md
"""

from __future__ import annotations

import os
import random

import pytest

from kaggle_environments import make
from lib.intent import World
from lib.kinematic_table import KinematicTable, get_default
from lib.trajectory import predict_fleet_fate


# ---------------------------------------------------------------------------
# World construction from real env seeds.
# ---------------------------------------------------------------------------


def _world_from_seed(seed: int, episode_step: int = 0) -> World:
    """Build a `World` from the env's actual initial state for `seed`."""
    env = make("orbit_wars", configuration={"seed": int(seed)}, debug=False)
    # Reset to ensure a fresh state.
    env.reset()
    # env.steps[0] is the initial-state observation per agent (4 in 4P, 2 in 2P).
    # The env's default is 4P; we take agent 0's obs.
    obs = env.steps[0][0]["observation"]
    # The kaggle obs is a dict-like Struct; coerce to plain dict.
    if not isinstance(obs, dict):
        obs = {k: getattr(obs, k) for k in dir(obs) if not k.startswith("_")}
    # If we want a later step, advance via env.run with random agents
    # (the parity gate doesn't care WHICH game-state we test, only
    # that it's a real env state).
    if episode_step > 0:
        env.run([
            "random", "random",
        ] if len(env.steps[0]) == 2 else ["random"] * len(env.steps[0]))
        # Pick a step. env.steps[i] is per-step list of per-agent states.
        if episode_step >= len(env.steps):
            episode_step = len(env.steps) - 1
        obs2 = env.steps[episode_step][0]["observation"]
        if not isinstance(obs2, dict):
            obs2 = {k: getattr(obs2, k) for k in dir(obs2) if not k.startswith("_")}
        obs = obs2
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# Parity assertion helpers.
# ---------------------------------------------------------------------------


def _predict_with_table(world, src, target, angle, ships, wait_N):
    """Run predict_fleet_fate WITH kinematic table primed + enabled."""
    os.environ["KINEMATIC_TABLE_ENABLED"] = "1"
    try:
        # Prime the singleton.
        from lib.kinematic_table import begin_turn as _kt_begin_turn, clear
        clear()
        _kt_begin_turn(world)
        return predict_fleet_fate(src, target, angle, ships, world, wait_N=wait_N)
    finally:
        os.environ.pop("KINEMATIC_TABLE_ENABLED", None)


def _predict_without_table(world, src, target, angle, ships, wait_N):
    """Run predict_fleet_fate with KT off → inline build path."""
    os.environ.pop("KINEMATIC_TABLE_ENABLED", None)
    # Reset the table to ensure no leakage.
    from lib.kinematic_table import clear
    clear()
    return predict_fleet_fate(src, target, angle, ships, world, wait_N=wait_N)


# ---------------------------------------------------------------------------
# Parity tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [42, 13, 99, 137, 256])
def test_parity_100_random_cases_per_seed(seed):
    """100 random (src, tgt, angle, ships, wait_N) per seed — assert
    table-backed result == inline-backed result. 5 seeds × 100 = 500
    cases."""
    world = _world_from_seed(seed, episode_step=0)
    planets = list(world.planets_by_id.values())
    if len(planets) < 2:
        pytest.skip(f"seed {seed} has <2 planets at step 0")

    rng = random.Random(seed * 7919 + 1)
    diverged: list[tuple] = []
    for _ in range(100):
        src = planets[rng.randrange(len(planets))]
        tgt = planets[rng.randrange(len(planets))]
        # Random angle covering full circle.
        angle = rng.uniform(-3.14159265, 3.14159265)
        # Ships size buckets covering the speed range.
        ships = rng.choice([1, 3, 7, 15, 30, 60, 120, 250, 500])
        # wait_N in a realistic range.
        wait_N = rng.choice([0, 0, 0, 1, 3, 7, 15, 25])

        with_t = _predict_with_table(world, src, tgt, angle, ships, wait_N)
        without_t = _predict_without_table(world, src, tgt, angle, ships, wait_N)
        if with_t != without_t:
            diverged.append((
                seed, src.id, tgt.id, angle, ships, wait_N, with_t, without_t,
            ))

    assert not diverged, (
        f"seed {seed}: {len(diverged)} of 100 cases diverged.\n"
        f"first: {diverged[0]}"
    )


@pytest.mark.parametrize("seed", [42, 7])
def test_parity_fixed_edge_cases(seed):
    """Targeted edge cases: wait_N=0 fire-now, wait_N=20 deep-future,
    extreme angles, small/large ships."""
    world = _world_from_seed(seed)
    planets = list(world.planets_by_id.values())
    if len(planets) < 2:
        pytest.skip(f"seed {seed} too few planets")

    src = planets[0]
    tgt = planets[1] if len(planets) > 1 else planets[0]
    cases = [
        # (angle, ships, wait_N)
        (0.0, 1, 0),
        (0.0, 1000, 0),
        (0.5, 50, 5),
        (-1.5, 7, 12),
        (3.0, 25, 20),
        (1.0, 100, 0),
        (-0.7, 250, 15),
    ]
    diverged = []
    for angle, ships, wait_N in cases:
        a = _predict_with_table(world, src, tgt, angle, ships, wait_N)
        b = _predict_without_table(world, src, tgt, angle, ships, wait_N)
        if a != b:
            diverged.append((angle, ships, wait_N, a, b))
    assert not diverged, (
        f"seed {seed}: edge-case parity broke for {diverged}"
    )


def test_parity_with_comet_present_seed42_midgame():
    """A seed-42 mid-game state where comets ARE in the obs. Exercises
    the comet code path of `table.window` + `_comet_paths_by_id` view
    swap."""
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.reset()
    # Play some random steps to get past comet-spawn tick 50.
    env.run(["random", "random"])
    # Pick a step that has comets.
    chosen_step = None
    for i in range(50, min(len(env.steps), 100)):
        obs = env.steps[i][0]["observation"]
        if not isinstance(obs, dict):
            obs_d = {k: getattr(obs, k) for k in dir(obs) if not k.startswith("_")}
        else:
            obs_d = obs
        if obs_d.get("comet_planet_ids"):
            chosen_step = i
            break
    if chosen_step is None:
        pytest.skip("no comets visible in random self-play; rerun")

    obs = env.steps[chosen_step][0]["observation"]
    obs_d = obs if isinstance(obs, dict) else {
        k: getattr(obs, k) for k in dir(obs) if not k.startswith("_")
    }
    world = World.from_obs(obs_d)
    assert len(world.comet_ids) > 0, "expected comets in this world"

    planets = list(world.planets_by_id.values())
    src = next(p for p in planets if p.id not in world.comet_ids)
    # Comet target: exercises the comet path lookup specifically.
    comet_id = next(iter(world.comet_ids))
    tgt = world.planets_by_id[comet_id]

    # Sample angles and ship sizes.
    rng = random.Random(42)
    diverged = []
    for _ in range(50):
        angle = rng.uniform(-3.14159265, 3.14159265)
        ships = rng.choice([1, 5, 25, 100, 250])
        wait_N = rng.choice([0, 0, 2, 8, 18])
        a = _predict_with_table(world, src, tgt, angle, ships, wait_N)
        b = _predict_without_table(world, src, tgt, angle, ships, wait_N)
        if a != b:
            diverged.append((angle, ships, wait_N, a, b))
    assert not diverged, (
        f"comet-target parity broke: {diverged[:3]} (showing first 3)"
    )
