"""Sub-phase 2 parity: JAX WorldModel pieces vs scalar `lib.world_model`.

Currently tests:
- `fleet_target_planet_batch` vs scalar `fleet_target_planet`

Future sub-phases test build_arrival_ledger and simulate_planet_timeline.
"""

from __future__ import annotations

import math
import random

import jax.numpy as jnp
import numpy as np
import pytest

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from lib.world_model import fleet_target_planet
from lib.game.jax.jax_world_model import (
    fleet_speed_batch, fleet_target_planet_batch,
    DEFAULT_HORIZON,
)


def _spawn_in_flight_fleets(env, num_agents: int = 2, n_steps: int = 15, rng_seed: int = 7):
    """Run N steps with random-policy launches so the env has live fleets."""
    rng = random.Random(rng_seed)
    for _ in range(n_steps):
        if env.state[0].status != "ACTIVE":
            break
        actions = []
        for pid_seat in range(num_agents):
            moves = []
            obs = env.state[pid_seat].observation
            for p in obs.planets:
                if p[1] == pid_seat and p[5] > 0 and rng.random() < 0.5:
                    angle = rng.uniform(0, 2 * math.pi)
                    ships = max(1, int(p[5] * rng.uniform(0.1, 0.7)))
                    if 0 < ships <= p[5]:
                        moves.append([p[0], angle, ships])
            actions.append(moves)
        env.step(actions)


@pytest.mark.parametrize("seed", [0, 7, 42, 137])
def test_fleet_target_planet_batch_parity(seed):
    """JAX raycast finds the same target + ETA as the scalar version
    for every in-flight fleet in a mid-game state.
    """
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _spawn_in_flight_fleets(env, num_agents=2)

    obs = env.state[0].observation
    fleets_raw = list(obs.fleets)
    planets_raw = list(obs.planets)
    if not fleets_raw:
        pytest.skip(f"seed {seed} produced no in-flight fleets")

    # Scalar reference: per-fleet scan.
    fleet_named = [Fleet(*f) for f in fleets_raw]
    planet_named = [Planet(*p) for p in planets_raw]
    scalar_targets = []
    scalar_etas = []
    for f in fleet_named:
        t, e = fleet_target_planet(f, planet_named, DEFAULT_HORIZON)
        scalar_targets.append(t.id if t is not None else None)
        scalar_etas.append(e)

    # JAX inputs: build arrays from the obs raw data.
    fleets_x = jnp.array([float(f[2]) for f in fleets_raw], dtype=jnp.float32)
    fleets_y = jnp.array([float(f[3]) for f in fleets_raw], dtype=jnp.float32)
    fleets_angle = jnp.array([float(f[4]) for f in fleets_raw], dtype=jnp.float32)
    fleets_ships = jnp.array([int(f[6]) for f in fleets_raw], dtype=jnp.int32)
    fleets_alive = jnp.ones(len(fleets_raw), dtype=bool)
    planets_x = jnp.array([float(p[2]) for p in planets_raw], dtype=jnp.float32)
    planets_y = jnp.array([float(p[3]) for p in planets_raw], dtype=jnp.float32)
    planets_radius = jnp.array([float(p[4]) for p in planets_raw], dtype=jnp.float32)
    planets_alive = jnp.ones(len(planets_raw), dtype=bool)

    jax_target_idx, jax_eta = fleet_target_planet_batch(
        fleets_x, fleets_y, fleets_angle, fleets_ships, fleets_alive,
        planets_x, planets_y, planets_radius, planets_alive,
        max_horizon=DEFAULT_HORIZON,
    )
    jax_target_arr = np.asarray(jax_target_idx)
    jax_eta_arr = np.asarray(jax_eta)

    # JAX returns SLOT index in the local-build planets array; map
    # to planet ids via the same input ordering used to build the arrays.
    planet_ids = [int(p[0]) for p in planets_raw]
    for i, f in enumerate(fleet_named):
        scalar_t = scalar_targets[i]
        scalar_e = scalar_etas[i]
        jax_idx = int(jax_target_arr[i])
        jax_e = int(jax_eta_arr[i])

        if scalar_t is None:
            assert jax_idx == -1, (
                f"fleet[{i}] scalar=no-hit but jax={planet_ids[jax_idx] if jax_idx >= 0 else 'no-hit'}"
            )
            continue
        assert jax_idx >= 0, (
            f"fleet[{i}] scalar hit pid={scalar_t} but jax=no-hit"
        )
        jax_pid = planet_ids[jax_idx]
        assert jax_pid == scalar_t, (
            f"fleet[{i}] target diverges: scalar pid={scalar_t}, jax pid={jax_pid}"
        )
        # ETA: tolerance ±1 (scalar uses int(ceil(turns)); float32 vs
        # float64 may flip the ceil at boundary cases). Most cases match exactly.
        assert abs(jax_e - scalar_e) <= 1, (
            f"fleet[{i}] eta diverges: scalar={scalar_e}, jax={jax_e}"
        )


def test_fleet_speed_batch_parity():
    """Vectorised fleet_speed matches scalar formula for a battery of
    ship counts."""
    from lib.fleet import speed as scalar_speed
    ships = jnp.array([1, 5, 10, 25, 50, 100, 500, 999, 1000], dtype=jnp.int32)
    jax_speeds = np.asarray(fleet_speed_batch(ships))
    for i, n in enumerate(np.asarray(ships)):
        expected = float(scalar_speed(int(n)))
        assert abs(float(jax_speeds[i]) - expected) < 1e-4, (
            f"ships={n}: jax={jax_speeds[i]}, scalar={expected}"
        )


# ---------------------------------------------------------------------------
# Sub-phase 2b: build_arrival_grid parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_build_arrival_grid_parity(seed):
    """JAX arrival grid (P, H+1, A) matches scalar `build_arrival_ledger`
    when aggregated by owner per (planet, step). Tests post-mid-game
    state with multiple in-flight fleets."""
    from lib.world_model import build_arrival_ledger, DEFAULT_HORIZON
    from lib.game.jax.jax_world_model import build_arrival_grid
    from lib.game.jax import scalar_to_jax

    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _spawn_in_flight_fleets(env, num_agents=2, n_steps=20, rng_seed=seed * 7 + 1)
    fleets_raw = list(env.state[0].observation.fleets)
    if not fleets_raw:
        pytest.skip(f"seed {seed} produced no in-flight fleets")

    # Build the JAX GameState + arrival grid.
    gs = scalar_to_jax(env.state, env.info["seed"])
    H = DEFAULT_HORIZON
    A = 4
    jax_grid = np.asarray(build_arrival_grid(gs, max_horizon=H, num_agents=A))
    # shape: (MAX_PLANETS, H+1, A)

    # Scalar reference ledger.
    fleet_named = [Fleet(*f) for f in fleets_raw]
    planet_named = [Planet(*p) for p in env.state[0].observation.planets]
    scalar_ledger = build_arrival_ledger(fleet_named, planet_named, H)

    # Compare: for each (planet_id, eta, owner) in the scalar dict,
    # find the corresponding (planet_slot, eta, owner) JAX cell.
    pid_to_slot = {int(pid): slot for slot, pid in enumerate(np.asarray(gs.planets_id)) if pid >= 0}

    # Build a parallel (P, H+1, A) grid from the scalar ledger.
    P_max = jax_grid.shape[0]
    scalar_grid = np.zeros_like(jax_grid)
    for planet_id, arrivals in scalar_ledger.items():
        slot = pid_to_slot.get(int(planet_id))
        if slot is None:
            continue
        for eta, owner, ships in arrivals:
            if ships <= 0:
                continue
            bucket = max(1, int(math.ceil(eta)))
            if bucket > H:
                continue
            scalar_grid[slot, bucket, int(owner)] += int(ships)

    # Per-cell comparison.
    diff_count = 0
    for p in range(P_max):
        if not bool(gs.planets_alive[p]):
            continue
        for t in range(H + 1):
            for a in range(A):
                if scalar_grid[p, t, a] != jax_grid[p, t, a]:
                    diff_count += 1
                    if diff_count <= 3:
                        print(
                            f"  diff at slot={p} (pid={int(gs.planets_id[p])}) "
                            f"step={t} owner={a}: scalar={scalar_grid[p, t, a]} "
                            f"jax={jax_grid[p, t, a]}"
                        )
    assert diff_count == 0, f"{diff_count} cells differ between scalar/JAX grid"


# ---------------------------------------------------------------------------
# Sub-phase 2c: simulate_planet_timeline parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_simulate_all_timelines_parity(seed):
    """Per-planet timelines (owner_at, ships_at over [0, horizon])
    match scalar `simulate_planet_timeline` for every alive planet."""
    from lib.world_model import (
        build_arrival_ledger, simulate_planet_timeline, DEFAULT_HORIZON,
    )
    from lib.game.jax.jax_world_model import (
        build_arrival_grid, simulate_all_timelines,
    )
    from lib.game.jax import scalar_to_jax

    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _spawn_in_flight_fleets(env, num_agents=2, n_steps=20, rng_seed=seed * 11 + 3)

    fleets_raw = list(env.state[0].observation.fleets)
    if not fleets_raw:
        pytest.skip("no in-flight fleets")

    gs = scalar_to_jax(env.state, env.info["seed"])
    H = DEFAULT_HORIZON
    arrival_grid = build_arrival_grid(gs, max_horizon=H, num_agents=4)
    owners_jax, ships_jax = simulate_all_timelines(gs, arrival_grid, max_horizon=H)
    owners_jax = np.asarray(owners_jax)
    ships_jax = np.asarray(ships_jax)

    # Scalar reference.
    fleet_named = [Fleet(*f) for f in fleets_raw]
    planet_named = [Planet(*p) for p in env.state[0].observation.planets]
    ledger = build_arrival_ledger(fleet_named, planet_named, H)

    pid_to_slot = {
        int(pid): slot
        for slot, pid in enumerate(np.asarray(gs.planets_id))
        if pid >= 0
    }
    diffs = []
    for p in planet_named:
        scalar_tl = simulate_planet_timeline(p, ledger[p.id], H)
        slot = pid_to_slot[int(p.id)]
        # Sample a handful of steps (0, 50, 100, 250) for comparison.
        for t in (0, 1, 10, 50, 100, 150, 250):
            so = scalar_tl["owner_at"][t]
            ss = int(scalar_tl["ships_at"][t])
            jo = int(owners_jax[slot, t])
            js = int(ships_jax[slot, t])
            if so != jo or ss != js:
                diffs.append(
                    f"  pid={p.id} step={t}: scalar=({so}, {ss}) jax=({jo}, {js})"
                )
            if len(diffs) >= 5:
                break
        if len(diffs) >= 5:
            break
    assert not diffs, "timeline divergence:\n" + "\n".join(diffs)
