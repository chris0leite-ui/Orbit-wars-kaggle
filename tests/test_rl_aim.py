"""RL aim/predictor correctness vs the JAX engine.

These tests gate the entire RL action pipeline: if the orbit predictor
or the intercept solver drifts from the engine, every learned launch is
garbage.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lib.game.jax.jax_interpreter import jax_step
from lib.game.jax.jax_types import MAX_AGENTS, MAX_LAUNCH_PER_AGENT, MAX_PLANETS
from rl.aim import planet_pos_at, solve_intercept_rows
from rl.make_pool import load_pool

POOL = "data/rl_pool_smoke.npz"


def _no_actions():
    return (
        -jnp.ones((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), jnp.int32),
        jnp.zeros((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), jnp.float32),
        jnp.zeros((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), jnp.int32),
    )


def _state(i=0):
    pool = load_pool(POOL)
    return jax.tree.map(lambda x: jnp.asarray(x[i]), pool)


@pytest.mark.parametrize("k", [1, 5, 20, 45])
def test_orbit_predictor_matches_engine(k):
    gs = _state(0)
    pred = planet_pos_at(gs, jnp.float32(k))
    pids, angles, ships = _no_actions()
    s = gs
    for _ in range(k):
        s = jax_step(s, pids, angles, ships)
    alive = np.asarray(s.planets_alive)
    actual = np.stack([np.asarray(s.planets_x), np.asarray(s.planets_y)], -1)
    err = np.linalg.norm(np.asarray(pred) - actual, axis=-1)
    assert err[alive].max() < 1e-3, f"k={k} max err {err[alive].max()}"


def test_intercept_hits_intended_target():
    """Launch at solver angle; the fleet must reach the chosen planet
    and its garrison must change accordingly (neutral target)."""
    gs = _state(0)
    owner = np.asarray(gs.planets_owner)
    alive = np.asarray(gs.planets_alive)
    ships0 = np.asarray(gs.planets_ships)

    my = np.where((owner == 0) & alive)[0]
    neutral = np.where((owner == -1) & alive)[0]
    assert len(my) >= 1 and len(neutral) >= 3

    hits = 0
    tried = 0
    for tgt in neutral[:6]:
        src = int(my[0])
        send = max(2, int(ships0[src]) // 2)
        tgt_idx = jnp.full((MAX_PLANETS,), int(tgt), jnp.int32)
        ships_vec = jnp.full((MAX_PLANETS,), send, jnp.int32)
        angle = solve_intercept_rows(gs, tgt_idx, ships_vec)[src]

        pids = -np.ones((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), np.int32)
        angs = np.zeros((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), np.float32)
        shp = np.zeros((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), np.int32)
        pids[0, 0] = int(np.asarray(gs.planets_id)[src])
        angs[0, 0] = float(angle)
        shp[0, 0] = send

        s = jax_step(gs, jnp.asarray(pids), jnp.asarray(angs), jnp.asarray(shp))
        na = _no_actions()
        arrived = False
        for _ in range(70):
            s = jax_step(s, *na)
            tgt_ships = int(np.asarray(s.planets_ships)[tgt])
            tgt_owner = int(np.asarray(s.planets_owner)[tgt])
            fleet_gone = not bool(np.asarray(s.fleets_alive).any())
            if fleet_gone:
                # Neutral garrison must have been reduced by `send` or
                # flipped to us (attacker > garrison).
                expected_drop = ships0[tgt] - send
                arrived = (
                    (tgt_owner == 0)
                    or (tgt_ships <= max(expected_drop, 0) + 1)
                )
                break
        tried += 1
        if arrived:
            hits += 1

    assert tried >= 4
    # Lead-aim should land the overwhelming majority of shots.
    assert hits / tried >= 0.8, f"only {hits}/{tried} intercepts landed"
