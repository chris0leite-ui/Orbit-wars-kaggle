"""Numpy inference mirror ↔ JAX training path parity.

If these drift, the submitted agent is not the agent we trained.
Tolerances are loose-ish (fp32 vs fp64) but catch real bugs.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lib.game.jax.jax_interpreter import jax_step
from lib.game.jax.jax_types import (
    MAX_AGENTS, MAX_COMET_PATH_LEN, MAX_LAUNCH_PER_AGENT, MAX_PLANETS,
)
from rl import net, numpy_infer as ni
from rl.features import seat_features, state_tables
from rl.make_pool import load_pool

POOL = "data/rl_pool_smoke.npz"


def _state_with_traffic(i=0, n_steps=60, seed=3):
    """Roll a pool state forward with random launches so fleets and
    comets are in play."""
    pool = load_pool(POOL)
    gs = jax.tree.map(lambda x: jnp.asarray(x[i]), pool)
    rng = np.random.default_rng(seed)
    for k in range(n_steps):
        pids = -np.ones((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), np.int32)
        angles = rng.uniform(0, 2 * np.pi,
                             (MAX_AGENTS, MAX_LAUNCH_PER_AGENT)).astype(np.float32)
        ships = np.zeros((MAX_AGENTS, MAX_LAUNCH_PER_AGENT), np.int32)
        owner = np.asarray(gs.planets_owner)
        alive = np.asarray(gs.planets_alive)
        gships = np.asarray(gs.planets_ships)
        for a in range(2):
            mine = np.where((owner == a) & alive & (gships > 4))[0]
            if len(mine) and k % 3 == 0:
                src = rng.choice(mine)
                pids[a, 0] = int(np.asarray(gs.planets_id)[src])
                ships[a, 0] = int(gships[src] // 3)
        gs = jax_step(gs, jnp.asarray(pids), jnp.asarray(angles),
                      jnp.asarray(ships))
    return gs


def gamestate_to_arrays(gs):
    """Build the numpy-side `a` dict directly from a GameState."""
    P = MAX_PLANETS
    a = {
        "x": np.asarray(gs.planets_x, np.float64),
        "y": np.asarray(gs.planets_y, np.float64),
        "pid": np.asarray(gs.planets_id, np.int64),
        "owner": np.asarray(gs.planets_owner, np.int64),
        "ships": np.asarray(gs.planets_ships, np.float64),
        "prod": np.asarray(gs.planets_prod, np.float64),
        "radius": np.asarray(gs.planets_radius, np.float64),
        "alive": np.asarray(gs.planets_alive),
        "ix": np.asarray(gs.initial_x, np.float64),
        "iy": np.asarray(gs.initial_y, np.float64),
        "is_comet": np.asarray(gs.is_comet),
        "step": int(gs.step),
        "omega": float(gs.angular_velocity),
    }
    # comet tables
    remain = np.full(P, 1e6)
    TT = ni.T_HORIZON + 61
    comet_pos = np.zeros((P, TT, 2))
    spawn = np.asarray(gs.planet_comet_spawn)
    path = np.asarray(gs.planet_comet_path)
    cpi = np.asarray(gs.comet_path_index)
    plen = np.asarray(gs.comet_paths_len)
    pxy = np.asarray(gs.comet_paths_xy, np.float64)
    for i in range(P):
        if spawn[i] >= 0 and a["alive"][i]:
            k, j = spawn[i], path[i]
            idx = cpi[k]
            L = plen[k, j]
            remain[i] = max(L - 1 - idx, 0)
            for t in range(TT):
                kk = min(max(idx + t, 0), L - 1)
                comet_pos[i, t] = pxy[k, j, kk]
    a["comet_remain"] = remain
    a["comet_pos"] = comet_pos

    a["f_x"] = np.asarray(gs.fleets_x, np.float64)
    a["f_y"] = np.asarray(gs.fleets_y, np.float64)
    a["f_angle"] = np.asarray(gs.fleets_angle, np.float64)
    a["f_owner"] = np.asarray(gs.fleets_owner, np.int64)
    a["f_ships"] = np.asarray(gs.fleets_ships, np.float64)
    a["f_alive"] = np.asarray(gs.fleets_alive)
    return a


@pytest.mark.parametrize("pool_i,steps", [(0, 0), (0, 60), (3, 75)])
def test_feature_parity(pool_i, steps):
    gs = (_state_with_traffic(pool_i, steps) if steps
          else jax.tree.map(lambda x: jnp.asarray(x[pool_i]), load_pool(POOL)))
    seat = 0
    tables = state_tables(gs)
    jn, je, jg, jsm, jtm = seat_features(gs, tables, seat)
    a = gamestate_to_arrays(gs)
    nn, ne, ng, nsm, ntm = ni.seat_features(a, seat, int(gs.num_agents))

    np.testing.assert_allclose(np.asarray(jn), nn, atol=2e-3,
                               err_msg="nodes")
    np.testing.assert_allclose(np.asarray(jg), ng, atol=2e-3,
                               err_msg="globals")
    assert (np.asarray(jsm) == nsm).all(), "src_mask"
    # Edge features: compare only rows/cols of alive planets; eta drift
    # near bucket boundaries can flip discrete features slightly.
    alive = np.asarray(gs.planets_alive)
    j_e = np.asarray(je)[alive][:, alive, :]
    n_e = ne[alive][:, alive, :]
    mismatch = np.abs(j_e - n_e) > 5e-3
    frac_bad = mismatch.mean()
    assert frac_bad < 0.01, f"edge mismatch fraction {frac_bad}"
    # Target masks: allow tiny disagreement from eta drift on borderline
    # sun/oob pairs.
    tm_diff = (np.asarray(jtm)[alive] != ntm[alive]).mean()
    assert tm_diff < 0.01, f"tgt_mask diff {tm_diff}"


def test_solve_intercept_rows_parity():
    """Launch-angle path numpy vs JAX — the function whose numpy port
    silently broke the entire exported agent (IndexError swallowed by
    the agent() wrapper -> never launched a fleet in any real game)."""
    from rl.aim import solve_intercept_rows as jax_rows

    gs = _state_with_traffic(0, 60)
    a = gamestate_to_arrays(gs)
    rng = np.random.default_rng(0)
    alive_idx = np.where(np.asarray(gs.planets_alive))[0]
    # Random targets (include duplicates — the old scatter bug clobbered
    # shared targets) and random ship counts.
    tgt = rng.choice(alive_idx, size=MAX_PLANETS, replace=True).astype(np.int64)
    # Self-targets are masked out of the action space; arctan2 of a
    # ~zero delta is float-noise-arbitrary, so exclude them here.
    self_rows = tgt == np.arange(MAX_PLANETS)
    tgt[self_rows] = alive_idx[0] if alive_idx[0] != 0 else alive_idx[1]
    tgt[tgt == np.arange(MAX_PLANETS)] = alive_idx[-1]
    ships = rng.integers(1, 200, MAX_PLANETS)

    j_angle = np.asarray(jax_rows(gs, jnp.asarray(tgt, jnp.int32),
                                  jnp.asarray(ships, jnp.int32)))
    n_angle, _ = ni.solve_intercept_rows(a, tgt, ships)

    alive = np.asarray(gs.planets_alive)
    diff = np.abs(np.angle(np.exp(1j * (j_angle - n_angle))))
    assert diff[alive].max() < 1e-3, f"angle drift {diff[alive].max()}"


def test_exported_agent_launches_in_real_game():
    """Behavioral gate: a trained/seeded export must EMIT launch actions
    in a real kaggle-env game (reproduces the night-1 silent-pacifist
    failure mode end-to-end)."""
    import subprocess, sys as _sys
    from kaggle_environments import make
    from rl import export_agent

    # Use any available checkpoint; fall back to random-init params.
    import glob as _glob, pickle, tempfile, os
    cands = (_glob.glob("/tmp/kernel_overnight/ckpt_final.pkl")
             + _glob.glob("/tmp/rl_smoke/ckpt_final.pkl"))
    with tempfile.TemporaryDirectory() as td:
        if not cands:
            import jax as _jax
            from rl import net as _net
            params = _jax.tree.map(np.asarray,
                                   _net.init_params(_jax.random.PRNGKey(0)))
            ck = os.path.join(td, "ck.pkl")
            with open(ck, "wb") as f:
                pickle.dump({"params": params, "meta": {}}, f)
            cands = [ck]
        out = os.path.join(td, "agent.py")
        export_agent.export(cands[0], out)

        env = make("orbit_wars", configuration={"seed": 77,
                                                "episodeSteps": 60})
        env.run([out, "random"])
        max_fleets = 0
        for st in env.steps:
            obs = st[0].observation
            fl = sum(1 for f in (obs.get("fleets") or []) if f[1] == 0)
            max_fleets = max(max_fleets, fl)
        assert max_fleets > 0, (
            "exported agent never launched a fleet — act() is likely "
            "throwing and being swallowed by the agent() wrapper")


def test_net_forward_parity():
    gs = _state_with_traffic(0, 60)
    seat = 0
    tables = state_tables(gs)
    jn, je, jg, jsm, jtm = seat_features(gs, tables, seat)
    params = net.init_params(jax.random.PRNGKey(7))
    jv, jl, jemb = net.forward(params, jn, je, jg, gs.planets_alive, jtm)

    p_np = {k: np.asarray(v, np.float64) for k, v in params.items()}
    nv, nl, nemb = ni.forward(p_np, np.asarray(jn, np.float64),
                              np.asarray(je, np.float64),
                              np.asarray(jg, np.float64),
                              np.asarray(gs.planets_alive),
                              np.asarray(jtm))
    assert abs(float(jv) - nv) < 1e-3
    alive = np.asarray(gs.planets_alive)
    jl_np = np.asarray(jl)[alive]
    nl_np = nl[alive]
    legal = np.asarray(jtm)[alive]
    np.testing.assert_allclose(jl_np[legal], nl_np[legal], atol=2e-3)

    # Fraction head parity on argmax targets.
    tgt = np.argmax(nl_np, axis=-1)
    full_tgt = np.zeros(MAX_PLANETS, np.int64)
    full_tgt[np.where(alive)[0]] = tgt
    jf = net.frac_logits_for(params, jemb, je, jnp.asarray(full_tgt))
    nf = ni.frac_logits_for(p_np, nemb, np.asarray(je, np.float64), full_tgt)
    np.testing.assert_allclose(np.asarray(jf)[alive], nf[alive], atol=2e-3)
