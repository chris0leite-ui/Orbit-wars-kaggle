"""Sub-phase 3 parity: JAX missions vs scalar mission builders.

Currently tests:
- `compute_snipe_score_matrix` (base, non-aggressive) vs scalar
  `lib.missions.snipe.propose_snipe_missions(aggressive=False)`.

Future sub-phases add aggressive sizing (3b), leader-spoiler + comet
bonus (3c), reinforce + recapture (3d), and `settle_plan` (3e).
"""

from __future__ import annotations

import math
import random

import jax.numpy as jnp
import numpy as np
import pytest

from kaggle_environments import make

from lib.intent import World
from lib.world_model import WorldModel
from lib.missions.snipe import propose_snipe_missions

from lib.game.jax import scalar_to_jax
from lib.game.jax.jax_world_model import build_world_model, DEFAULT_HORIZON
from lib.game.jax.jax_missions import compute_snipe_score_matrix


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


def _check_snipe_parity(env, my_id: int, aggressive: bool):
    """Run scalar `propose_snipe_missions` and JAX `compute_snipe_score_matrix`
    on the same state, return (diffs, extra) lists.

    Comet targets are skipped on the scalar side — JAX base form uses
    non-comet time_to_hold; the comet-lifetime correction lands in 3c.
    """
    obs = env.state[my_id].observation
    scalar_world = World.from_obs(obs)
    scalar_wm = WorldModel.from_world(scalar_world)
    scalar_missions = propose_snipe_missions(
        scalar_world, scalar_wm, aggressive=aggressive,
    )

    gs = scalar_to_jax(env.state, env.info["seed"])
    jax_wm = build_world_model(gs, max_horizon=DEFAULT_HORIZON, num_agents=4)
    out = compute_snipe_score_matrix(
        gs, jax_wm, my_id=my_id, aggressive=aggressive,
    )
    score = np.asarray(out["score"])
    ships = np.asarray(out["ships"])
    eta = np.asarray(out["eta"])
    valid = np.asarray(out["valid"])

    pid_to_slot = {
        int(pid): slot
        for slot, pid in enumerate(np.asarray(gs.planets_id))
        if pid >= 0
    }

    diffs = []
    matched_pairs = set()
    for m in scalar_missions:
        if m.target_id in scalar_world.comet_ids:
            continue
        if m.src_id not in pid_to_slot or m.target_id not in pid_to_slot:
            continue
        s_slot = pid_to_slot[m.src_id]
        t_slot = pid_to_slot[m.target_id]
        matched_pairs.add((s_slot, t_slot))

        if not bool(valid[s_slot, t_slot]):
            diffs.append(
                f"  pid={m.src_id}->{m.target_id}: scalar has mission "
                f"(score={m.score:.4f}) but JAX valid=False"
            )
            continue
        if int(ships[s_slot, t_slot]) != int(m.ships):
            diffs.append(
                f"  pid={m.src_id}->{m.target_id}: ships scalar={m.ships} "
                f"jax={int(ships[s_slot, t_slot])}"
            )
        if abs(int(eta[s_slot, t_slot]) - int(m.eta)) > 1:
            diffs.append(
                f"  pid={m.src_id}->{m.target_id}: eta scalar={m.eta} "
                f"jax={int(eta[s_slot, t_slot])}"
            )
        jax_score = float(score[s_slot, t_slot])
        rel = abs(jax_score - m.score) / max(abs(m.score), 1e-6)
        if rel > 1e-3 and abs(jax_score - m.score) > 1e-3:
            diffs.append(
                f"  pid={m.src_id}->{m.target_id}: score scalar={m.score:.6f} "
                f"jax={jax_score:.6f} rel={rel:.4e}"
            )
        if len(diffs) >= 6:
            break

    comet_slots = {pid_to_slot[c] for c in scalar_world.comet_ids if c in pid_to_slot}
    P = score.shape[0]
    extra = []
    for s in range(P):
        for t in range(P):
            if not bool(valid[s, t]):
                continue
            if (s, t) in matched_pairs:
                continue
            if t in comet_slots:
                continue
            extra.append(
                f"  src_slot={s} (pid={int(gs.planets_id[s])}) "
                f"-> tgt_slot={t} (pid={int(gs.planets_id[t])}): "
                f"JAX valid, scalar absent. score={float(score[s, t]):.4f}"
            )
            if len(extra) >= 3:
                break
        if len(extra) >= 3:
            break

    return diffs, extra


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_compute_snipe_score_matrix_parity(seed):
    """Sub-phase 3a: JAX snipe score matrix matches scalar
    `propose_snipe_missions(aggressive=False)` per (src, target) pair.

    Scope: 2-player games (LEADER_MULTIPLIER never fires); base form.
    """
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _spawn_in_flight_fleets(env, num_agents=2, n_steps=25, rng_seed=seed * 13 + 1)
    if env.state[0].status != "ACTIVE":
        pytest.skip(f"seed {seed} terminated before mid-game")

    diffs, extra = _check_snipe_parity(env, my_id=0, aggressive=False)
    assert not diffs, "snipe score matrix divergence:\n" + "\n".join(diffs)
    assert not extra, "JAX snipe matrix has extra valid pairs:\n" + "\n".join(extra)


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_compute_snipe_score_matrix_aggressive_parity(seed):
    """Sub-phase 3b: aggressive=True (top-10 fraction sizing) matches scalar.

    Verifies src-conditioned base_ships:
      - garrison ≤ AGGRESSIVE_MIN_GARRISON → falls back to target_min
      - else → max(target_min, min(int(src.ships * 0.7), src.ships - 5))
    """
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _spawn_in_flight_fleets(env, num_agents=2, n_steps=25, rng_seed=seed * 17 + 5)
    if env.state[0].status != "ACTIVE":
        pytest.skip(f"seed {seed} terminated before mid-game")

    diffs, extra = _check_snipe_parity(env, my_id=0, aggressive=True)
    assert not diffs, "aggressive snipe matrix divergence:\n" + "\n".join(diffs)
    assert not extra, "JAX aggressive matrix has extra valid pairs:\n" + "\n".join(extra)
