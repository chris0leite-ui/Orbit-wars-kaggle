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


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_compute_snipe_score_matrix_parity(seed):
    """JAX snipe score matrix matches scalar `propose_snipe_missions`
    (non-aggressive base form) per (src, target) pair.

    Scope (sub-phase 3a):
    - 2-player games (LEADER_MULTIPLIER never fires).
    - aggressive=False (matches JAX base form).
    - Comet targets are skipped — JAX uses non-comet time_to_hold; the
      comet-lifetime correction lands in sub-phase 3c.
    """
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _spawn_in_flight_fleets(env, num_agents=2, n_steps=25, rng_seed=seed * 13 + 1)
    if env.state[0].status != "ACTIVE":
        pytest.skip(f"seed {seed} terminated before mid-game")

    my_id = 0
    obs = env.state[my_id].observation
    if not any(p[1] == my_id for p in obs.planets):
        pytest.skip(f"seed {seed}: agent {my_id} owns no planets")
    if not any(p[1] != my_id and p[1] != -1 or p[1] == -1 for p in obs.planets):
        pytest.skip(f"seed {seed}: no non-self planets")

    # Scalar reference: build World + WorldModel + missions.
    scalar_world = World.from_obs(obs)
    scalar_wm = WorldModel.from_world(scalar_world)
    scalar_missions = propose_snipe_missions(scalar_world, scalar_wm, aggressive=False)

    # JAX equivalent.
    gs = scalar_to_jax(env.state, env.info["seed"])
    jax_wm = build_world_model(gs, max_horizon=DEFAULT_HORIZON, num_agents=4)
    out = compute_snipe_score_matrix(gs, jax_wm, my_id=my_id)
    score = np.asarray(out["score"])
    ships = np.asarray(out["ships"])
    eta = np.asarray(out["eta"])
    valid = np.asarray(out["valid"])

    # planet_id ↔ JAX slot mapping.
    pid_to_slot = {
        int(pid): slot
        for slot, pid in enumerate(np.asarray(gs.planets_id))
        if pid >= 0
    }

    # Verify every scalar non-comet mission has a matching JAX cell.
    diffs = []
    matched_pairs = set()
    for m in scalar_missions:
        if m.target_id in scalar_world.comet_ids:
            continue  # JAX base form doesn't handle comet lifetime (3c).
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
        # Ships exact.
        if int(ships[s_slot, t_slot]) != int(m.ships):
            diffs.append(
                f"  pid={m.src_id}->{m.target_id}: ships scalar={m.ships} "
                f"jax={int(ships[s_slot, t_slot])}"
            )
        # ETA: tolerance ±1 (float32 vs float64 boundary on ceil).
        if abs(int(eta[s_slot, t_slot]) - int(m.eta)) > 1:
            diffs.append(
                f"  pid={m.src_id}->{m.target_id}: eta scalar={m.eta} "
                f"jax={int(eta[s_slot, t_slot])}"
            )
        # Score: float32 vs float64 → relative tolerance 1e-3.
        jax_score = float(score[s_slot, t_slot])
        rel = abs(jax_score - m.score) / max(abs(m.score), 1e-6)
        if rel > 1e-3 and abs(jax_score - m.score) > 1e-3:
            diffs.append(
                f"  pid={m.src_id}->{m.target_id}: score scalar={m.score:.6f} "
                f"jax={jax_score:.6f} rel={rel:.4e}"
            )
        if len(diffs) >= 6:
            break

    # Verify JAX cells flagged valid correspond to scalar missions
    # (modulo comet targets, which we'd allow JAX to also score).
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
                continue  # JAX may legitimately keep comets; scalar excluded.
            # JAX valid but scalar didn't produce mission. Allowed cause:
            # neither the redundancy filter nor the affordability filter
            # would have dropped it scalar-side — so this is a real diff.
            extra.append(
                f"  src_slot={s} (pid={int(gs.planets_id[s])}) "
                f"-> tgt_slot={t} (pid={int(gs.planets_id[t])}): "
                f"JAX valid, scalar absent. score={float(score[s, t]):.4f}"
            )
            if len(extra) >= 3:
                break
        if len(extra) >= 3:
            break

    assert not diffs, "snipe score matrix divergence:\n" + "\n".join(diffs)
    assert not extra, "JAX snipe matrix has extra valid pairs:\n" + "\n".join(extra)
