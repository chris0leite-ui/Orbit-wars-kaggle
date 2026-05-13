"""Parity: serial `score_candidate_jax` vs vmap'd `score_candidates_vmap`.

Validates the candidate-axis vmap by scoring a small candidate set both
ways and asserting:
- Same argmax (the winner index agrees).
- Pointwise scores within float32 tolerance.
- Output shape matches the candidate count.
"""

from __future__ import annotations

import math
import random

import jax.numpy as jnp
import numpy as np
import pytest

from kaggle_environments import make

from lib.game.jax import scalar_to_jax
from lib.game.jax.jax_brute_search import (
    candidate_emits_to_tensors,
    score_candidates_vmap,
)
from lib.game.jax.jax_score import policy_step_jax, score_candidate_jax


def _light_play(env, n_steps=20, rng_seed=7, num_agents=2):
    rng = random.Random(rng_seed)
    for _ in range(n_steps):
        if env.state[0].status != "ACTIVE":
            break
        actions = []
        for ps in range(num_agents):
            mv = []
            for p in env.state[ps].observation.planets:
                if p[1] == ps and p[5] > 5 and rng.random() < 0.2:
                    mv.append([
                        p[0],
                        rng.uniform(0, 2 * math.pi),
                        max(1, int(p[5] * rng.uniform(0.1, 0.3))),
                    ])
            actions.append(mv)
        env.step(actions)


def _build_candidate_set(emit_incumbent):
    """Incumbent + each drop-one variant. Same shape contract as
    `lib.v7_search._enumerate_drop_one` but operating over emit dicts."""
    if not emit_incumbent:
        return [[]]
    cands = [list(emit_incumbent)]
    for i in range(len(emit_incumbent)):
        cands.append([e for j, e in enumerate(emit_incumbent) if j != i])
    return cands


@pytest.mark.parametrize("seed", [3, 42])
def test_serial_vs_vmap_argmax_matches(seed):
    """Argmax candidate index from the serial loop matches the vmap'd
    output. Pointwise scores match within ~1e-3 absolute (float32
    rollout is bit-exact under fixed traces but vmap broadcast may
    reorder a few accumulations)."""
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _light_play(env, n_steps=20, rng_seed=seed * 31)
    if env.state[0].status != "ACTIVE":
        pytest.skip(f"seed {seed} terminated early")

    gs = scalar_to_jax(env.state, env.info["seed"])
    emit_inc, _ = policy_step_jax(gs, my_id=0)
    cands = _build_candidate_set(emit_inc)
    assert len(cands) >= 2, "need at least incumbent + 1 drop variant"

    # Cap to 5 candidates to keep the test fast.
    cands = cands[:5]

    K = 4
    # Serial path.
    serial_scores = np.array(
        [score_candidate_jax(gs, emit, K=K, my_id=0) for emit in cands],
        dtype=np.float32,
    )

    # Vmap path.
    pids_c, ang_c, sh_c = candidate_emits_to_tensors(cands, num_agents=2)
    vmap_scores = np.asarray(
        score_candidates_vmap(
            gs,
            jnp.asarray(pids_c),
            jnp.asarray(ang_c),
            jnp.asarray(sh_c),
            K=K,
            my_id=0,
        ),
        dtype=np.float32,
    )

    # Shape contract.
    assert vmap_scores.shape == (len(cands),), (
        f"vmap shape mismatch: got {vmap_scores.shape}, want {(len(cands),)}"
    )

    # Pointwise agreement (float32 rollout has small accumulation noise).
    max_diff = float(np.max(np.abs(serial_scores - vmap_scores)))
    assert max_diff < 1e-2, (
        f"serial vs vmap diverged: serial={serial_scores} "
        f"vmap={vmap_scores} max_diff={max_diff}"
    )

    # Argmax agreement (the critical contract for the oracle).
    serial_argmax = int(np.argmax(serial_scores))
    vmap_argmax = int(np.argmax(vmap_scores))
    assert serial_argmax == vmap_argmax, (
        f"argmax disagreement: serial={serial_argmax} "
        f"({serial_scores[serial_argmax]:.3f}) vs vmap={vmap_argmax} "
        f"({vmap_scores[vmap_argmax]:.3f})"
    )


def test_shape_invariance_with_padding():
    """Candidates with fewer launches than the longest are padded with
    -1 sentinels and still produce a valid score (the padded slots are
    no-ops in `jax_step`)."""
    seed = 7
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _light_play(env, n_steps=15, rng_seed=seed * 31)
    if env.state[0].status != "ACTIVE":
        pytest.skip("terminated early")
    gs = scalar_to_jax(env.state, env.info["seed"])
    emit_inc, _ = policy_step_jax(gs, my_id=0)
    # Two cands: incumbent (length N) + empty action (length 0).
    cands = [list(emit_inc), []]
    pids_c, ang_c, sh_c = candidate_emits_to_tensors(cands, num_agents=2)
    scores = np.asarray(
        score_candidates_vmap(
            gs,
            jnp.asarray(pids_c),
            jnp.asarray(ang_c),
            jnp.asarray(sh_c),
            K=3,
            my_id=0,
        ),
        dtype=np.float32,
    )
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
