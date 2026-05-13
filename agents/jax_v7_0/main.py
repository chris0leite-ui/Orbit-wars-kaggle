"""JAX-backed v7.0 drop-one agent.

Mirror of `agents/v7_ablations/v7_0_drop_one/main.py` but running on the
JAX engine + missions + rollout. Live-ladder use is OFFLINE-ONLY: this
agent is for the local + Kaggle-kernel A/B harness; the production
submission keeps the pure-Python v7_0 bundle.

Pipeline per call:
1. obs → `GameState` via `scalar_to_jax`.
2. Build incumbent emit via `policy_step_jax(state, my_id)`.
3. Drop-one candidates: incumbent + (incumbent minus i) for each i.
4. Score each via `score_candidate_jax(K=10, my_id)`.
5. Pick the highest-scoring emit; format as
   `[[src_id, angle, ships], ...]` for kaggle_environments.

Wall-clock watchdog (`wallclock_ms`) bails early to the incumbent if
the candidate sweep is going to exceed budget.
"""

from __future__ import annotations

import time

from lib.game.jax import scalar_to_jax
from lib.game.jax.jax_score import policy_step_jax, score_candidate_jax


DEFAULT_K = 10
DEFAULT_WALLCLOCK_MS = 700.0


def _emit_to_actions(emit):
    """Convert apply_mechanisms_numpy output → kaggle env action list."""
    return [[int(e["src_pid"]), float(e["angle"]), int(e["ships"])] for e in emit]


def _state_from_obs_only(obs, configuration):
    """Build a one-seat GameState from a single obs dict.

    `scalar_to_jax` expects `env.state` (a list of per-seat states with
    rewards/status). When called from the kaggle agent harness we only
    have `obs`. We synthesise a minimal-shape state list.
    """
    import types
    fake_state = types.SimpleNamespace(observation=obs, reward=0)
    # Episode seed lives in configuration when available; otherwise 0.
    seed = 0
    if configuration is not None:
        raw_seed = (
            configuration.get("seed", 0)
            if hasattr(configuration, "get")
            else getattr(configuration, "seed", 0)
        )
        seed = int(raw_seed) if raw_seed is not None else 0
    return scalar_to_jax([fake_state], episode_seed=seed)


def agent(obs, configuration=None, K=DEFAULT_K, wallclock_ms=DEFAULT_WALLCLOCK_MS):
    t_start = time.perf_counter()
    my_id = obs.get("player", 0) if isinstance(obs, dict) else obs.player

    state = _state_from_obs_only(obs, configuration)

    # Incumbent.
    incumbent_emit, _ = policy_step_jax(state, my_id=my_id)
    if not incumbent_emit:
        return []

    # Candidates: incumbent + each drop-one variant.
    candidates = [incumbent_emit]
    for i in range(len(incumbent_emit)):
        variant = [e for j, e in enumerate(incumbent_emit) if j != i]
        candidates.append(variant)

    # Score each within wallclock budget.
    best_emit = incumbent_emit
    best_score = score_candidate_jax(state, incumbent_emit, K=K, my_id=my_id)
    for variant in candidates[1:]:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if elapsed_ms > wallclock_ms:
            break
        try:
            val = score_candidate_jax(state, variant, K=K, my_id=my_id)
        except Exception:
            continue
        if val > best_score:
            best_score = val
            best_emit = variant

    return _emit_to_actions(best_emit)
