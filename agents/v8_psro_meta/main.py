"""v8_psro_meta — PSRO meta-agent (mixed Nash over policy pool).

At game start (step 0), compute a deterministic seed from
`obs.initial_planets` and sample a policy index from the pool's
mixed-Nash distribution `NASH_PROBS`. Delegate all subsequent turns
to the chosen policy.

The Nash distribution is computed offline by `scripts/psro_solve.py`
on a payoff matrix from `scripts/psro_tournament.py`. Hardcoded
constants below; re-solve and re-paste when the pool or matrix
changes.

Per-game determinism property: σ-equivariant rotation of obs.initial_planets
produces the SAME hash → same policy index sampled → both seats play
the same policy. In self-play, this means both v8 agents pick the
same policy and play that policy's self-play game. So v8 self-play
inherits whatever the chosen policy's self-play property is (v7
self-play, v3 self-play, etc.).

Bundle note: each pool policy's lib dependencies must be inlined
into the bundle. We share lib/* across (v7, v3_snipe, roi) — those
all use the same lib. Precision has its own lib (agents/precision/*)
which would require additional inlining. First-iteration pool
EXCLUDES precision for bundle simplicity; precision was a
calibration opponent in the tournament but isn't sampled.
"""

from __future__ import annotations

import hashlib
import random
from typing import Callable


# ───────────────────────────────────────────────────────────────────────────
# EMBEDDED NASH DISTRIBUTION
# Re-paste from `scripts/psro_solve.py` output after each tournament re-run.
# ───────────────────────────────────────────────────────────────────────────

POOL_NAMES: list[str] = ["v7_minimax", "v3_snipe", "roi"]
NASH_PROBS: list[float] = [0.5, 0.5, 0.0]    # placeholder; replace after solving

assert abs(sum(NASH_PROBS) - 1.0) < 1e-6, "Nash probs must sum to 1"
assert len(POOL_NAMES) == len(NASH_PROBS)


# ───────────────────────────────────────────────────────────────────────────
# Lazy policy loaders — bundle inlines these via the bundler's --extra-agents
# ───────────────────────────────────────────────────────────────────────────

_POLICY_CACHE: dict[str, Callable] = {}


def _get_policy(name: str) -> Callable:
    """Return the agent function for a pool policy. Lazy + cached."""
    if name in _POLICY_CACHE:
        return _POLICY_CACHE[name]

    if name == "v7_minimax":
        from agents.v7_minimax.main import agent as fn
    elif name == "v3_snipe":
        from agents.v3_snipe.main import agent as fn
    elif name == "roi":
        from agents.simple.roi import agent as fn
    elif name == "v2":
        from agents.v2.main import agent as fn
    elif name == "v4_endgame":
        from agents.v4_endgame.main import agent as fn
    else:
        raise ValueError(f"unknown pool policy: {name}")

    _POLICY_CACHE[name] = fn
    return fn


# ───────────────────────────────────────────────────────────────────────────
# Per-player state
# ───────────────────────────────────────────────────────────────────────────

_STATE: dict[int, dict] = {}


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _seed_from_obs(obs) -> int:
    """Deterministic seed from initial planet positions.

    σ-equivariant: rotating the board 180° produces the same initial-
    planets set (positions just relabeled), so both seats hash to the
    same integer. Critical for self-play: both v8 agents sample the
    same policy.
    """
    ip = _obs_get(obs, "initial_planets", []) or []
    h = hashlib.md5()
    # Sort by ID first so rotation doesn't affect hash order
    for p in sorted(ip, key=lambda x: x[0]):
        # Use ID + rounded coords (σ-invariant after rotation when
        # positions are σ-symmetric — they always are by env design)
        h.update(f"{p[0]}:{round(p[2], 3)}:{round(p[3], 3)}".encode())
    return int(h.hexdigest()[:8], 16)


def _sample_policy_idx(game_seed: int) -> int:
    """Sample policy index from NASH_PROBS using a seeded RNG."""
    rng = random.Random(game_seed)
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(NASH_PROBS):
        cum += p
        if r <= cum:
            return i
    return len(NASH_PROBS) - 1


def _reset_state(my_id: int, obs):
    game_seed = _seed_from_obs(obs)
    idx = _sample_policy_idx(game_seed)
    name = POOL_NAMES[idx]
    _STATE[my_id] = {
        "policy_idx": idx,
        "policy_name": name,
        "policy_fn": _get_policy(name),
        "game_seed": game_seed,
    }


def agent(obs):
    my_id = int(_obs_get(obs, "player", 0))
    step = int(_obs_get(obs, "step", 0))
    st = _STATE.get(my_id)
    if step == 0 or st is None:
        _reset_state(my_id, obs)
        st = _STATE[my_id]
    return st["policy_fn"](obs)
