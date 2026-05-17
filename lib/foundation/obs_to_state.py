"""Convert a single-seat Kaggle obs to a JAX `GameState`.

Kaggle invokes `agent(obs, configuration)` with one seat's observation
(a dict or a `kaggle_environments.utils.Struct`). The foundation's
`Strategy.emit` consumes a `GameState` (the JAX Pytree). This module
bridges the two.

Why not just call `scalar_to_jax` directly? `scalar_to_jax` expects
`env.state` — a list of per-seat observations — not a single agent's
obs. We use `lib.fast_sim.from_obs` to build a `Snapshot` (which
already understands single-agent obs) and then convert its
`Snapshot.state` (a list of per-seat Structs) into a `GameState`.

Cost: ~30-80 ms per turn (Snapshot build + JAX-array packing). Most
of it is one-shot; subsequent calls to `scalar_to_jax` within the
same game amortise via the comet-schedule LRU cache at
`lib/game/jax/conversions.py:44`.
"""

from __future__ import annotations

from typing import Any

from lib.fast_sim import from_obs as _snapshot_from_obs
from lib.game.jax.conversions import scalar_to_jax
from lib.game.jax.jax_types import GameState


def _read_field(obs: Any, key: str, default: Any = None) -> Any:
    """Dual dict-or-attr access — Kaggle's live obs is a dict, the
    in-memory env uses Structs."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def obs_to_jax_state(
    obs: Any,
    configuration: Any = None,
    *,
    episode_seed: int = 0,
    num_seats: int = 2,
) -> GameState:
    """Build a JAX `GameState` from one seat's `obs`.

    Args:
        obs: dict or Struct, as Kaggle passes it.
        configuration: Kaggle's `configuration` arg (or None for defaults).
        episode_seed: env's `info["seed"]` if available; 0 otherwise.
            Within a horizon that doesn't cross a comet spawn boundary
            the value doesn't matter (see `lib/fast_sim.py` caveat).
        num_seats: 2 or 4 — currently inferred from observation if
            possible, otherwise 2.

    Returns a `GameState` ready to feed into `Strategy.emit` or any
    `lib.foundation.jax_engine` entry point.
    """
    # If the obs has explicit num_agents hint, prefer that.
    raw_num_seats = _read_field(obs, "num_agents", None)
    if raw_num_seats is not None:
        num_seats = int(raw_num_seats)

    snap = _snapshot_from_obs(
        obs,
        configuration=configuration,
        episode_seed=episode_seed,
        num_seats=num_seats,
    )
    return scalar_to_jax(snap.state, episode_seed)


def my_id_from_obs(obs: Any) -> int:
    """Read `obs.player` (or `obs["player"]`); default 0."""
    return int(_read_field(obs, "player", 0) or 0)
