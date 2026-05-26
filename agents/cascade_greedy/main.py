"""cascade_greedy — admissibility-based deterministic chooser.

Replaces SA in `agents/sa_online` with iterative ctx-rebuild + greedy
selection over the cascade-DAG. Seat-aware (reads `obs.player`), no
plan carryover across turns, no opponent model, no module-load
bootstrap (so it survives Kaggle's blocked recursive `make()`).

Per-turn budget target: ~280 ms (3 rebuilds × ~80 ms ctx + joints +
greedy). Comfortably under the 1 s actTimeout.

Critical: NO `__file__` at module top-level. Kaggle loads agents via
`exec(compile(source), {})` with an empty namespace; any reference
to `__file__` raises NameError which is silently swallowed into a
fallback no-op.
"""
from __future__ import annotations

import copy
import os

# Module-level caches; Kaggle starts one fresh process per episode.
_CACHED_SNAP = None
_PATH_GRAPH = None
_EPISODE_RESET = False

from lib.cascade_chooser import cascade_greedy_select
from lib.fast_sim import from_obs as fs_from_obs
from lib.intent import World as _SAWorld
from lib.path_graph import build_path_graph as _build_path_graph
from lib.sa_core import reset_fate_cache


# Horizon in turns. Cascade DP gains from looking further out, but
# admissibility enumeration cost grows linearly with the horizon.
DEFAULT_HORIZON = 25


def _read_obs_field(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _get_step(obs) -> int:
    if isinstance(obs, dict):
        return int(obs.get("step", 0))
    return int(getattr(obs, "step", 0))


def _resolve_seed_and_steps(obs, configuration) -> tuple[int, int]:
    """Pull seed + episodeSteps from configuration (Kaggle supplies both)."""
    if configuration is None:
        return 0, 200
    if isinstance(configuration, dict):
        seed_v = configuration.get("seed")
        steps_v = configuration.get("episodeSteps")
    else:
        seed_v = getattr(configuration, "seed", None)
        steps_v = getattr(configuration, "episodeSteps", None)
    try:
        seed = int(seed_v) if seed_v is not None else 0
    except (TypeError, ValueError):
        seed = 0
    try:
        steps = int(steps_v) if steps_v is not None else 200
    except (TypeError, ValueError):
        steps = 200
    return seed, steps


def _get_or_refresh_snap(obs, configuration, seed):
    """Build snap once with planet_position_cache, mutate fields in place.

    The position cache is GAME-INVARIANT (depends only on episode seed
    + initial planets, which don't change mid-game), so we keep it
    across turns and only refresh the mutable game state. This is the
    pattern from `agents/sa_online/main.py:123-162`.
    """
    global _CACHED_SNAP
    if _CACHED_SNAP is None:
        _CACHED_SNAP = fs_from_obs(obs, configuration,
                                    episode_seed=seed, num_seats=2)
        return _CACHED_SNAP
    obs0 = _CACHED_SNAP.state[0].observation
    for k in ("planets", "fleets", "comets"):
        v = _read_obs_field(obs, k)
        if v is not None:
            setattr(obs0, k, copy.deepcopy(v))
    for k in ("step", "next_fleet_id"):
        v = _read_obs_field(obs, k)
        if v is not None:
            setattr(obs0, k, v)
    if len(_CACHED_SNAP.state) > 1:
        obs1 = _CACHED_SNAP.state[1].observation
        for k in ("planets", "fleets", "comets", "step", "next_fleet_id"):
            setattr(obs1, k, getattr(obs0, k))
    return _CACHED_SNAP


def _get_or_build_path_graph(obs, steps):
    """Lazy-build feasibility graph past Kaggle's turn-0 actTimeout."""
    global _PATH_GRAPH
    if _PATH_GRAPH is not None:
        return _PATH_GRAPH
    try:
        obs_d = obs if isinstance(obs, dict) else dict(obs)
        world = _SAWorld.from_obs(obs_d)
        orb_bucket = int(os.environ.get(
            "CASCADE_PATH_GRAPH_ORBITING_BUCKET", "8"))
        com_bucket = int(os.environ.get(
            "CASCADE_PATH_GRAPH_COMET_BUCKET", "2"))
        _PATH_GRAPH = _build_path_graph(
            world, t_max=int(steps),
            orbiting_bucket=orb_bucket, comet_bucket=com_bucket)
    except Exception:
        _PATH_GRAPH = None
    return _PATH_GRAPH


def agent(obs, configuration=None):
    global _EPISODE_RESET
    if not _EPISODE_RESET:
        # Fate cache is module-level. Kaggle starts a fresh process per
        # episode (so the cache is empty); tests share processes, so we
        # reset on first invocation to keep behaviour identical.
        reset_fate_cache()
        _EPISODE_RESET = True

    seed, steps = _resolve_seed_and_steps(obs, configuration)
    t = _get_step(obs)
    me = int(_read_obs_field(obs, "player", 0))

    snap = _get_or_refresh_snap(obs, configuration, seed)
    pg = _get_or_build_path_graph(obs, steps)

    horizon = min(int(steps) - t, DEFAULT_HORIZON)
    if horizon <= 0:
        return []

    try:
        plan = cascade_greedy_select(
            snap,
            t_start=t, t_end=t + horizon, me=me,
            opp_policy=None,
            path_graph=pg,
        )
    except Exception:
        # Per-turn refine failure: emit nothing rather than crash. The
        # outer loop will reset state next call.
        return []

    return [list(a) for tau, a in plan if int(tau) == int(t)]
