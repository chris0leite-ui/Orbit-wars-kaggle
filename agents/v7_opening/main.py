"""v7_opening — cluster-conditional opening overlay + v7 search.

For steps 0..29: classify board into one of 4 cluster archetypes
(`lib.opening_overlay`) and propose launches matching the cluster's
empirical opening template (fit from 60 top-10 win replays in
`audit/2026-05-14-opening-overlay-model.json`).

For steps >= 30: defer to `lib.v7_search.choose(... enumerator_mode='drop_one')`
— same code path as the live v7_pv anchor.

State: per-episode `launches_so_far` counter, reset whenever a fresh
step-0 obs is seen. Single global is safe because Kaggle runs each
agent in its own process.

Origin: audit/2026-05-14-loss-mode-mine.md (gap is opening-100 launch
rate, 0.44 vs top-10's 0.70) + audit/2026-05-14-board-taxonomy.json
(Mine 1: 4 clusters with prod 1.25, dist 17.3 spread).
"""

from __future__ import annotations

from lib.opening_overlay import (
    OVERLAY_HORIZON, classify_board, propose_opening_actions,
)
from lib.v7_search import choose


class _State:
    cluster: int = 0
    launches: int = 0
    episode_seed: int | None = None


# Per-seat state. In production the Kaggle harness runs each agent in
# its own process so seat==0 is the only entry; for local self-play we
# need separate counters because env.run() loops both seats inside one
# Python process and would otherwise reset each other's launches at
# step 0.
_state_by_seat: dict[int, _State] = {}


def _state_for(seat: int) -> _State:
    s = _state_by_seat.get(seat)
    if s is None:
        s = _State()
        _state_by_seat[seat] = s
    return s


def _read_obs(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _episode_reset_if_needed(obs, configuration, state: _State) -> None:
    """Reset `state` when we see a fresh episode (step 0 or new seed)."""
    step = int(_read_obs(obs, "step", 0))
    seed = None
    if configuration is not None:
        seed = (
            configuration.get("seed")
            if isinstance(configuration, dict)
            else getattr(configuration, "seed", None)
        )
    if step == 0 or (seed is not None and seed != state.episode_seed):
        planets = _read_obs(obs, "planets", []) or []
        omega = float(_read_obs(obs, "angular_velocity", 0.0) or 0.0)
        state.cluster = classify_board(planets, omega)
        state.launches = 0
        state.episode_seed = seed


def agent(obs, configuration=None):
    my_seat = int(_read_obs(obs, "player", 0))
    state = _state_for(my_seat)
    _episode_reset_if_needed(obs, configuration, state)
    step = int(_read_obs(obs, "step", 0))

    if step < OVERLAY_HORIZON and state.cluster >= 0:
        # cluster == -1 means classify_board flagged the board as an
        # outlier (beyond p90 centroid distance on training corpus);
        # skip overlay entirely and use pure v7. Origin: v2 sweep
        # diagnostic — seed 100 had d_min=6.11 vs training p90=4.31
        # and the cluster-1 template over-fired on its sparse 1.62
        # mean_prod board.
        actions = propose_opening_actions(
            obs, my_seat, state.cluster, state.launches,
        )
        if actions:
            state.launches += len(actions)
            return actions
        # Else fall through to v7 — overlay declined this turn but v7
        # may still want to launch (e.g. defend a snipe). This is a
        # purely-additive overlay: it raises the floor on launch
        # cadence but never blocks v7's reactive plays. Count v7's
        # launches into state.launches too so the overlay's pro-rated
        # cadence target reflects ACTUAL launches (overlay + v7), not
        # only overlay-issued ones — otherwise we'd over-stack when
        # v7 fires reactively and overlay still thinks we owe the
        # template count.

    actions = choose(
        obs, configuration,
        enumerator_mode="drop_one",
        K=10,
        wallclock_ms=700.0,
    )
    if step < OVERLAY_HORIZON:
        state.launches += len(actions) if isinstance(actions, list) else 0
    return actions
