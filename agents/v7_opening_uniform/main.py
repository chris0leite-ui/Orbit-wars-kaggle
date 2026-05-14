"""v7_opening_uniform — opening cadence bump WITHOUT cluster classification.

Tests whether the v2 sweep's apparent wins came from the broken
classifier accidentally forcing the high-cadence cluster-3 template
onto every board, rather than from cluster-conditional finesse. If
this variant matches or beats v7_0_drop_one at the same rate the
broken v2 did, the cluster classifier was contributing nothing — and
the fix is to ship a uniform aggressive overlay.

Mechanism: turns 0-30 enforce a uniform target of `LAUNCHES_30 = 6`
fleets, with the same ROI target selection as `lib.opening_overlay`.
No board fingerprint, no centroid distance, no outlier fallback.
"""

from __future__ import annotations

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet  # noqa: F401
from lib.opening_overlay import (
    OVERLAY_HORIZON, Template, propose_opening_actions,
)
from lib.v7_search import choose


LAUNCHES_30 = 6.0
FIRST_LAUNCH = 4


class _UniformTemplate:
    """Drop-in for `lib.opening_overlay.Template` with fixed fields."""
    first_launch = FIRST_LAUNCH
    launches_30 = LAUNCHES_30
    target_dist = 35.0
    target_prod = 3.0


_FIXED_TEMPLATE = _UniformTemplate()


class _State:
    launches: int = 0
    episode_seed: int | None = None


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


def _reset_if_new_episode(obs, configuration, state: _State) -> None:
    step = int(_read_obs(obs, "step", 0))
    seed = None
    if configuration is not None:
        seed = (
            configuration.get("seed")
            if isinstance(configuration, dict)
            else getattr(configuration, "seed", None)
        )
    if step == 0 or (seed is not None and seed != state.episode_seed):
        state.launches = 0
        state.episode_seed = seed


def _propose_uniform(obs, my_seat: int, launches_so_far: int) -> list[list]:
    """Same ROI-style proposer as lib.opening_overlay.propose_opening_actions
    but with the fixed `_FIXED_TEMPLATE` instead of a cluster lookup."""
    planets = _read_obs(obs, "planets", []) or []
    step = int(_read_obs(obs, "step", 0))
    if step < _FIXED_TEMPLATE.first_launch:
        return []
    target_so_far = int(math.ceil(_FIXED_TEMPLATE.launches_30 * (step + 1) / 30))
    if launches_so_far >= target_so_far:
        return []
    my_planets = [p for p in planets if p[1] == my_seat and p[5] > 5]
    if not my_planets:
        return []
    candidates = []
    for src in my_planets:
        for t in planets:
            if t[1] == my_seat:
                continue
            d = math.hypot(t[2] - src[2], t[3] - src[3])
            cost = max(1, int(t[5]) + 1)
            if cost >= int(src[5]) - 5:
                continue
            roi = t[6] / (cost + d + 1.0)
            tmpl = 1.0 - min(1.0, abs(t[6] - _FIXED_TEMPLATE.target_prod) / 3.0)
            score = roi * (1.0 + 0.2 * tmpl)
            candidates.append((-score, d, src, t, cost))
    if not candidates:
        return []
    candidates.sort(key=lambda c: (c[0], c[1]))
    _, _, src, target, cost = candidates[0]
    aim = math.atan2(target[3] - src[3], target[2] - src[2])
    return [[int(src[0]), float(aim), int(cost)]]


def agent(obs, configuration=None):
    my_seat = int(_read_obs(obs, "player", 0))
    state = _state_for(my_seat)
    _reset_if_new_episode(obs, configuration, state)
    step = int(_read_obs(obs, "step", 0))

    if step < OVERLAY_HORIZON:
        actions = _propose_uniform(obs, my_seat, state.launches)
        if actions:
            state.launches += len(actions)
            return actions

    actions = choose(
        obs, configuration,
        enumerator_mode="drop_one",
        K=10,
        wallclock_ms=700.0,
    )
    if step < OVERLAY_HORIZON:
        state.launches += len(actions) if isinstance(actions, list) else 0
    return actions
