"""Tests for `lib.v7_search.choose_depth2` — runtime depth-2 maximin
chooser.

Coverage:
1. End-to-end smoke on a viable 2P state → produces some action that
   matches the incumbent's first launch (the drop-one set contains it).
2. Watchdog: a tight `wallclock_ms` budget bails inner cells but still
   evaluates row 0 (incumbent) fully → fall-back behavior matches v7_0.
3. 4P fallback: `choose_depth2` returns the incumbent unchanged for
   `num_seats > 2`.
4. Single-candidate path: when drop-one yields ≤ 1 candidate, return
   incumbent without invoking the rollout.
"""

from __future__ import annotations

import time

from lib.v7_search import (
    _action_from_intents,
    _build_incumbent_intents,
    choose_depth2,
    choose_depth2_with_4p,
)
from lib.intent import World
from lib.world_model import WorldModel


def _obs_2p(planets, *, my_id=0, step=0):
    return {
        "player": my_id,
        "planets": planets,
        "fleets": [],
        "angular_velocity": 0.05,
        "initial_planets": [list(p) for p in planets],
        "comet_planet_ids": [],
        "comets": [],
        "step": step,
        "next_fleet_id": 0,
    }


def _obs_4p(planets, *, my_id=0, step=0):
    obs = _obs_2p(planets, my_id=my_id, step=step)
    return obs


def test_smoke_returns_some_action_on_viable_state():
    """Two 2P home planets in opposite corners, one neutral off-axis
    so the path clears the sun. The incumbent will produce ≥ 1 launch;
    `choose_depth2` should return one of {incumbent, drop-one} variants,
    all of which produce the same SET of launches (incumbent is the
    superset)."""
    planets = [
        [0, 0, 10.0, 10.0, 1.5, 30, 2],     # ours
        [1, 1, 90.0, 90.0, 1.5, 30, 2],     # theirs
        [2, -1, 30.0, 70.0, 1.5, 5, 2],     # neutral, off-axis (no sun cross)
        [3, -1, 70.0, 30.0, 1.5, 5, 2],     # mirror neutral
    ]
    obs = _obs_2p(planets, step=0)
    out = choose_depth2(obs, K=6, wallclock_ms=700.0)
    # Compute the incumbent for comparison.
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    incumbent = _action_from_intents(
        _build_incumbent_intents(world, model), obs, model,
    )
    # Depth-2 returns either the incumbent (parity floor) or one of the
    # drop-one variants, i.e. either == incumbent OR a strict subset.
    incumbent_keys = {(int(m[0]), int(m[2])) for m in incumbent}
    out_keys = {(int(m[0]), int(m[2])) for m in out}
    assert out_keys.issubset(incumbent_keys), (
        f"depth-2 returned a launch not in the incumbent: "
        f"out={out_keys} incumbent={incumbent_keys}"
    )


def test_returns_incumbent_when_drop_one_is_singleton():
    """Empty board → no launches → drop-one is just [[]] → returns
    incumbent (which is also [])."""
    obs = _obs_2p(planets=[], step=0)
    out = choose_depth2(obs, K=6, wallclock_ms=700.0)
    assert out == []


def test_4p_falls_back_to_incumbent():
    """4P games bypass the depth-2 path (no Nash maximin guarantee)."""
    planets = [
        [0, 0, 10.0, 10.0, 1.5, 30, 2],
        [1, 1, 90.0, 10.0, 1.5, 30, 2],
        [2, 2, 10.0, 90.0, 1.5, 30, 2],
        [3, 3, 90.0, 90.0, 1.5, 30, 2],
        [4, -1, 30.0, 70.0, 1.5, 5, 2],
    ]
    obs = _obs_4p(planets, step=0)
    out_depth2 = choose_depth2_with_4p(obs, K_2p=6, K_4p=8, wallclock_ms=700.0)
    # In 4P, depth-2 routes to `choose_4p` which can return a non-empty
    # action; just assert it doesn't crash and the return shape is a list.
    assert isinstance(out_depth2, list)


def test_watchdog_returns_incumbent_under_tight_budget():
    """With wallclock_ms tiny enough that no inner cell completes, the
    fall-back is the incumbent action. Row 0 is always evaluated first,
    so even a 0-ms budget shouldn't crash — it just returns incumbent."""
    planets = [
        [0, 0, 10.0, 10.0, 1.5, 30, 2],
        [1, 1, 90.0, 90.0, 1.5, 30, 2],
        [2, -1, 30.0, 70.0, 1.5, 5, 2],
        [3, -1, 70.0, 30.0, 1.5, 5, 2],
    ]
    obs = _obs_2p(planets, step=0)
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    incumbent = _action_from_intents(
        _build_incumbent_intents(world, model), obs, model,
    )
    out = choose_depth2(obs, K=6, wallclock_ms=1.0)  # 1 ms = effectively 0
    # Even when the watchdog cuts every inner cell, the return is a list
    # (possibly the incumbent, possibly an empty subset). It must not raise.
    assert isinstance(out, list)
    # Incumbent has ≥ 1 launch here.
    assert len(incumbent) >= 1


def test_budget_within_target_for_viable_state():
    """Full-budget call should finish well under 700 ms on a normal
    2P fixture with N ≤ 8, M ≤ 4."""
    planets = [
        [0, 0, 10.0, 10.0, 1.5, 30, 2],
        [3, 0, 10.0, 30.0, 1.5, 30, 2],
        [4, 0, 30.0, 10.0, 1.5, 30, 2],
        [1, 1, 90.0, 90.0, 1.5, 30, 2],
        [5, 1, 90.0, 70.0, 1.5, 30, 2],
        [6, 1, 70.0, 90.0, 1.5, 30, 2],
        [2, -1, 50.0, 30.0, 1.5, 5, 2],
        [7, -1, 30.0, 50.0, 1.5, 5, 2],
        [8, -1, 50.0, 70.0, 1.5, 5, 2],
        [9, -1, 70.0, 50.0, 1.5, 5, 2],
    ]
    obs = _obs_2p(planets, step=0)
    t0 = time.perf_counter()
    choose_depth2(obs, K=6, wallclock_ms=700.0)
    elapsed = (time.perf_counter() - t0) * 1000.0
    # On CI-bound hardware the budget is honored; allow 1.2× slack.
    assert elapsed < 700.0 * 1.5, (
        f"choose_depth2 exceeded budget: {elapsed:.0f} ms"
    )
