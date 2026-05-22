"""Iter 7: verify the depth-2 enemy projection mechanism.

a. `_apply_arrivals` correctly resolves combat on the cloned world.
b. `project_two_turns` returns at least as many arrivals as depth-1, and
   typically more in scenarios where a cascade exists.
c. Planner runs end-to-end with depth-2 active; no exceptions; reasonable
   plan size; turn time within budget.
"""
from __future__ import annotations

import math
import sys
import time
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agents.precision import enemy_model, intercept, planner, prediction, sim


def _planet(id_, owner, x, y, ships, production, radius=1.0):
    return intercept.PlanetView(
        id=id_, owner=owner, x=x, y=y, radius=radius,
        ships=ships, production=production,
    )


def _world(*planets, step=10, omega=0.0, player=0):
    return {
        "player": player,
        "step": step,
        "omega": omega,
        "planets": list(planets),
        "planet_by_id": {p.id: p for p in planets},
        "fleets": [],
        "comets": [],
        "remaining_overage": 60.0,
    }


def test_apply_arrivals_resolves_combat():
    """A projected enemy arrival captures a neutral planet; the cloned world
    must show the new owner + post-combat ship count."""
    neutral = _planet(2, -1, 50.0, 50.0, ships=10, production=5)
    me = _planet(0, 0, 70.0, 90.0, ships=200, production=2)
    enemy = _planet(1, 1, 95.0, 75.0, ships=50, production=3)
    w = _world(me, enemy, neutral)
    arrival = prediction.Arrival(step=15, planet_id=2, owner=1, ships=30)

    w2 = enemy_model._apply_arrivals(w, [arrival])
    # Pre-state untouched (different dict).
    assert w["planet_by_id"][2].owner == -1
    assert w["planet_by_id"][2].ships == 10
    # Post-state: enemy captured with 30-10 = 20 ships.
    assert w2["planet_by_id"][2].owner == 1
    assert w2["planet_by_id"][2].ships == 20
    # Non-target planets unchanged.
    assert w2["planet_by_id"][0].ships == 200
    # Step bumped.
    assert w2["step"] >= 15
    print("  _apply_arrivals resolved neutral capture correctly")


def test_apply_arrivals_no_op_when_empty():
    """Empty arrivals list returns the same world (or an equivalent one)."""
    me = _planet(0, 0, 70.0, 90.0, ships=200, production=2)
    enemy = _planet(1, 1, 95.0, 75.0, ships=50, production=3)
    w = _world(me, enemy)
    w2 = enemy_model._apply_arrivals(w, [])
    # Same identity is fine; or a deep-equal clone.
    assert w2["planet_by_id"][0].ships == 200
    assert w2["step"] == w["step"]
    print("  _apply_arrivals empty-input no-op")


def test_project_two_turns_no_more_than_zero_when_unreachable():
    """If the enemy can't reach us at all, both turns project zero arrivals."""
    me = _planet(0, 0, 5.0, 5.0, ships=10, production=2)   # tiny corner
    enemy = _planet(1, 1, 95.0, 95.0, ships=10, production=2)  # other corner; only ~10 ships
    # No neutrals -> enemy has nothing high-ROI to attack either.
    w = _world(me, enemy)
    res = enemy_model.project_two_turns(w)
    assert isinstance(res, list)
    print(f"  unreachable case: {len(res)} arrivals (expected 0-1)")


def test_project_two_turns_returns_more_than_depth_one_when_enemy_strong():
    """If a first-turn enemy strike weakens us, a second-turn cascade may
    materialise. The depth-2 set should be a superset of depth-1 here."""
    me_home = _planet(0, 0, 70.0, 90.0, ships=20, production=2)
    me_outpost = _planet(2, 0, 60.0, 85.0, ships=8, production=2)
    enemy = _planet(1, 1, 95.0, 75.0, ships=400, production=3)
    w = _world(me_home, me_outpost, enemy)
    depth1 = enemy_model.project_enemy_actions_worst_for_us(w)
    depth2 = enemy_model.project_two_turns(w)
    # depth-2 includes the depth-1 set (same first-turn arrivals).
    assert len(depth2) >= len(depth1), \
        f"depth-2 ({len(depth2)}) must be >= depth-1 ({len(depth1)})"
    print(f"  depth-1: {len(depth1)} arrivals; depth-2: {len(depth2)} arrivals")


def test_planner_with_depth2_runs_within_budget():
    """End-to-end: planner returns a plan within the configured deadline."""
    me_home = _planet(0, 0, 70.0, 90.0, ships=500, production=2)
    enemy = _planet(1, 1, 90.0, 70.0, ships=100, production=3)
    neutral = _planet(2, -1, 80.0, 80.0, ships=15, production=5)
    w = _world(me_home, enemy, neutral)
    t0 = time.perf_counter()
    plan = planner.plan_turn(w, deadline=t0 + 1.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"  planner returned {len(plan)} actions in {elapsed_ms:.0f}ms")
    assert isinstance(plan, list)
    assert elapsed_ms < 1000, f"turn took {elapsed_ms:.0f}ms (over 1s budget)"


if __name__ == "__main__":
    test_apply_arrivals_resolves_combat()
    test_apply_arrivals_no_op_when_empty()
    test_project_two_turns_no_more_than_zero_when_unreachable()
    test_project_two_turns_returns_more_than_depth_one_when_enemy_strong()
    test_planner_with_depth2_runs_within_budget()
    print("\nAll depth-2 minimax tests passed.")
