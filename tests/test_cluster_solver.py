"""Cluster solver correctness on constructed scenarios.

Three scenarios mirror `tests/test_analytics.py`'s capture-math units,
but verify the solver's CHOICE, not just the closed-form solver's
prediction. The cluster solver should:

- Pick a launch action in a free-capture cluster.
- Pick IDLE (no launch) in a bounce cluster.
- Pick IDLE at turn 0 in a wait-and-fire cluster (because no
  launch candidate is feasible yet); the future launch is encoded
  in the recursion's value, not the root action.

Search depth is set to 14 — enough to see a 5-12 turn ETA capture
payoff without burning the budget on long idle tails.
"""

from __future__ import annotations

from lib.cluster_solver.detector import find_solvable_clusters
from lib.cluster_solver.minimax import solve
from lib.trajectory_layer import World
from tests.scenarios.base import _obs, _planet


def _setup_and_solve(planets, max_depth=14):
    obs = _obs(planets=planets, step=10, player=0)
    world = World.from_obs(obs)
    clusters = find_solvable_clusters(world)
    assert clusters, "Detector failed to find a cluster"
    cluster = clusters[0]
    return solve(cluster.isolated_obs, cluster.my_id, cluster.opp_id,
                 max_depth=max_depth, budget_ms=20000)


def test_free_capture_solver_launches():
    planets = [
        _planet(0, owner=0, x=10.0, y=50.0, ships=100, production=2),
        _planet(1, owner=-1, x=30.0, y=50.0, ships=5, production=1),
        _planet(2, owner=1, x=90.0, y=10.0, ships=10, production=1),
    ]
    result = _setup_and_solve(planets, max_depth=14)
    assert result.best_action, (
        f"free capture must launch; got best_action={result.best_action}"
    )
    # First move should be a launch from planet 0 toward planet 1 (angle ≈ 0)
    src, angle, ships = result.best_action[0]
    assert src == 0
    assert abs(angle) < 0.1, f"angle should be ~0 (east); got {angle}"
    assert ships >= 5


def test_bounce_solver_idles():
    planets = [
        _planet(0, owner=0, x=10.0, y=50.0, ships=10, production=1),
        _planet(1, owner=1, x=30.0, y=50.0, ships=100, production=2),
    ]
    result = _setup_and_solve(planets, max_depth=14)
    assert result.best_action == [], (
        f"bounce scenario must pick IDLE; got {result.best_action}"
    )


def test_wait_and_fire_idles_at_root():
    """Source has 5 ships, target needs ≥31. No launch candidate at turn 0.
    Solver must pick IDLE at the root; the future launch is in the recursion."""
    planets = [
        _planet(0, owner=0, x=10.0, y=50.0, ships=5, production=4),
        _planet(1, owner=-1, x=30.0, y=50.0, ships=30, production=1),
        _planet(2, owner=1, x=90.0, y=10.0, ships=10, production=1),
    ]
    result = _setup_and_solve(planets, max_depth=14)
    assert result.best_action == [], (
        f"wait-and-fire scenario must IDLE at root; got {result.best_action}"
    )
    # Value should reflect SOME ship gain (idle accrues production).
    assert result.value > 0
