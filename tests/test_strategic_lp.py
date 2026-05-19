"""Unit tests for agents/baseline/strategic_lp — Slice 6.

Plan reference: /root/.claude/plans/take-the-lens-of-magical-shore.md §12.

Covers `build_capture_matrix`, `solve_assignment`, and the
`compute_lp_assignment` convenience wrapper.
"""

from __future__ import annotations

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from agents.baseline.strategic_lp import (
    INFEASIBLE_COST,
    _greedy_assignment,
    build_capture_matrix,
    compute_lp_assignment,
    solve_assignment,
)
from lib.intent import World
from lib.world_model import WorldModel


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    return Planet(pid, owner, x, y, radius, ships, production)


def _world(my_id, planets, *, step=0, omega=0.0):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": [],
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "comets": [],
        "step": step,
    }
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# build_capture_matrix
# ---------------------------------------------------------------------------


def test_build_capture_matrix_empty_when_no_sources():
    """No my-planets with ships → empty sources, empty matrix."""
    tgt = _planet(0, -1, 10.0, 50.0, ships=5)
    world = _world(0, [tgt])
    model = WorldModel.from_world(world)
    sources, targets, matrix = build_capture_matrix(world, model, me=0)
    assert sources == []
    assert len(targets) == 1
    assert matrix == []


def test_build_capture_matrix_empty_when_no_targets():
    """All planets are mine → empty targets, empty matrix."""
    a = _planet(0, 0, 10.0, 50.0, ships=80)
    b = _planet(1, 0, 50.0, 50.0, ships=20)
    world = _world(0, [a, b])
    model = WorldModel.from_world(world)
    sources, targets, matrix = build_capture_matrix(world, model, me=0)
    assert len(sources) == 2
    assert targets == []
    # Each source row is empty (no targets to fill).
    assert all(len(row) == 0 for row in matrix)


def test_build_capture_matrix_one_pair_feasible():
    """One src, one weak neutral tgt → matrix has a finite entry."""
    src = _planet(0, 0, 10.0, 50.0, ships=100, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=5, production=2)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    sources, targets, matrix = build_capture_matrix(world, model, me=0)
    assert len(sources) == 1
    assert len(targets) == 1
    assert matrix[0][0] < float("inf")
    assert matrix[0][0] > 0  # some travel time


def test_build_capture_matrix_infeasible_when_undersized():
    """src.ships too small to overcome tgt garrison → +inf."""
    src = _planet(0, 0, 10.0, 50.0, ships=5)
    tgt = _planet(1, -1, 30.0, 50.0, ships=200, production=2)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    sources, targets, matrix = build_capture_matrix(world, model, me=0)
    assert matrix[0][0] == float("inf")


# ---------------------------------------------------------------------------
# solve_assignment
# ---------------------------------------------------------------------------


def test_solve_assignment_empty_inputs():
    """No sources or no targets → empty assignment."""
    assert solve_assignment([], [], []) == {}
    src = _planet(0, 0, 10.0, 50.0)
    assert solve_assignment([(0, src)], [], []) == {}


def test_solve_assignment_picks_higher_value_target():
    """One source, two targets — picks the higher-value one."""
    src = _planet(0, 0, 10.0, 50.0, ships=80)
    tgt_high = _planet(1, -1, 30.0, 50.0, ships=5, production=5)
    tgt_low = _planet(2, -1, 30.0, 50.0, ships=5, production=1)
    sources = [(0, src)]
    targets = [(1, tgt_high), (2, tgt_low)]
    matrix = [[5.0, 5.0]]  # same capture time, different value
    assignment = solve_assignment(sources, targets, matrix)
    assert assignment == {0: 1}  # picks high-production target


def test_solve_assignment_one_source_per_target():
    """Two sources, two targets, each src assigned to a unique tgt."""
    src_a = _planet(0, 0, 10.0, 50.0, ships=80)
    src_b = _planet(1, 0, 70.0, 50.0, ships=80)
    tgt_a = _planet(2, -1, 25.0, 50.0, ships=5, production=2)
    tgt_b = _planet(3, -1, 75.0, 50.0, ships=5, production=2)
    sources = [(0, src_a), (1, src_b)]
    targets = [(2, tgt_a), (3, tgt_b)]
    # Each source is closest to one target (lower cost).
    matrix = [[3.0, 20.0], [20.0, 3.0]]
    assignment = solve_assignment(sources, targets, matrix)
    assert assignment == {0: 2, 1: 3}


def test_solve_assignment_skips_infeasible():
    """Pairs marked +inf are never picked."""
    src = _planet(0, 0, 10.0, 50.0, ships=80)
    tgt = _planet(1, -1, 30.0, 50.0, ships=5, production=2)
    sources = [(0, src)]
    targets = [(1, tgt)]
    matrix = [[float("inf")]]
    assignment = solve_assignment(sources, targets, matrix)
    assert assignment == {}


# ---------------------------------------------------------------------------
# _greedy_assignment (pure-Python fallback)
# ---------------------------------------------------------------------------


def test_greedy_assignment_picks_minimum():
    """Greedy picks the lowest cost first, then next-best on remaining rows/cols."""
    cost = [
        [-10.0, -3.0],
        [-1.0, -5.0],
    ]
    row_ind, col_ind = _greedy_assignment(cost)
    assert (row_ind, col_ind) == ([0, 1], [0, 1])  # picks -10 first, then -5


def test_greedy_assignment_skips_infeasible():
    cost = [[INFEASIBLE_COST, -3.0]]
    row_ind, col_ind = _greedy_assignment(cost)
    assert (row_ind, col_ind) == ([0], [1])


def test_greedy_assignment_all_infeasible():
    cost = [[INFEASIBLE_COST, INFEASIBLE_COST]]
    row_ind, col_ind = _greedy_assignment(cost)
    assert (row_ind, col_ind) == ([], [])


# ---------------------------------------------------------------------------
# compute_lp_assignment (end-to-end wrapper)
# ---------------------------------------------------------------------------


def test_compute_lp_assignment_assigns_natural_pairing():
    """End-to-end: two sources, two targets, geometry implies natural pairing."""
    src_a = _planet(0, 0, 10.0, 50.0, ships=80, production=2)
    src_b = _planet(1, 0, 90.0, 50.0, ships=80, production=2)
    tgt_a = _planet(2, -1, 25.0, 50.0, ships=5, production=2)
    tgt_b = _planet(3, -1, 75.0, 50.0, ships=5, production=2)
    world = _world(0, [src_a, src_b, tgt_a, tgt_b])
    model = WorldModel.from_world(world)
    assignment = compute_lp_assignment(world, model, me=0)
    assert assignment == {0: 2, 1: 3}


def test_compute_lp_assignment_empty_when_no_my_planets():
    """No sources → empty assignment."""
    tgt = _planet(0, -1, 10.0, 50.0, ships=5)
    world = _world(0, [tgt])
    model = WorldModel.from_world(world)
    assert compute_lp_assignment(world, model, me=0) == {}
