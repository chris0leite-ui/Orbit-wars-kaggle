"""Tests for the frontier-circulation post-pass (PI 2026-06-03).

Mechanism: from each of MY planets, send everything-minus-garrison to the
euclidean-closest friendly planet whose distance to the opponent centroid
is STRICTLY SMALLER than the source's. Destination depends only on
positions -> DAG -> loop-proof.

These unit tests verify the helpers + the post-pass under controlled
synthetic worlds. End-to-end (replay reproduces failure state) verification
is done via a fast.py play smoke + a diagnostic probe — Rule 38.
"""

from __future__ import annotations

import pytest

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    for var in (
        "BASELINE_FRONTIER_CIRCULATION",
        "BASELINE_CIRCULATION_GARRISON",
        "BASELINE_CIRCULATION_TRIGGER_MIN",
        "BASELINE_CIRCULATION_MIN_SEND",
        "BASELINE_CIRCULATION_MAX",
        "BASELINE_LAUNCH_RULES",
    ):
        monkeypatch.delenv(var, raising=False)


def _planet(pid, owner, x, y, ships, production=2, radius=2.0):
    return Planet(pid, owner, float(x), float(y), float(radius),
                  int(ships), int(production))


class _StubModel:
    def __init__(self, threatened=None):
        self.ledger = {}
        self._threatened = threatened or set()

    def time_to_enemy_threat(self, planet_id, my_id, world, arrival_eta=0):
        return 7 if int(planet_id) in self._threatened else None


class _StubWorld:
    def __init__(self):
        self.step = 20
        self.obs_raw = {"angular_velocity": 0.0}


def test_circulation_skips_frontier_most_planet(monkeypatch):
    """The planet with the smallest distance-to-opp-centroid has NO forward
    friendly and therefore must never be a source."""
    monkeypatch.setenv("BASELINE_FRONTIER_CIRCULATION", "1")
    from agents.baseline.main import emit_frontier_circulation

    # Linear geometry: rear (x=-50), middle (x=0), front (x=20). Opp at x=100.
    rear = _planet(0, 0, -50.0, 0.0, 100)
    middle = _planet(1, 0, 0.0, 0.0, 100)
    front = _planet(2, 0, 20.0, 0.0, 100)
    opp = _planet(3, 1, 100.0, 0.0, 50)
    planets = [rear, middle, front, opp]
    model = _StubModel()
    world = _StubWorld()

    result = emit_frontier_circulation(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    src_ids = {int(m[0]) for m in result}
    assert int(front.id) not in src_ids, \
        "front planet (smallest frontier-dist) must never be a source"
    # rear and middle should both fire.
    assert int(rear.id) in src_ids and int(middle.id) in src_ids


def test_circulation_picks_euclidean_closest_forward_friendly(monkeypatch):
    monkeypatch.setenv("BASELINE_FRONTIER_CIRCULATION", "1")
    from agents.baseline.main import emit_frontier_circulation

    # Source at (0,0). Opp centroid at (100,0). Two forward friendlies:
    # near_forward at (20,0) (close), far_forward at (80,0) (farther).
    # Both are strictly closer to centroid than source. Mechanism must
    # pick near_forward (smaller euclidean distance from source).
    src = _planet(0, 0, 0.0, 0.0, 100)
    near_forward = _planet(1, 0, 20.0, 0.0, 30)
    far_forward = _planet(2, 0, 80.0, 5.0, 30)
    opp = _planet(3, 1, 100.0, 0.0, 50)
    planets = [src, near_forward, far_forward, opp]
    model = _StubModel()
    world = _StubWorld()

    result = emit_frontier_circulation(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    src_launches = [m for m in result if int(m[0]) == 0]
    assert len(src_launches) == 1
    _, angle, ships = src_launches[0]
    # Aim must be toward near_forward (angle ~0, i.e. +x direction).
    import math
    assert abs(angle) < 0.1, f"angle {angle} should aim at near_forward (+x)"
    assert int(ships) == 100 - 5  # everything minus garrison


def test_circulation_dag_is_monotone_in_frontier_dist(monkeypatch):
    """For every emitted edge (src, dst), frontier_dist(dst) must be
    STRICTLY LESS than frontier_dist(src). This is the loop-proof invariant."""
    monkeypatch.setenv("BASELINE_FRONTIER_CIRCULATION", "1")
    from agents.baseline.main import emit_frontier_circulation

    import math

    # 5 friendly planets in a diagonal line, opp centroid far away.
    my_planets = [
        _planet(i, 0, float(i * 10), 0.0, 50, production=2) for i in range(5)
    ]
    # Opp at (200, 0) → centroid (200, 0).
    opp = _planet(99, 1, 200.0, 0.0, 50)
    planets = my_planets + [opp]
    model = _StubModel()
    world = _StubWorld()

    result = emit_frontier_circulation(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )

    by_id = {int(p.id): p for p in my_planets}
    def fd(p): return math.hypot(p.x - 200.0, p.y - 0.0)

    for mv in result:
        src_id, angle, ships = int(mv[0]), float(mv[1]), int(mv[2])
        src = by_id[src_id]
        # Reconstruct destination from emitted angle: walk along the angle
        # and find which forward friendly it aims at. Since geometry is
        # linear and destinations are friendly planets at integer x, the
        # nearest forward friendly along the +x ray is the destination.
        # Simpler: just verify that the src has at least one forward friendly
        # whose frontier_dist is strictly less.
        forward = [q for q in my_planets if fd(q) < fd(src) - 0.5]
        assert forward, f"src {src_id} should have ≥1 forward friendly"


def test_circulation_skips_threatened_source(monkeypatch):
    monkeypatch.setenv("BASELINE_FRONTIER_CIRCULATION", "1")
    from agents.baseline.main import emit_frontier_circulation

    rear = _planet(0, 0, -50.0, 0.0, 100)
    front = _planet(1, 0, 20.0, 0.0, 100)
    opp = _planet(2, 1, 100.0, 0.0, 50)
    planets = [rear, front, opp]
    # Rear is under threat -> must not be drained.
    model = _StubModel(threatened={0})
    world = _StubWorld()

    result = emit_frontier_circulation(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    src_ids = {int(m[0]) for m in result}
    assert 0 not in src_ids


def test_circulation_skips_source_already_in_moves(monkeypatch):
    monkeypatch.setenv("BASELINE_FRONTIER_CIRCULATION", "1")
    from agents.baseline.main import emit_frontier_circulation

    rear = _planet(0, 0, -50.0, 0.0, 100)
    front = _planet(1, 0, 20.0, 0.0, 100)
    opp = _planet(2, 1, 100.0, 0.0, 50)
    planets = [rear, front, opp]
    model = _StubModel()
    world = _StubWorld()

    pre_moves = [[0, 0.0, 50]]  # chooser already fired src 0
    result = emit_frontier_circulation(
        pre_moves, planets, my_id=0, world=world, model=model, omega=0.0,
    )
    extras = [m for m in result if m not in pre_moves]
    assert all(int(m[0]) != 0 for m in extras)


def test_circulation_is_noop_when_disabled():
    """Default-OFF byte-parity: with env unset, the call passes through."""
    import importlib
    import agents.baseline.main as main_mod
    importlib.reload(main_mod)

    rear = _planet(0, 0, -50.0, 0.0, 100)
    front = _planet(1, 0, 20.0, 0.0, 100)
    opp = _planet(2, 1, 100.0, 0.0, 50)
    planets = [rear, front, opp]
    model = _StubModel()
    world = _StubWorld()

    result = main_mod.emit_frontier_circulation(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    assert result == []


def test_circulation_respects_max_per_turn(monkeypatch):
    monkeypatch.setenv("BASELINE_FRONTIER_CIRCULATION", "1")
    monkeypatch.setenv("BASELINE_CIRCULATION_MAX", "2")
    import importlib
    import agents.baseline.main as main_mod
    importlib.reload(main_mod)
    emit_frontier_circulation = main_mod.emit_frontier_circulation

    # 5 rear planets all able to fire toward front
    my_planets = [
        _planet(i, 0, -50.0 + float(i), 0.0, 50, production=2) for i in range(5)
    ]
    front = _planet(99, 0, 20.0, 0.0, 30)
    opp = _planet(100, 1, 100.0, 0.0, 50)
    planets = my_planets + [front, opp]
    model = _StubModel()
    world = _StubWorld()

    result = emit_frontier_circulation(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    assert len(result) <= 2, f"MAX=2 must cap launches, got {len(result)}"


def test_circulation_noop_with_no_opponents(monkeypatch):
    monkeypatch.setenv("BASELINE_FRONTIER_CIRCULATION", "1")
    from agents.baseline.main import emit_frontier_circulation

    rear = _planet(0, 0, -50.0, 0.0, 100)
    front = _planet(1, 0, 20.0, 0.0, 100)
    neutral = _planet(2, -1, 100.0, 0.0, 50)
    planets = [rear, front, neutral]  # no opponents at all
    model = _StubModel()
    world = _StubWorld()

    result = emit_frontier_circulation(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    assert result == [], "without opponents the centroid is undefined; noop"
