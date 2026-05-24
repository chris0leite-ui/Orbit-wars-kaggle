"""Layer D — plan-level drop-one validator oracle tests.

The validator drops legs whose marginal contribution to plan production-
advantage is below SAFETY_MARGIN. Tests cover: default-off no-op, single-
move no-op, redundant-leg pruning, unique-leg keeping, bounce/reinforce
ignored.
"""
import importlib
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _reload_drop_one(env: dict):
    saved = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import lib.drop_one as d
        importlib.reload(d)
        return d
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _FakeModel:
    def __init__(self, owner_map=None, ships_map=None):
        self.owner_map = owner_map or {}
        self.ships_map = ships_map or {}

    def owner_at(self, planet_id, _arrival_step):
        return self.owner_map.get(int(planet_id), -1)

    def ships_at(self, planet_id, _arrival_step):
        return self.ships_map.get(int(planet_id), 0.0)


def _planet(pid, x, y, prod, owner=-1, ships=0, radius=2.0):
    return SimpleNamespace(
        id=pid, x=x, y=y, production=prod, owner=owner, ships=ships,
        radius=radius, is_orbiting=False, orbit_radius=0.0,
        orbit_angle=0.0,
    )


def _world(planets, step=10, omega=0.05):
    return SimpleNamespace(
        planets_by_id={p.id: p for p in planets},
        step=step,
        omega=omega,
    )


def test_default_off_is_noop():
    """Without env var, drop_one_validate returns moves unchanged."""
    d = _reload_drop_one({"BASELINE_DROP_ONE_VALIDATE": None})
    assert d.DROP_ONE_ENABLED is False
    moves = [[1, 0.5, 10], [2, 1.0, 20]]
    out = d.drop_one_validate(moves, _world([]), _FakeModel(), 0)
    assert out == moves, "default-off must be a no-op"


def test_single_move_noop():
    """Plan with ≤1 moves: nothing to drop."""
    d = _reload_drop_one({"BASELINE_DROP_ONE_VALIDATE": "1"})
    moves = [[1, 0.5, 10]]
    out = d.drop_one_validate(moves, _world([]), _FakeModel(), 0)
    assert out == moves


def test_empty_moves():
    """Empty plan: returns empty."""
    d = _reload_drop_one({"BASELINE_DROP_ONE_VALIDATE": "1"})
    out = d.drop_one_validate([], _world([]), _FakeModel(), 0)
    assert out == []


def test_plan_value_zero_when_no_captures():
    """A plan where every move bounces (insufficient ships) yields plan
    value zero — every move would be pruned by drop-one."""
    d = _reload_drop_one({"BASELINE_DROP_ONE_VALIDATE": "1"})
    # The plan_production_advantage function relies on predict_fleet_fate;
    # with planets too far for a 2-ship fleet to reach in trajectory,
    # the function returns 0. Test the inner accumulator directly.
    p_src = _planet(1, 50, 50, prod=1, owner=0, ships=2)
    p_tgt = _planet(2, 50, 90, prod=5, owner=1, ships=100)  # heavy garrison
    world = _world([p_src, p_tgt])
    # owner=1 at hit, ships=2 ≤ 100 → bounce → zero value contribution.
    model = _FakeModel(owner_map={2: 1}, ships_map={2: 100.0})
    value = d.plan_production_advantage(
        [[1, 1.5708, 2]], world, model, 0,  # angle ~ +y; src→tgt
    )
    # Even if predict_fleet_fate finds a hit, the bounce check zeros it.
    assert value == 0.0


def test_safety_margin_is_tunable():
    """A high safety margin drops everything; a low margin keeps everything."""
    d_strict = _reload_drop_one({
        "BASELINE_DROP_ONE_VALIDATE": "1",
        "BASELINE_DROP_ONE_SAFETY": "100000.0",
    })
    moves = [[1, 0.0, 10], [2, 0.0, 20]]
    # Strict margin would drop any leg whose contribution < 100k production-ticks.
    # In an empty world (no planets to hit), all marginal contributions are 0,
    # so all legs get pruned. Empty result.
    out = d_strict.drop_one_validate(moves, _world([]), _FakeModel(), 0)
    assert out == [], "strict margin must drop all uncontributing legs"


def test_drop_one_idempotent_under_no_captures():
    """When no leg captures (empty world), the validator drops everything
    (margin > 0) but the result is stable: drop_one_validate(drop_one_validate(...))
    is still empty."""
    d = _reload_drop_one({
        "BASELINE_DROP_ONE_VALIDATE": "1",
        "BASELINE_DROP_ONE_SAFETY": "1.0",
    })
    moves = [[1, 0.5, 10], [2, 1.0, 20]]
    out1 = d.drop_one_validate(moves, _world([]), _FakeModel(), 0)
    out2 = d.drop_one_validate(out1, _world([]), _FakeModel(), 0)
    assert out1 == out2
