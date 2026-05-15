"""Step 1 — Foundation skeleton: Strategy ABC, Memory ABC, Action types.

Tests:
- Registry round-trip (register / get / list / clear)
- `ActionSpec` ↔ `ActionTensor` padding round-trip preserves data
- Bounds checks raise `ValueError` cleanly
- Dummy strategy passes memory through across two calls (threading
  semantics — no in-place mutation)
- `EmptyMemory` is a true no-op and satisfies the `Memory` protocol
"""

from __future__ import annotations

import math

import pytest

from lib.foundation import (
    ActionSpec,
    EmptyMemory,
    Memory,
    Strategy,
    StrategyCtx,
    clear_registry,
    get_strategy,
    list_strategies,
    register_strategy,
    specs_to_tensor,
    tensor_to_specs,
)


# -- Registry --------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


class _DummyStrategy:
    name = "dummy"

    def emit(self, state, my_id, ctx, memory):
        return specs_to_tensor([[]], horizon=1), memory


def test_register_and_get():
    s = _DummyStrategy()
    register_strategy("dummy", s)
    assert get_strategy("dummy") is s


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="not registered"):
        get_strategy("never_registered")


def test_list_strategies_sorted():
    register_strategy("zeta", _DummyStrategy())
    register_strategy("alpha", _DummyStrategy())
    register_strategy("mu", _DummyStrategy())
    assert list_strategies() == ["alpha", "mu", "zeta"]


def test_register_overwrites():
    a = _DummyStrategy()
    b = _DummyStrategy()
    register_strategy("dummy", a)
    register_strategy("dummy", b)
    assert get_strategy("dummy") is b


def test_clear_registry():
    register_strategy("dummy", _DummyStrategy())
    assert list_strategies() == ["dummy"]
    clear_registry()
    assert list_strategies() == []


# -- ActionSpec ↔ ActionTensor round-trip ----------------------------------


def test_empty_candidate_round_trip():
    tensor = specs_to_tensor([[]], horizon=5)
    # (C, T, A, L) = (1, 5, MAX_AGENTS=4, MAX_LAUNCH=20)
    assert tensor.pids.shape == (1, 5, 4, 20)
    assert (tensor.pids == -1).all()
    assert (tensor.ships == 0).all()

    back = tensor_to_specs(tensor)
    assert back == [[]]


def _specs_equal(a: ActionSpec, b: ActionSpec, angle_tol: float = 1e-5) -> bool:
    """Structural equality with float32-tolerance on `dir_angle` — the
    tensor stores angles in float32, so a 1e-7 drift is expected on
    round-trip."""
    return (
        a.from_planet_id == b.from_planet_id
        and a.ships == b.ships
        and a.launch_turn == b.launch_turn
        and a.agent_id == b.agent_id
        and math.isclose(a.dir_angle, b.dir_angle, abs_tol=angle_tol)
    )


def test_single_launch_round_trip():
    specs = [
        [
            ActionSpec(
                from_planet_id=7, dir_angle=1.234, ships=42,
                launch_turn=0, agent_id=0,
            )
        ]
    ]
    tensor = specs_to_tensor(specs, horizon=3)
    back = tensor_to_specs(tensor)
    assert len(back) == 1 and len(back[0]) == 1
    assert _specs_equal(back[0][0], specs[0][0])


def test_multi_turn_multi_agent_round_trip():
    specs_one_candidate = [
        ActionSpec(from_planet_id=1, dir_angle=0.5, ships=10, launch_turn=0, agent_id=0),
        ActionSpec(from_planet_id=2, dir_angle=1.5, ships=20, launch_turn=0, agent_id=0),
        ActionSpec(from_planet_id=3, dir_angle=2.5, ships=30, launch_turn=2, agent_id=1),
    ]
    tensor = specs_to_tensor([specs_one_candidate], horizon=4)
    back = tensor_to_specs(tensor)
    # Sort both by (turn, agent, pid) for comparison (slot order is
    # deterministic but irrelevant for content).
    def _key(s):
        return (s.launch_turn, s.agent_id, s.from_planet_id)

    assert sorted(back[0], key=_key) == sorted(specs_one_candidate, key=_key)


def test_specs_to_tensor_rejects_out_of_horizon():
    with pytest.raises(ValueError, match="launch_turn"):
        specs_to_tensor(
            [[ActionSpec(from_planet_id=1, dir_angle=0.0, ships=1, launch_turn=10)]],
            horizon=5,
        )


def test_specs_to_tensor_rejects_bad_agent_id():
    with pytest.raises(ValueError, match="agent_id"):
        specs_to_tensor(
            [[
                ActionSpec(
                    from_planet_id=1, dir_angle=0.0, ships=1,
                    launch_turn=0, agent_id=99,
                )
            ]],
            horizon=1,
        )


def test_specs_to_tensor_rejects_overflow():
    # MAX_LAUNCH_PER_AGENT = 20; try 21 launches at the same (turn, agent).
    too_many = [
        ActionSpec(
            from_planet_id=i, dir_angle=0.0, ships=1,
            launch_turn=0, agent_id=0,
        )
        for i in range(21)
    ]
    with pytest.raises(ValueError, match="more than"):
        specs_to_tensor([too_many], horizon=1)


def test_specs_to_tensor_rejects_empty_candidates():
    with pytest.raises(ValueError, match="at least one"):
        specs_to_tensor([], horizon=1)


def test_specs_to_tensor_rejects_zero_horizon():
    with pytest.raises(ValueError, match="horizon"):
        specs_to_tensor([[]], horizon=0)


def test_multi_candidate_independent():
    specs_a = [
        ActionSpec(from_planet_id=1, dir_angle=0.0, ships=10, launch_turn=0)
    ]
    specs_b = [
        ActionSpec(from_planet_id=2, dir_angle=math.pi, ships=20, launch_turn=1)
    ]
    tensor = specs_to_tensor([specs_a, specs_b], horizon=3)
    assert tensor.shape == (2, 3, 4, 20)
    back = tensor_to_specs(tensor)
    assert len(back) == 2
    assert len(back[0]) == 1 and _specs_equal(back[0][0], specs_a[0])
    assert len(back[1]) == 1 and _specs_equal(back[1][0], specs_b[0])


# -- Memory ----------------------------------------------------------------


def test_empty_memory_update_is_identity():
    m = EmptyMemory()
    assert m.update(state=None, action_taken=None) is m


def test_empty_memory_reset_returns_empty():
    m = EmptyMemory()
    fresh = m.reset()
    assert isinstance(fresh, EmptyMemory)


def test_empty_memory_protocol_conformance():
    m = EmptyMemory()
    assert isinstance(m, Memory)


# -- Strategy threading memory --------------------------------------------


class _CountingMemory:
    """Memory that counts updates; for the threading test."""

    def __init__(self, count: int = 0):
        self.count = count

    def update(self, state, action_taken):
        return _CountingMemory(self.count + 1)

    def reset(self):
        return _CountingMemory(0)


class _ThreadingStrategy:
    """Strategy that explicitly calls `memory.update` inside `emit`,
    so we can verify the `new_memory` it returns reflects the update."""

    name = "threading"

    def emit(self, state, my_id, ctx, memory):
        new_mem = memory.update(state, action_taken="noop")
        return specs_to_tensor([[]], horizon=1), new_mem


def test_memory_threads_across_calls():
    strat = _ThreadingStrategy()
    ctx = StrategyCtx()
    mem = _CountingMemory(count=0)

    _, mem1 = strat.emit(state=None, my_id=0, ctx=ctx, memory=mem)
    assert mem1.count == 1

    _, mem2 = strat.emit(state=None, my_id=0, ctx=ctx, memory=mem1)
    assert mem2.count == 2

    # Original memory not mutated.
    assert mem.count == 0


def test_counting_memory_protocol_conformance():
    assert isinstance(_CountingMemory(), Memory)


def test_strategy_protocol_conformance():
    strat = _ThreadingStrategy()
    assert isinstance(strat, Strategy)


def test_dummy_strategy_protocol_conformance():
    strat = _DummyStrategy()
    assert isinstance(strat, Strategy)


# -- StrategyCtx -----------------------------------------------------------


def test_strategy_ctx_defaults():
    ctx = StrategyCtx()
    assert ctx.turn_budget_ms == 1000.0
    assert ctx.rng_key is None
    assert ctx.world_model is None


def test_strategy_ctx_is_frozen():
    ctx = StrategyCtx()
    with pytest.raises(Exception):  # FrozenInstanceError, dataclass-frozen
        ctx.turn_budget_ms = 500.0  # type: ignore[misc]
