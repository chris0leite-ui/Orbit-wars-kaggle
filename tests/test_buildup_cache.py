"""Cache behaviour tests for `agents.buildup_planner.buildup.step`.

The 2026-05-25 commit-and-execute refactor solves the MILP once per
game (first BUILDUP turn) and caches the schedule in the caller's
per-seat state. These tests pin down the contract:

  1. `opening_plan` is called exactly ONCE across 30 simulated turns.
  2. Cached entries are emitted at their `fire_step` and consumed.
  3. The source-ownership guard skips entries whose `src_id` was
     captured by the opponent between solve-time and fire-time.
  4. An empty planner result is cached as `[]` and not re-solved.
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch, MagicMock

from agents.buildup_planner import buildup as bp_buildup
from lib.joint_solver.opening_planner import ScheduleEntry


@dataclass
class _FakePlanet:
    id: int
    owner: int


class _FakeWorld:
    """Minimum surface buildup.step touches: `planets_by_id` lookup."""

    def __init__(self, owned_src_ids: list[int], me: int = 0):
        self.planets_by_id = {
            sid: _FakePlanet(id=sid, owner=me) for sid in owned_src_ids
        }

    def set_owner(self, src_id: int, owner: int) -> None:
        self.planets_by_id[src_id] = _FakePlanet(id=src_id, owner=owner)


def _entry(fire_step: int, src_id: int, ships: int = 10,
           angle: float = 0.5) -> ScheduleEntry:
    return ScheduleEntry(fire_step=fire_step, src_id=src_id, tgt_id=999,
                         ships=ships, angle=angle, eta=8, value=100.0)


def _fake_plan(schedule: list[ScheduleEntry]):
    plan = MagicMock()
    plan.schedule = schedule
    return plan


def test_buildup_caches_schedule_across_turns():
    """opening_plan must be called exactly ONCE across 30 simulated
    turns at the same game_id. Pre-cache code re-solved every turn."""
    schedule = [
        _entry(fire_step=3, src_id=8),
        _entry(fire_step=12, src_id=8),
        _entry(fire_step=22, src_id=8),
    ]
    world = _FakeWorld(owned_src_ids=[8])
    state = {"opening_schedule": None}

    with patch.object(bp_buildup, "opening_plan",
                      return_value=_fake_plan(schedule)) as m_plan:
        emitted_per_turn: dict[int, list] = {}
        for t in range(30):
            out = bp_buildup.step(world, model=None, me=0, num_seats=2,
                                  step_now=t, state=state)
            assert out is not None, f"unexpected None at step={t}"
            if out:
                emitted_per_turn[t] = out

    assert m_plan.call_count == 1, (
        f"expected 1 opening_plan call, got {m_plan.call_count}"
    )
    # Each scheduled fire_step emits exactly once, in turn-order.
    assert set(emitted_per_turn.keys()) == {3, 12, 22}
    assert len(emitted_per_turn[3]) == 1
    assert emitted_per_turn[3][0][0] == 8  # src_id


def test_buildup_skips_opp_captured_source():
    """Source-ownership guard: if the planned src_id has been captured
    by opp between solve-time and fire-time, the entry is silently
    dropped (matches env semantics; no exception)."""
    schedule = [
        _entry(fire_step=5, src_id=8),
        _entry(fire_step=15, src_id=9),
    ]
    world = _FakeWorld(owned_src_ids=[8, 9], me=0)
    state = {"opening_schedule": None}

    with patch.object(bp_buildup, "opening_plan",
                      return_value=_fake_plan(schedule)):
        # Step 0 triggers the cache solve.
        out0 = bp_buildup.step(world, model=None, me=0, num_seats=2,
                               step_now=0, state=state)
        assert out0 == []
        # Opp captures src=8 between t=0 and t=5.
        world.set_owner(8, owner=1)
        out5 = bp_buildup.step(world, model=None, me=0, num_seats=2,
                               step_now=5, state=state)
        # Entry for src=8 silently dropped; no exception, no emit.
        assert out5 == []
        # src=9 still owned, fires at t=15 as planned.
        out15 = bp_buildup.step(world, model=None, me=0, num_seats=2,
                                step_now=15, state=state)
        assert len(out15) == 1
        assert out15[0][0] == 9


def test_buildup_caches_empty_schedule_and_does_not_resolve():
    """If the planner returns an empty schedule on the first call, we
    cache `[]` and never call opening_plan again — same as the
    well-populated case."""
    world = _FakeWorld(owned_src_ids=[8])
    state = {"opening_schedule": None}

    with patch.object(bp_buildup, "opening_plan",
                      return_value=_fake_plan([])) as m_plan:
        for t in range(10):
            out = bp_buildup.step(world, model=None, me=0, num_seats=2,
                                  step_now=t, state=state)
            assert out == []
    assert m_plan.call_count == 1


def test_buildup_returns_none_past_opening_horizon():
    """Above OPENING_HORIZON the planner is never consulted; signal to
    main.py via None that the opening is exhausted."""
    from lib.joint_solver.opening_planner import OPENING_HORIZON
    world = _FakeWorld(owned_src_ids=[8])
    state = {"opening_schedule": None}

    with patch.object(bp_buildup, "opening_plan") as m_plan:
        out = bp_buildup.step(world, model=None, me=0, num_seats=2,
                              step_now=OPENING_HORIZON, state=state)
    assert out is None
    assert m_plan.call_count == 0
