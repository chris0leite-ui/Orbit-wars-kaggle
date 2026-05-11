"""Tests for lib/planner.settle_plan — per-source greedy + same-turn ledger.

v3.1 solver semantics:
- Exactly one Intent per source that produced at least one mission.
- Sources processed in priority order (highest top-mission score first).
- Each source's first non-over-committed mission wins.
- Two sources MAY pick the same target if a single source's contribution
  is insufficient (gang-up scenario).
- A target is "over-committed" when the cumulative this-turn arrivals by
  some eta already exceed the predicted enemy garrison + 1 buffer; the
  subsequent source falls back to its next candidate.
"""

from __future__ import annotations

from types import SimpleNamespace

from lib.intent import World
from lib.mission import Mission
from lib.planner import settle_plan
from lib.world_model import WorldModel


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    return SimpleNamespace(
        id=pid, owner=owner, x=x, y=y, radius=radius,
        ships=ships, production=production,
    )


def _world(my_id, planets):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": [],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": 0,
    }
    return World.from_obs(obs)


def _setup_world():
    world = _world(my_id=0, planets=[
        _planet(0, 0, 0.0, 0.0, ships=100),
        _planet(1, 0, 100.0, 0.0, ships=100),
        _planet(2, 1, 50.0, 0.0, ships=5),
        _planet(3, 1, 50.0, 50.0, ships=5),
    ])
    return world, WorldModel.from_world(world)


def test_empty_missions_returns_empty():
    world, model = _setup_world()
    assert settle_plan([], world, model) == []


def test_one_intent_per_source_top_score_wins():
    world, model = _setup_world()
    missions = [
        Mission("snipe", src_id=0, target_id=2, ships=6, score=0.95, eta=10),
        Mission("snipe", src_id=0, target_id=3, ships=6, score=0.40, eta=12),
        Mission("snipe", src_id=1, target_id=3, ships=6, score=0.80, eta=10),
    ]
    intents = settle_plan(missions, world, model)
    by_src = {i.src_id: i for i in intents}
    assert len(intents) == 2
    assert by_src[0].target_id == 2   # 0.95 > 0.40
    assert by_src[1].target_id == 3


def test_gangup_when_single_source_insufficient():
    """Two sources stack on the same heavily-defended target when one
    source's contribution alone wouldn't capture it. With target.ships=5
    + production=2, the predicted defender at eta=10 is ~25 ships — one
    6-ship fleet leaves ~19 defenders, so the second source's 6 ships
    is still useful (12 vs ~26 still doesn't capture, but the planner
    can't predict perfectly; it just refuses to over-commit)."""
    world, model = _setup_world()
    missions = [
        Mission("snipe", src_id=0, target_id=2, ships=6, score=0.95, eta=10),
        Mission("snipe", src_id=1, target_id=2, ships=6, score=0.90, eta=11),
    ]
    intents = settle_plan(missions, world, model)
    assert len(intents) == 2
    assert {i.target_id for i in intents} == {2}


def test_skip_overcommit_when_one_source_already_suffices():
    """Source 0 commits 50 ships to target 2 (5-ship garrison + 20 growth
    over 10 turns ≈ 25 defenders). The first 50-ship fleet alone is more
    than enough. Source 1 should NOT also commit 50 — instead it falls
    back to its second-best target."""
    world, model = _setup_world()
    missions = [
        # Source 0: 50 ships at target 2 → cumulative will exceed
        # pred_enemy + 1 (~26) by a wide margin.
        Mission("snipe", src_id=0, target_id=2, ships=50, score=0.95, eta=10),
        # Source 1 also wants target 2 with 50 ships. Should be skipped.
        Mission("snipe", src_id=1, target_id=2, ships=50, score=0.90, eta=11),
        # Source 1's fallback — target 3 (a different lightly-defended planet).
        Mission("snipe", src_id=1, target_id=3, ships=6, score=0.50, eta=12),
    ]
    intents = settle_plan(missions, world, model)
    by_src = {i.src_id: i for i in intents}
    assert by_src[0].target_id == 2
    # Source 1 fell back to target 3, NOT redundantly stacking on 2.
    assert by_src[1].target_id == 3


def test_source_with_no_fallback_drops_when_overcommitted():
    """When a source's ONLY candidate is already over-committed by an
    earlier pick, that source emits no intent (rather than firing a
    wasted fleet at an already-handled target)."""
    world, model = _setup_world()
    missions = [
        Mission("snipe", src_id=0, target_id=2, ships=50, score=0.95, eta=10),
        # Source 1's only candidate is the over-committed target.
        Mission("snipe", src_id=1, target_id=2, ships=50, score=0.90, eta=11),
    ]
    intents = settle_plan(missions, world, model)
    # Only source 0's intent survives; source 1 has no fallback.
    assert len(intents) == 1
    assert intents[0].src_id == 0


def test_intents_ordered_by_source_top_score_desc():
    world, model = _setup_world()
    missions = [
        Mission("snipe", src_id=0, target_id=2, ships=6, score=0.40, eta=10),
        Mission("snipe", src_id=1, target_id=3, ships=6, score=0.95, eta=10),
    ]
    intents = settle_plan(missions, world, model)
    # Source 1 has higher top score → its intent appears first.
    assert intents[0].src_id == 1
    assert intents[1].src_id == 0


def test_source_with_no_missions_produces_no_intent():
    world, model = _setup_world()
    missions = [
        Mission("snipe", src_id=0, target_id=2, ships=6, score=0.95, eta=10),
    ]
    intents = settle_plan(missions, world, model)
    assert len(intents) == 1
    assert intents[0].src_id == 0
