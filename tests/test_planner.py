"""Tests for lib/planner.settle_plan — per-source top-score selection.

v0 solver semantics:
- Exactly one Intent per source that produced at least one mission.
- Picks the mission with the highest `score` for that source.
- Returns Intents ordered by source-best-score descending.

No-double-commit / multi-source coordination is intentionally NOT enforced
in v0 — see lib/planner.py docstring. v3.1 gang_up will introduce
coordinated multi-source arrivals through a mission class, not a planner-
level filter.
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


def test_both_sources_may_pick_same_target():
    """v0 lets two sources concentrate on the same target — gang-up is the
    right play on dense boards; v3.1's gang_up class formalises it but
    nothing in v0 prevents it."""
    world, model = _setup_world()
    missions = [
        Mission("snipe", src_id=0, target_id=2, ships=6, score=0.95, eta=10),
        Mission("snipe", src_id=1, target_id=2, ships=6, score=0.90, eta=11),
    ]
    intents = settle_plan(missions, world, model)
    assert len(intents) == 2
    assert {i.target_id for i in intents} == {2}


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
