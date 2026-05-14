"""H11: assert that `propose_opening_missions` is wired into v7_0's
incumbent mission set via `_build_incumbent_intents` in `lib/v7_search.py`.

The opening proposer is fully built (see `tests/test_mission_opening.py`)
but until this wire-up it wasn't actually called during agent execution.
The wire-up is a 1-line addition; the test exists to prevent silent
regressions on later refactors of `_build_incumbent_intents`.
"""

from __future__ import annotations

from lib.intent import World
from lib.v7_search import _build_incumbent_intents
from lib.world_model import WorldModel


def _planet(pid, owner, ships, prod=2, x=50.0, y=50.0, radius=1.5):
    return [pid, owner, x, y, radius, ships, prod]


def _world(planets, *, my_id=0, step=0):
    obs = {
        "player": my_id,
        "planets": planets,
        "angular_velocity": 0.05,
        "comet_planet_ids": [],
        "step": step,
        "comets": [],
        "fleets": [],
    }
    return World.from_obs(obs)


def test_opening_mission_chosen_at_step_3():
    """At step 3 with a 20-ship home planet and a reachable neutral,
    settle_plan should pick the opening mission for our source."""
    planets = [
        _planet(0, owner=0, ships=20, prod=2, x=10.0, y=10.0),
        _planet(1, owner=-1, ships=5, prod=3, x=40.0, y=10.0),
    ]
    world = _world(planets, step=3)
    model = WorldModel.from_world(world)
    intents = _build_incumbent_intents(world, model)
    # Exactly one source → one intent.
    assert len(intents) == 1, f"expected 1 intent, got {len(intents)}"
    intent = intents[0]
    assert intent.src_id == 0
    assert intent.target_id == 1


def test_opening_mission_not_chosen_after_window():
    """At step 6 (past OPENING_WINDOW=5), the opening proposer emits
    nothing and the snipe proposer takes over. We don't assert which
    mission class wins — only that v7_0 still produces a launch for the
    source if a viable target exists."""
    planets = [
        _planet(0, owner=0, ships=20, prod=2, x=10.0, y=10.0),
        _planet(1, owner=-1, ships=5, prod=3, x=40.0, y=10.0),
    ]
    world = _world(planets, step=6)
    model = WorldModel.from_world(world)
    intents = _build_incumbent_intents(world, model)
    # Snipe proposer still finds the target. The class is no longer
    # "opening" but the launch direction is unchanged.
    assert len(intents) >= 1
    assert intents[0].src_id == 0
    assert intents[0].target_id == 1


def test_no_intent_when_no_source_in_opening_window():
    """Empty board → no intents."""
    world = _world(planets=[], step=0)
    model = WorldModel.from_world(world)
    intents = _build_incumbent_intents(world, model)
    assert intents == []


def test_opening_class_label_present_in_intent_note():
    """The Intent's `note` field carries the originating mission class
    so downstream telemetry can attribute idle reasons. After H11 wire,
    intents from `propose_opening_missions` carry note='opening'."""
    planets = [
        _planet(0, owner=0, ships=20, prod=2, x=10.0, y=10.0),
        _planet(1, owner=-1, ships=5, prod=3, x=40.0, y=10.0),
    ]
    world = _world(planets, step=2)
    model = WorldModel.from_world(world)
    intents = _build_incumbent_intents(world, model)
    assert len(intents) == 1
    # Mission.to_intent() puts the mission_class into note; assert it.
    assert intents[0].note == "opening", (
        f"expected note='opening', got {intents[0].note!r}"
    )
