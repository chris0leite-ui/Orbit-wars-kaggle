"""Tests for lib/intent.py — Intent dataclass + World view + realize() pipeline."""

from __future__ import annotations

import pytest

from lib.intent import Intent, World, realize


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------


def test_intent_defaults_aim_angle_to_none():
    i = Intent(src_id=0, target_id=1, ships=10)
    assert i.aim_angle is None
    assert i.note == ""


def test_intent_is_mutable_so_mechanisms_can_set_aim():
    i = Intent(src_id=0, target_id=1, ships=10)
    i.aim_angle = 0.5
    assert i.aim_angle == 0.5


# ---------------------------------------------------------------------------
# World.from_obs — supports dict-shaped and Struct-shaped obs
# ---------------------------------------------------------------------------


class _ObsStruct:
    """Mimic kaggle_environments.utils.Struct: attribute access on a dict."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


_PLANET_TUPLE = [3, 0, 25.0, 25.0, 1.5, 50, 2]   # owned by player 0


def test_world_from_obs_handles_dict():
    obs = {
        "player": 0,
        "planets": [_PLANET_TUPLE],
        "angular_velocity": 0.04,
        "comet_planet_ids": [7, 8],
        "step": 42,
    }
    w = World.from_obs(obs)
    assert w.my_id == 0
    assert w.omega == pytest.approx(0.04)
    assert w.comet_ids == frozenset({7, 8})
    assert w.step == 42
    assert 3 in w.planets_by_id
    assert w.planets_by_id[3].owner == 0
    assert w.planets_by_id[3].ships == 50


def test_world_from_obs_handles_attribute_style():
    obs = _ObsStruct(
        player=1,
        planets=[_PLANET_TUPLE],
        angular_velocity=0.03,
        comet_planet_ids=[],
        step=10,
    )
    w = World.from_obs(obs)
    assert w.my_id == 1
    assert w.comet_ids == frozenset()
    assert w.step == 10


def test_world_from_obs_defaults_when_step_missing():
    obs = {"player": 0, "planets": [], "angular_velocity": 0.04}
    w = World.from_obs(obs)
    assert w.step == 0


# ---------------------------------------------------------------------------
# realize() pipeline + emission
# ---------------------------------------------------------------------------


def _passthrough(intents, world):
    return intents


def _set_angle_05(intents, world):
    for i in intents:
        i.aim_angle = 0.5
    return intents


def _drop_all(intents, world):
    return []


def _bump_ships(intents, world):
    for i in intents:
        i.ships += 5
    return intents


_OBS_TWO_PLANETS = {
    "player": 0,
    "planets": [
        [0, 0, 10.0, 10.0, 1.0, 99, 1],   # ours
        [1, -1, 30.0, 30.0, 1.0, 5, 1],    # neutral target
    ],
    "angular_velocity": 0.0,
    "comet_planet_ids": [],
    "step": 1,
}


def test_realize_runs_mechanisms_in_given_order():
    """`_bump_ships` then `_set_angle_05` should both fire; final action has
    bumped ships and angle 0.5."""
    intents = [Intent(src_id=0, target_id=1, ships=10)]
    actions = realize(intents, _OBS_TWO_PLANETS, mechanisms=[_bump_ships, _set_angle_05])
    assert actions == [[0, 0.5, 15]]


def test_realize_drops_intents_without_aim_angle():
    """If no mechanism populated aim_angle, the intent is silently dropped at
    emission — matches the validate-first contract."""
    intents = [Intent(src_id=0, target_id=1, ships=10)]
    actions = realize(intents, _OBS_TWO_PLANETS, mechanisms=[_passthrough])
    assert actions == []


def test_realize_drops_intents_with_zero_or_negative_ships():
    intents = [
        Intent(src_id=0, target_id=1, ships=0, aim_angle=0.0),
        Intent(src_id=0, target_id=1, ships=-5, aim_angle=0.0),
    ]
    actions = realize(intents, _OBS_TWO_PLANETS, mechanisms=[_passthrough])
    assert actions == []


def test_realize_handles_empty_intent_list():
    actions = realize([], _OBS_TWO_PLANETS, mechanisms=[_passthrough])
    assert actions == []


def test_realize_handles_mechanism_that_drops_everything():
    intents = [Intent(src_id=0, target_id=1, ships=10, aim_angle=0.0)]
    actions = realize(intents, _OBS_TWO_PLANETS, mechanisms=[_drop_all])
    assert actions == []


# ---------------------------------------------------------------------------
# reasons out-param — MECHANISM_DROP attribution (Phase 0 instrumentation)
# ---------------------------------------------------------------------------


def _drop_by_src(target_src):
    def _fn(intents, world):
        return [i for i in intents if i.src_id != target_src]
    _fn.__name__ = f"_drop_by_src_{target_src}"
    return _fn


def test_realize_reasons_default_none_no_change_in_behaviour():
    """Omitting `reasons` keeps actions identical to baseline."""
    intents = [Intent(src_id=0, target_id=1, ships=10, aim_angle=0.0)]
    actions_no_reasons = realize(intents, _OBS_TWO_PLANETS, mechanisms=[_passthrough])
    actions_with_reasons = realize(
        [Intent(src_id=0, target_id=1, ships=10, aim_angle=0.0)],
        _OBS_TWO_PLANETS,
        mechanisms=[_passthrough],
        reasons={},
    )
    assert actions_no_reasons == actions_with_reasons


def test_realize_reasons_attributes_drop_to_mechanism_name():
    """A mechanism that drops src_id=0 should appear in reasons with its
    function name in the value."""
    intents = [
        Intent(src_id=0, target_id=1, ships=10, aim_angle=0.0),
        Intent(src_id=2, target_id=1, ships=5, aim_angle=0.0),
    ]
    reasons: dict[int, str] = {}
    actions = realize(
        intents,
        _OBS_TWO_PLANETS,
        mechanisms=[_drop_by_src(0)],
        reasons=reasons,
    )
    # Only src 2 survives.
    assert actions == [[2, 0.0, 5]]
    assert 0 in reasons
    assert reasons[0].startswith("MECHANISM_DROP:_drop_by_src_0")
    assert 2 not in reasons


def test_realize_reasons_marks_final_emit_no_aim():
    """Intent that exits the pipeline with aim_angle=None is dropped at
    final emit; reasons should record `final_emit_no_aim`."""
    intents = [Intent(src_id=7, target_id=1, ships=10)]
    reasons: dict[int, str] = {}
    actions = realize(
        intents, _OBS_TWO_PLANETS, mechanisms=[_passthrough], reasons=reasons,
    )
    assert actions == []
    assert reasons.get(7) == "MECHANISM_DROP:final_emit_no_aim"


def test_realize_reasons_marks_final_emit_zero_ships():
    """Intent with ships=0 after pipeline → final_emit_zero_ships."""
    intents = [Intent(src_id=9, target_id=1, ships=0, aim_angle=0.1)]
    reasons: dict[int, str] = {}
    actions = realize(
        intents, _OBS_TWO_PLANETS, mechanisms=[_passthrough], reasons=reasons,
    )
    assert actions == []
    assert reasons.get(9) == "MECHANISM_DROP:final_emit_zero_ships"


def test_realize_reasons_first_dropping_mechanism_wins_attribution():
    """When a chain of mechanisms could each drop a src, only the FIRST
    drop is recorded (later mechanisms can't see the src to drop it)."""
    intents = [Intent(src_id=3, target_id=1, ships=10, aim_angle=0.0)]
    reasons: dict[int, str] = {}
    realize(
        intents,
        _OBS_TWO_PLANETS,
        mechanisms=[_drop_by_src(3), _drop_all],
        reasons=reasons,
    )
    assert reasons[3] == "MECHANISM_DROP:_drop_by_src_3"
