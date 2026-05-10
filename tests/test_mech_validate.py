"""Tests for lib/mechanism.validate — drops intents that violate constraints."""

from __future__ import annotations

from lib.intent import Intent, World
from lib.mechanism import validate


_OBS = {
    "player": 0,
    "planets": [
        [0, 0, 10.0, 10.0, 1.0, 50, 1],   # ours, garrison 50
        [1, 1, 30.0, 30.0, 1.0, 30, 1],   # enemy
        [2, -1, 80.0, 80.0, 1.0, 5, 1],   # neutral
    ],
    "angular_velocity": 0.0,
    "comet_planet_ids": [],
    "step": 1,
}


def _world():
    return World.from_obs(_OBS)


def test_validate_passes_through_legal_intent():
    intents = [Intent(src_id=0, target_id=2, ships=6)]
    out = validate(intents, _world())
    assert len(out) == 1
    assert out[0].target_id == 2


def test_validate_drops_unowned_source():
    intents = [Intent(src_id=1, target_id=2, ships=6)]   # 1 is enemy
    assert validate(intents, _world()) == []


def test_validate_drops_unknown_source():
    intents = [Intent(src_id=999, target_id=2, ships=6)]
    assert validate(intents, _world()) == []


def test_validate_drops_self_target():
    intents = [Intent(src_id=0, target_id=0, ships=6)]
    assert validate(intents, _world()) == []


def test_validate_drops_zero_or_negative_ships():
    intents = [
        Intent(src_id=0, target_id=2, ships=0),
        Intent(src_id=0, target_id=2, ships=-5),
    ]
    assert validate(intents, _world()) == []


def test_validate_drops_overcommit_against_garrison():
    intents = [Intent(src_id=0, target_id=2, ships=51)]   # garrison is 50
    assert validate(intents, _world()) == []


def test_validate_keeps_exact_full_garrison():
    intents = [Intent(src_id=0, target_id=2, ships=50)]   # garrison exactly
    assert len(validate(intents, _world())) == 1


def test_validate_preserves_multiple_legal_intents_in_order():
    intents = [
        Intent(src_id=0, target_id=2, ships=6),
        Intent(src_id=0, target_id=1, ships=10),
    ]
    out = validate(intents, _world())
    assert [i.target_id for i in out] == [2, 1]
