"""Tests for lib/mission.Mission — dataclass + to_intent boundary."""

from __future__ import annotations

from agent import Intent
from agent import Mission


def test_mission_to_intent_preserves_core_fields():
    m = Mission(
        mission_class="snipe", src_id=3, target_id=7, ships=12,
        score=0.42, eta=8,
    )
    i = m.to_intent()
    assert isinstance(i, Intent)
    assert i.src_id == 3
    assert i.target_id == 7
    assert i.ships == 12
    # Defaults from Intent — planner output shouldn't pre-fill aim.
    assert i.aim_angle is None
    assert i.arrival_xy is None


def test_mission_to_intent_note_carries_class():
    m = Mission(
        mission_class="snipe", src_id=0, target_id=1, ships=5,
        score=0.1, eta=4, note="rng=0.42",
    )
    i = m.to_intent()
    # Class is encoded in the note so downstream telemetry / debug can
    # see which mission class fired.
    assert "snipe" in i.note
    assert "rng=0.42" in i.note


def test_mission_to_intent_note_when_empty_uses_class_only():
    m = Mission(
        mission_class="snipe", src_id=0, target_id=1, ships=5,
        score=0.1, eta=4,
    )
    i = m.to_intent()
    assert i.note == "snipe"
