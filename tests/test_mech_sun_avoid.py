"""Tests for lib/mechanism.sun_avoid — drop intents whose path crosses the sun.

The drop-only contract: if `path_clears_sun(src, target, safety=1.0)` is
False, the intent is silently dropped (ships stay in garrison). Aimed
intents whose path is clear pass through unchanged. Unaimed intents
pass through (sun_avoid is order-tolerant — runs before or after lead_aim
but only acts when aim is set).
"""

from __future__ import annotations

import math

from agent import CENTER, SUN_RADIUS, Intent, World, sun_avoid


def _world(planets, *, my_id=0):
    obs = {
        "player": my_id,
        "planets": planets,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": 1,
    }
    return World.from_obs(obs)


def _src_at(x, y, garrison=99):
    return [0, 0, x, y, 1.0, garrison, 1]


def _target_at(pid, x, y):
    return [pid, -1, x, y, 1.0, 5, 1]


# ---------------------------------------------------------------------------
# Pass-through cases
# ---------------------------------------------------------------------------


def test_sun_avoid_passes_unaimed_intent_unchanged():
    src = _src_at(0.0, 0.0)
    target = _target_at(1, 100.0, 100.0)   # would cross sun if aimed
    world = _world([src, target])
    intent = Intent(src_id=0, target_id=1, ships=10, aim_angle=None)
    out = sun_avoid([intent], world)
    assert len(out) == 1
    assert out[0] is intent


def test_sun_avoid_passes_clear_path_through():
    """Path along the left edge: 50 units from sun centre, well beyond the
    10-radius + 1 safety margin. Pass through unchanged."""
    src = _src_at(0.0, 0.0)
    target = _target_at(1, 0.0, 100.0)
    world = _world([src, target])
    angle = math.atan2(100.0, 0.0)
    intent = Intent(src_id=0, target_id=1, ships=10, aim_angle=angle)
    out = sun_avoid([intent], world)
    assert len(out) == 1
    assert out[0].aim_angle == angle


def test_sun_avoid_passes_intent_when_src_or_target_unknown():
    src = _src_at(0.0, 0.0)
    world = _world([src])
    intent = Intent(src_id=0, target_id=999, ships=10, aim_angle=0.0)
    out = sun_avoid([intent], world)
    assert len(out) == 1   # defensive pass-through; emission filter handles it


# ---------------------------------------------------------------------------
# Drop cases
# ---------------------------------------------------------------------------


def test_sun_avoid_drops_diagonal_through_sun():
    """Corner-to-corner path passes straight through the sun at (50, 50)."""
    src = _src_at(0.0, 0.0)
    target = _target_at(1, 100.0, 100.0)
    world = _world([src, target])
    angle = math.atan2(100.0, 100.0)
    intent = Intent(src_id=0, target_id=1, ships=10, aim_angle=angle)
    out = sun_avoid([intent], world)
    assert out == []


def test_sun_avoid_drops_path_grazing_safety_margin():
    """Path that grazes within 1 unit of the sun edge. With safety=1.0, this
    is too close — drop."""
    # Path at x=60 (10 units perpendicular distance from sun centre = exact sun radius).
    src = _src_at(60.0, 0.0)
    target = _target_at(1, 60.0, 100.0)
    world = _world([src, target])
    intent = Intent(src_id=0, target_id=1, ships=10, aim_angle=math.pi / 2)
    out = sun_avoid([intent], world)
    assert out == []   # tangent to sun + 1-unit safety = still inside


def test_sun_avoid_keeps_path_just_outside_safety_margin():
    """11 units away from sun centre = clears 10 + 1 safety. Pass."""
    # Move past 11.5 to clearly clear safety=1.0.
    src = _src_at(CENTER + SUN_RADIUS + 1.5, 0.0)
    target = _target_at(1, CENTER + SUN_RADIUS + 1.5, 100.0)
    world = _world([src, target])
    intent = Intent(src_id=0, target_id=1, ships=10, aim_angle=math.pi / 2)
    out = sun_avoid([intent], world)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Mixed batches
# ---------------------------------------------------------------------------


def test_sun_avoid_drops_only_sun_blocked_in_mixed_batch():
    src = _src_at(0.0, 0.0)
    safe_target = _target_at(1, 0.0, 100.0)
    blocked_target = _target_at(2, 100.0, 100.0)
    world = _world([src, safe_target, blocked_target])
    safe_intent = Intent(
        src_id=0, target_id=1, ships=10,
        aim_angle=math.atan2(100.0, 0.0), note="safe",
    )
    blocked_intent = Intent(
        src_id=0, target_id=2, ships=10,
        aim_angle=math.atan2(100.0, 100.0), note="blocked",
    )
    out = sun_avoid([safe_intent, blocked_intent], world)
    assert [i.note for i in out] == ["safe"]
