"""Tests for lib/mechanism.lead_aim — orbit-aware aim population.

Behavioural parity with v1's `_aim_angle` is the load-bearing property; the
end-to-end check lives in tests/test_v1_parity.py. Here we cover the
algorithmic edge cases at the unit level.
"""

from __future__ import annotations

import math

import pytest

from agent import CENTER, Intent, World, lead_aim


def _make_obs(target_x, target_y, *, target_owner=-1, omega=0.04, comet_ids=()):
    """Single-source single-target obs with src far from sun; omega + comet
    list configurable."""
    return {
        "player": 0,
        "planets": [
            [0, 0, 5.0, 5.0, 1.0, 100, 1],     # ours
            [1, target_owner, target_x, target_y, 1.0, 30, 1],
        ],
        "angular_velocity": omega,
        "comet_planet_ids": list(comet_ids),
        "step": 0,
    }


def _world(obs):
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# Static targets and zero-omega: simple atan2
# ---------------------------------------------------------------------------


def test_lead_aim_static_target_uses_atan2_to_current_position():
    """Outer-radius planet → static; aim is exactly atan2 of current dx/dy."""
    obs = _make_obs(target_x=CENTER + 49.0, target_y=CENTER, omega=0.04)
    intent = Intent(src_id=0, target_id=1, ships=10)
    out = lead_aim([intent], _world(obs))
    expected = math.atan2(CENTER - 5.0, CENTER + 49.0 - 5.0)
    assert out[0].aim_angle == pytest.approx(expected)


def test_lead_aim_zero_omega_falls_through_to_current_position():
    """Even orbiting planets aim at current position when omega=0."""
    obs = _make_obs(target_x=CENTER + 10.0, target_y=CENTER, omega=0.0)
    intent = Intent(src_id=0, target_id=1, ships=10)
    out = lead_aim([intent], _world(obs))
    expected = math.atan2(CENTER - 5.0, CENTER + 10.0 - 5.0)
    assert out[0].aim_angle == pytest.approx(expected)


def test_lead_aim_comet_target_aims_at_current_position():
    """Comet IDs in `world.comet_ids` are intentionally not led here — that's
    `comet_aim`'s job (3.5.C). lead_aim should leave them at atan2 of current."""
    obs = _make_obs(
        target_x=CENTER + 10.0, target_y=CENTER, omega=0.04, comet_ids=[1],
    )
    intent = Intent(src_id=0, target_id=1, ships=10)
    out = lead_aim([intent], _world(obs))
    expected = math.atan2(CENTER - 5.0, CENTER + 10.0 - 5.0)
    assert out[0].aim_angle == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Orbiting targets: lead is non-zero and points "ahead" of current
# ---------------------------------------------------------------------------


def test_lead_aim_orbiting_target_has_nonzero_lead():
    obs = _make_obs(target_x=CENTER + 10.0, target_y=CENTER, omega=0.04)
    intent = Intent(src_id=0, target_id=1, ships=10)
    out = lead_aim([intent], _world(obs))
    no_lead = math.atan2(CENTER - 5.0, CENTER + 10.0 - 5.0)
    assert out[0].aim_angle != pytest.approx(no_lead)
    # Forward (omega > 0) lead points the fleet "ahead" — i.e., the angle has
    # changed by a small amount in the same direction as omega's rotation.
    assert abs(out[0].aim_angle - no_lead) > 1e-3


# ---------------------------------------------------------------------------
# Idempotency / no-op edges
# ---------------------------------------------------------------------------


def test_lead_aim_does_not_overwrite_existing_aim_angle():
    obs = _make_obs(target_x=CENTER + 10.0, target_y=CENTER, omega=0.04)
    intent = Intent(src_id=0, target_id=1, ships=10, aim_angle=1.234)
    out = lead_aim([intent], _world(obs))
    assert out[0].aim_angle == 1.234


def test_lead_aim_skips_intent_with_unknown_src_or_target():
    obs = _make_obs(target_x=CENTER + 10.0, target_y=CENTER, omega=0.04)
    intent = Intent(src_id=999, target_id=1, ships=10)
    out = lead_aim([intent], _world(obs))
    assert out[0].aim_angle is None


def test_lead_aim_returns_intents_in_input_order():
    obs = _make_obs(target_x=CENTER + 10.0, target_y=CENTER, omega=0.04)
    intents = [
        Intent(src_id=0, target_id=1, ships=10, note="a"),
        Intent(src_id=0, target_id=1, ships=20, note="b"),
    ]
    out = lead_aim(intents, _world(obs))
    assert [i.note for i in out] == ["a", "b"]
