"""Rule-38 pin tests for `lib/missions/snipe.py::_followon_hold_estimate`
orbital arrival safety (sibling fix to commit f1774a7).

Bug: the function used the followon's CURRENT (x, y) and each enemy planet's
CURRENT (x, y) to compute threat distances at a future capture tick `f_eta`.
For ORBITING followons that rotate INTO enemy territory by `f_eta`, the
distance was wrong → hold horizon was falsely long → snipe ROI scored
follow-on captures we'd immediately lose as attractive.

Fix: when `BASELINE_ORBITAL_SAFETY=1` and the board rotates, predict both
followon and enemy positions at `f_eta` via `lib.orbit.predict_relative`.
Pre-arrival inbound enemy fleets filtered (combat resolution at our capture
handles those). Default OFF preserves submitted-bundle parity.
"""

from __future__ import annotations

import math
import os

import pytest

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.geometry import CENTER
from lib.intent import World
from lib.missions.snipe import _followon_hold_estimate
from lib.world_model import WorldModel


def _world(my_id, planets, *, omega=0.04, step=0, fleets=None):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": fleets or [],
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "step": step,
    }
    return World.from_obs(obs)


def _model(planet_ids):
    return WorldModel(
        ledger={pid: [] for pid in planet_ids},
        timelines={},
        horizon=200,
    )


def _polar_planet(pid, owner, *, orb_r, theta, ships=10, production=2,
                  radius=1.0):
    """Place an orbiting planet at polar (orb_r, theta). orb_r+radius must
    be < ROTATION_RADIUS_LIMIT=50 for is_orbiting=True."""
    x = CENTER + orb_r * math.cos(theta)
    y = CENTER + orb_r * math.sin(theta)
    return Planet(pid, owner, x, y, radius, ships, production)


@pytest.fixture(autouse=True)
def _clear_env():
    """Ensure BASELINE_ORBITAL_SAFETY starts from a known state per test."""
    prev = os.environ.pop("BASELINE_ORBITAL_SAFETY", None)
    yield
    if prev is None:
        os.environ.pop("BASELINE_ORBITAL_SAFETY", None)
    else:
        os.environ["BASELINE_ORBITAL_SAFETY"] = prev


# ---------------------------------------------------------------------------
# Fixture 1 — orbital drift changes the hold estimate when the gate is ON.
#
# Setup:
#   - `target` (P0, opp) — about-to-be-captured forward base. Far from
#     followon and threats; doesn't matter much.
#   - `followon` (P1, opp → soon ours) — ORBITING at theta=0; rotates
#     ~45° counter-clockwise over 20 ticks. Capture at f_eta=20.
#   - `enemy` (P2, opp) — STATIC, placed near P1's CURRENT position so
#     pre-fix sees a SHORT threat distance (bug: short hold) and post-fix
#     sees a LONG threat distance (followon has rotated away).
# ---------------------------------------------------------------------------

def test_followon_hold_estimate_rotates_followon_when_gate_on():
    """With `BASELINE_ORBITAL_SAFETY=1`, the hold estimate must DIFFER
    from the gate-OFF result when followon rotates relative to enemy.

    The gated branch in `_followon_hold_estimate` is the only path where
    `predict_relative` is invoked; this test pins that the branch is
    actually reached and changes the answer.
    """
    target = Planet(0, 1, 5.0, 5.0, 1.0, 10, 2)
    # followon: orbiting at (CENTER+20, CENTER), theta=0.
    followon = _polar_planet(1, 1, orb_r=20.0, theta=0.0)
    # enemy: static near followon's current position. orb_r+radius=51 ≥ 50
    # → not orbiting.
    enemy = Planet(2, 1, CENTER + 49.0, CENTER, 2.0, 50, 2)

    world = _world(my_id=0, planets=[target, followon, enemy], omega=0.04)
    model = _model([0, 1, 2])

    # Gate OFF (default) — uses current positions.
    os.environ.pop("BASELINE_ORBITAL_SAFETY", None)
    hold_off = _followon_hold_estimate(followon, target, world, model,
                                       my_id=0, f_eta=20)
    # Gate ON — predicts followon + enemy at f_eta=20.
    os.environ["BASELINE_ORBITAL_SAFETY"] = "1"
    hold_on = _followon_hold_estimate(followon, target, world, model,
                                      my_id=0, f_eta=20)

    assert hold_off != hold_on, (
        f"BASELINE_ORBITAL_SAFETY did not change the hold estimate — "
        f"gated branch never executed. hold_off={hold_off}, hold_on={hold_on}"
    )


# ---------------------------------------------------------------------------
# Fixture 2 — static board (omega=0): gate is a no-op.
# ---------------------------------------------------------------------------

def test_followon_hold_estimate_static_board_gate_noop():
    """On a non-rotating board (omega=0.0), the gate must NOT affect the
    hold estimate. Locks the "no orbital prediction when not needed"
    contract."""
    target = Planet(0, 1, 5.0, 5.0, 1.0, 10, 2)
    followon = _polar_planet(1, 1, orb_r=20.0, theta=0.0)
    enemy = Planet(2, 1, CENTER + 49.0, CENTER, 2.0, 50, 2)

    world = _world(my_id=0, planets=[target, followon, enemy], omega=0.0)
    model = _model([0, 1, 2])

    os.environ.pop("BASELINE_ORBITAL_SAFETY", None)
    hold_off = _followon_hold_estimate(followon, target, world, model,
                                       my_id=0, f_eta=20)
    os.environ["BASELINE_ORBITAL_SAFETY"] = "1"
    hold_on = _followon_hold_estimate(followon, target, world, model,
                                      my_id=0, f_eta=20)

    assert hold_off == hold_on, (
        f"gate changed hold estimate on non-rotating board (omega=0); "
        f"expected identical. hold_off={hold_off}, hold_on={hold_on}"
    )


# ---------------------------------------------------------------------------
# Fixture 3 — target planet is excluded from threat set under both modes.
#
# The docstring promises `target` (the about-to-be-captured forward base)
# is excluded from enemy threats. Verify both modes honour that.
# ---------------------------------------------------------------------------

def test_followon_hold_estimate_target_excluded_from_threats_both_modes():
    """Whether the gate is ON or OFF, the `target` planet (which we are
    about to flip to our side) must not contribute to the followon's
    threat ETA. Locks the docstring contract."""
    # target is right next to followon — if it counted as a threat,
    # the hold estimate would be near-zero. The exclusion must keep
    # hold non-trivial.
    followon = _polar_planet(1, 1, orb_r=20.0, theta=0.0)
    target = Planet(0, 1, followon.x + 2.0, followon.y, 1.0, 5, 2)
    # A SECOND enemy farther away, so there IS a non-zero threat.
    far_enemy = Planet(2, 1, CENTER - 40.0, CENTER, 2.0, 50, 2)
    world = _world(my_id=0, planets=[target, followon, far_enemy], omega=0.0)
    model = _model([0, 1, 2])

    os.environ.pop("BASELINE_ORBITAL_SAFETY", None)
    hold_off = _followon_hold_estimate(followon, target, world, model,
                                       my_id=0, f_eta=10)
    os.environ["BASELINE_ORBITAL_SAFETY"] = "1"
    hold_on = _followon_hold_estimate(followon, target, world, model,
                                      my_id=0, f_eta=10)

    # If target had counted as a threat, hold would be ~0 (target is
    # 2 units from followon — trivial flight time). Both modes should
    # see a longer hold dominated by far_enemy.
    assert hold_off > 5, (
        f"OFF mode: target was wrongly counted as a threat; "
        f"hold={hold_off} suggests target's 2-unit distance dominated"
    )
    assert hold_on > 5, (
        f"ON mode: target was wrongly counted as a threat; hold={hold_on}"
    )
