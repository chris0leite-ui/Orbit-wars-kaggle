"""Regression for arrival_size's targeted off-by-one fix on dynamic targets.

Backlog item A from `audit/2026-05-11-v3-snipe-games-analysis.md`. The
blanket `(eta + 1)` fix tried in v3.3 regressed 42.2% in 32-seed 2P
A/B because it over-sized static-target launches (whose ETA already
overshoots by (r_src + r_target)/v thanks to radius-entry capture).
The targeted fix applies the extra production tick ONLY to dynamic
targets (orbiting non-comet planets and comets) — which the swept-pair
collision resolves at the entry-turn position, exactly one prod-tick
later than the static formula models.

These tests assert:
- static targets: unchanged from pre-fix behaviour
- orbiting non-comet targets: +production × 1 ships extra
- comet targets: +production × 1 ships extra (even when not flagged as orbiting)
- the WorldModel branch still gates on max(static, model) and the
  static now includes the extra tick for dynamic targets
"""

from __future__ import annotations

import math

from agent import Intent, World, arrival_size, fleet_speed


def _world(planets, *, my_id=0, omega=0.05, comet_ids=()):
    obs = {
        "player": my_id,
        "planets": planets,
        "angular_velocity": omega,
        "comet_planet_ids": list(comet_ids),
        "step": 1,
    }
    return World.from_obs(obs)


def _src(garrison=9999):
    """Owned source at static rim — (5, 5) is outside the rotation limit."""
    return [0, 0, 5.0, 5.0, 1.0, garrison, 1]


def _static_enemy(planet_id, ships, production, x=90.0, y=5.0):
    """Static-ring planet (orbital_radius + radius >= 50)."""
    return [planet_id, 1, x, y, 1.0, ships, production]


def _orbiting_enemy(planet_id, ships, production, x=55.0, y=50.0):
    """Inner-orbit planet (orbital_radius + radius < 50).

    (55, 50) has orbital_radius = |55-50| + |50-50| / center_dist = 5;
    plus radius=1 → 6, well under the limit (50)."""
    return [planet_id, 1, x, y, 1.0, ships, production]


# ---------------------------------------------------------------------------
# Static target: unchanged behaviour
# ---------------------------------------------------------------------------


def test_static_enemy_target_no_extra_tick():
    """Distance 85, production=2, ships=20. Static target; eta = ceil(85/v),
    needed = 20 + 2*eta + 1 (NO extra tick)."""
    target = _static_enemy(1, ships=20, production=2)
    world = _world([_src(), target])
    intent = Intent(src_id=0, target_id=1, ships=21)
    out = arrival_size([intent], world)
    v = fleet_speed(21)
    eta = math.ceil(85.0 / v)
    expected_static = 20 + 2 * eta + 1
    assert out[0].ships == expected_static, (
        f"Static target should keep eta-only formula; got {out[0].ships}, "
        f"expected {expected_static}"
    )


# ---------------------------------------------------------------------------
# Orbiting non-comet target: +production × 1 extra tick
# ---------------------------------------------------------------------------


def test_orbiting_enemy_target_adds_one_extra_tick():
    """Orbiting planet at inner-radius — the swept-pair collision resolves
    at entry-turn position, one prod tick after the static formula."""
    src = _src()
    target = _orbiting_enemy(1, ships=20, production=2)
    world = _world([src, target])
    intent = Intent(src_id=0, target_id=1, ships=21)
    out = arrival_size([intent], world)
    d = math.hypot(target[2] - src[2], target[3] - src[3])
    v = fleet_speed(21)
    eta = math.ceil(d / v)
    expected_dynamic = 20 + 2 * (eta + 1) + 1
    assert out[0].ships == expected_dynamic, (
        f"Orbiting target should add one prod tick; got {out[0].ships}, "
        f"expected {expected_dynamic}"
    )


# ---------------------------------------------------------------------------
# Comet target: +production × 1 extra tick
# ---------------------------------------------------------------------------


def test_comet_target_adds_one_extra_tick():
    """Comet planet (flagged via comet_planet_ids), production=1, ships=15.
    Adds +1 prod tick regardless of orbital geometry."""
    src = _src()
    # Comet placed at static-ring coords — only the comet_ids flag matters
    target = _static_enemy(7, ships=15, production=1, x=80.0, y=20.0)
    world = _world([src, target], comet_ids=[7])
    intent = Intent(src_id=0, target_id=7, ships=16)
    out = arrival_size([intent], world)
    d = math.hypot(target[2] - src[2], target[3] - src[3])
    v = fleet_speed(16)
    eta = math.ceil(d / v)
    expected = 15 + 1 * (eta + 1) + 1
    assert out[0].ships == expected, (
        f"Comet should add one prod tick; got {out[0].ships}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Pass-throughs unchanged
# ---------------------------------------------------------------------------


def test_neutral_target_still_passes_through():
    """The whole bump-logic only applies to enemy targets."""
    target = [1, -1, 70.0, 5.0, 1.0, 20, 3]   # neutral
    world = _world([_src(), target])
    intent = Intent(src_id=0, target_id=1, ships=21)
    out = arrival_size([intent], world)
    assert out[0].ships == 21


def test_friendly_target_still_passes_through():
    target = [1, 0, 70.0, 5.0, 1.0, 20, 3]   # ours
    world = _world([_src(), target])
    intent = Intent(src_id=0, target_id=1, ships=21)
    out = arrival_size([intent], world)
    assert out[0].ships == 21


# ---------------------------------------------------------------------------
# WorldModel branch — max(static-with-extra-tick, model) still respected
# ---------------------------------------------------------------------------


class _StubModel:
    def __init__(self, owners, ships):
        self._owners = owners
        self._ships = ships

    def owner_at(self, pid, step):
        return self._owners.get(pid)

    def ships_at(self, pid, step):
        return self._ships.get(pid)


def test_orbiting_target_with_model_takes_max_of_static_with_tick_and_model():
    """Model says 50 ships at arrival; static-with-tick says (20 + 2*(eta+1) + 1).
    Whichever is larger wins."""
    src = _src()
    target = _orbiting_enemy(1, ships=20, production=2)
    world = _world([src, target])
    model = _StubModel(owners={1: 1}, ships={1: 50.0})
    intent = Intent(src_id=0, target_id=1, ships=21)
    out = arrival_size([intent], world, model)
    d = math.hypot(target[2] - src[2], target[3] - src[3])
    v = fleet_speed(21)
    eta = math.ceil(d / v)
    static_with_tick = 20 + 2 * (eta + 1) + 1
    model_needed = 51
    expected = max(static_with_tick, model_needed)
    assert out[0].ships == expected
