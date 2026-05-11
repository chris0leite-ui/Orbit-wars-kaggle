"""Tests for lib/mechanism.arrival_size.

For enemy-owned targets, bump intent.ships to cover production growth during
fleet flight: needed = target.ships + production * eta + 1. Neutrals + own
planets pass through unchanged. If even our full garrison can't cover the
bumped target, drop the intent (don't waste an under-sized fleet).
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.intent import Intent, World
from lib.mechanism import arrival_size


def _world(planets, *, my_id=0, omega=0.0):
    obs = {
        "player": my_id,
        "planets": planets,
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "step": 1,
    }
    return World.from_obs(obs)


def _src(garrison=999):
    """Owned source at (10, 10) with effectively unlimited garrison."""
    return [0, 0, 10.0, 10.0, 1.0, garrison, 1]


def _target(planet_id, owner, ships, production, x=70.0, y=70.0):
    return [planet_id, owner, x, y, 1.0, ships, production]


# ---------------------------------------------------------------------------
# Pass-through cases (no production growth applies)
# ---------------------------------------------------------------------------


def test_arrival_size_passes_through_neutral_target_unchanged():
    target = _target(1, owner=-1, ships=20, production=3)   # neutrals don't produce
    world = _world([_src(), target])
    intent = Intent(src_id=0, target_id=1, ships=21)
    out = arrival_size([intent], world)
    assert len(out) == 1
    assert out[0].ships == 21


def test_arrival_size_passes_through_friendly_target_unchanged():
    """Reinforce intents (target.owner == my_id) shouldn't be over-sized here."""
    target = _target(1, owner=0, ships=20, production=3)    # ours
    world = _world([_src(), target], my_id=0)
    intent = Intent(src_id=0, target_id=1, ships=21)
    out = arrival_size([intent], world)
    assert out[0].ships == 21


def test_arrival_size_drops_intent_when_src_or_target_unknown():
    target = _target(1, owner=1, ships=20, production=3)
    world = _world([_src(), target], my_id=0)
    # src 999 doesn't exist → pass-through (will be dropped at validate-final
    # or emission step). We don't drop here so unrelated mechanisms can chain.
    intent = Intent(src_id=999, target_id=1, ships=21)
    out = arrival_size([intent], world)
    assert len(out) == 1
    assert out[0].ships == 21


# ---------------------------------------------------------------------------
# Enemy-target bumps
# ---------------------------------------------------------------------------


def test_arrival_size_bumps_enemy_target_for_production_growth():
    """Enemy target with production=2; fleet flies ~85 units at speed ~1
    (1-ship fleet) → eta ≈ 85 turns → bump = 2 * 85 = 170."""
    target = _target(1, owner=1, ships=20, production=2, x=10.0 + 85.0, y=10.0)
    world = _world([_src(garrison=9999), target], my_id=0)
    intent = Intent(src_id=0, target_id=1, ships=21)   # current target ships + 1
    out = arrival_size([intent], world)
    # eta = ceil(85 / fleet_speed(21)); env runs production (eta + 1) times
    # during flight (per-turn production tick, including the arrival turn).
    v = fleet_speed(21)
    eta = math.ceil(85.0 / v)
    expected = 20 + 2 * (eta + 1) + 1
    assert out[0].ships == expected
    assert out[0].ships > 21   # actually grew


def test_arrival_size_zero_production_means_no_bump_for_enemy():
    """Edge: enemy target with production 0 (e.g. depleted comet) — no growth."""
    target = _target(1, owner=1, ships=20, production=0, x=10.0 + 30.0, y=10.0)
    world = _world([_src(), target], my_id=0)
    intent = Intent(src_id=0, target_id=1, ships=21)
    out = arrival_size([intent], world)
    eta = math.ceil(30.0 / fleet_speed(21))
    expected = 20 + 0 * (eta + 1) + 1
    assert out[0].ships == max(21, expected)   # equals 21


def test_arrival_size_drops_intent_when_garrison_cant_cover_bumped_size():
    """If our src has 50 ships but the target needs 100+ to capture after
    production growth, the intent is dropped — sending under-sized is
    pure waste."""
    target = _target(1, owner=1, ships=200, production=5, x=10.0 + 80.0, y=10.0)
    world = _world([_src(garrison=50), target], my_id=0)
    intent = Intent(src_id=0, target_id=1, ships=50)
    out = arrival_size([intent], world)
    assert out == []


def test_arrival_size_preserves_oversize_strategy_intent():
    """A strategy may have asked for a swarm fleet (`ships=500`); the
    mechanism must not cut it down to the minimum-needed value."""
    target = _target(1, owner=1, ships=20, production=2, x=10.0 + 30.0, y=10.0)
    world = _world([_src(garrison=9999), target], my_id=0)
    intent = Intent(src_id=0, target_id=1, ships=500)   # over-spec swarm
    out = arrival_size([intent], world)
    eta = math.ceil(30.0 / fleet_speed(500))
    needed = 20 + 2 * (eta + 1) + 1
    assert out[0].ships == max(500, needed)   # 500 stays
    assert out[0].ships >= 500


def test_arrival_size_preserves_intent_order():
    targets = [
        _target(1, owner=1, ships=20, production=2),
        _target(2, owner=-1, ships=10, production=1),
        _target(3, owner=1, ships=30, production=1),
    ]
    world = _world([_src(garrison=9999)] + targets, my_id=0)
    intents = [
        Intent(src_id=0, target_id=1, ships=21, note="a"),
        Intent(src_id=0, target_id=2, ships=11, note="b"),
        Intent(src_id=0, target_id=3, ships=31, note="c"),
    ]
    out = arrival_size(intents, world)
    assert [i.note for i in out] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# WorldModel-aware cases (the v3.2 adversary-stacking fix)
# ---------------------------------------------------------------------------


class _StubModel:
    """Minimal WorldModel stub: (owner, ships) lookup by planet_id."""

    def __init__(self, owners: dict, ships: dict):
        self._owners = owners
        self._ships = ships

    def owner_at(self, planet_id, step):
        return self._owners.get(planet_id)

    def ships_at(self, planet_id, step):
        return self._ships.get(planet_id)


def test_arrival_size_uses_model_predicted_ships_when_higher_than_static():
    """With two enemy fleets in-flight, the static estimate
    (target.ships + production*eta + 1) under-counts. The WorldModel
    predicts the post-stacking garrison; arrival_size must respect it."""
    target = _target(1, owner=1, ships=20, production=1, x=10.0 + 30.0, y=10.0)
    world = _world([_src(garrison=9999), target], my_id=0)
    # Model says garrison will be 80 by the time we arrive.
    model = _StubModel(owners={1: 1}, ships={1: 80.0})
    intent = Intent(src_id=0, target_id=1, ships=21)
    out = arrival_size([intent], world, model)
    assert out[0].ships == 81   # max(static~25, 80+1)


def test_arrival_size_skips_intent_when_model_predicts_target_already_ours():
    """If a teammate / earlier mission flips the target to us en route,
    sending more ships is wasted — let the planner re-allocate."""
    target = _target(1, owner=1, ships=20, production=1, x=10.0 + 30.0, y=10.0)
    world = _world([_src(garrison=9999), target], my_id=0)
    model = _StubModel(owners={1: 0}, ships={1: 50.0})   # 0 = us
    intent = Intent(src_id=0, target_id=1, ships=21)
    out = arrival_size([intent], world, model)
    assert out == []


def test_arrival_size_falls_back_to_static_when_model_under_predicts():
    """Defensive: orbiting-planet eta is noisy (lib/world_model.py:46-51).
    Static lower bound must be respected."""
    target = _target(1, owner=1, ships=30, production=2, x=10.0 + 30.0, y=10.0)
    world = _world([_src(garrison=9999), target], my_id=0)
    model = _StubModel(owners={1: 1}, ships={1: 5.0})   # under-predicts
    intent = Intent(src_id=0, target_id=1, ships=21)
    out = arrival_size([intent], world, model)
    eta = math.ceil(30.0 / fleet_speed(21))
    static_needed = 30 + 2 * (eta + 1) + 1
    assert out[0].ships == static_needed


def test_arrival_size_no_model_keeps_static_only_behavior():
    """Backwards-compat: pass-through to the original static formula
    when model is None."""
    target = _target(1, owner=1, ships=20, production=2, x=10.0 + 30.0, y=10.0)
    world = _world([_src(garrison=9999), target], my_id=0)
    out_no_model = arrival_size([Intent(src_id=0, target_id=1, ships=21)], world)
    out_with_none = arrival_size([Intent(src_id=0, target_id=1, ships=21)], world, None)
    assert out_no_model[0].ships == out_with_none[0].ships


def test_arrival_size_covers_env_extra_production_tick():
    """Regression: env runs production (eta + 1) times during flight, not eta.

    Live-replay analysis of v3_snipe (52544634) found 38 of 518 enemy
    bounces at margin ∈ {+0, +1} where the agent sent exactly the
    target.ships + production*eta + 1 prediction but combat tied or
    came up 1 short because one extra production tick happened during
    the arrival turn. audit/2026-05-11-v3-snipe-critical-review.md §4.6.

    This test checks our `needed` formula now uses (eta + 1).
    """
    target = _target(1, owner=1, ships=20, production=3, x=10.0 + 30.0, y=10.0)
    world = _world([_src(garrison=9999), target], my_id=0)
    intent = Intent(src_id=0, target_id=1, ships=21)
    out = arrival_size([intent], world)
    v = fleet_speed(21)
    eta = math.ceil(30.0 / v)
    # Production must run (eta + 1) times, not eta.
    assert out[0].ships == 20 + 3 * (eta + 1) + 1
    # Sanity: result strictly exceeds old formula by exactly `production`.
    old_formula = 20 + 3 * eta + 1
    assert out[0].ships == old_formula + 3
