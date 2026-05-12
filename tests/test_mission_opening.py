"""Opening-landgrab Mission class — fires steps 0-5 from owned planets
with > 8 ships toward best neutral target.

Top-10 fingerprint: median first-launch step 4.1 (midpack 10.5). This
test asserts the proposer:
1. Fires for step ∈ [0, 5] when conditions met
2. Does NOT fire for step > 5
3. Skips owned planets with ≤ 8 ships
4. Skips comet targets (none active in opening window anyway)
5. Skips targets the source can't capture at all (cost > src.ships)
6. Uses front-loaded value: production × (remaining)^1.5
7. Sizes launches to drain the source to OPENING_RESERVE ships
   (bowwowforeach #1 archetype, garrison-at-launch ~7.7 vs midpack 22)
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.intent import World
from lib.missions.opening import (
    EPISODE_STEPS,
    MIN_LAUNCH_GARRISON,
    OPENING_RESERVE,
    OPENING_WINDOW,
    propose_opening_missions,
)
from lib.world_model import WorldModel


def _world(planets, *, my_id=0, step=0, comet_ids=()):
    obs = {
        "player": my_id,
        "planets": planets,
        "angular_velocity": 0.05,
        "comet_planet_ids": list(comet_ids),
        "step": step,
        "comets": [],
        "fleets": [],
    }
    return World.from_obs(obs)


def _model(world):
    return WorldModel.from_world(world)


def _planet(pid, owner, ships, prod=1, x=50.0, y=50.0, radius=1.0):
    return [pid, owner, x, y, radius, ships, prod]


# ---------------------------------------------------------------------------
# Firing window
# ---------------------------------------------------------------------------


def test_fires_at_step_zero():
    planets = [
        _planet(0, owner=0, ships=20, prod=1, x=10.0, y=10.0),
        _planet(1, owner=-1, ships=5, prod=2, x=70.0, y=10.0),
    ]
    w = _world(planets, step=0)
    out = propose_opening_missions(w, _model(w))
    assert len(out) == 1
    assert out[0].mission_class == "opening"
    assert out[0].src_id == 0 and out[0].target_id == 1


def test_fires_through_step_5():
    planets = [
        _planet(0, owner=0, ships=20, x=10.0, y=10.0),
        _planet(1, owner=-1, ships=5, prod=2, x=70.0, y=10.0),
    ]
    for step in range(0, OPENING_WINDOW + 1):
        w = _world(planets, step=step)
        out = propose_opening_missions(w, _model(w))
        assert len(out) == 1, f"step {step}: expected 1 mission, got {len(out)}"


def test_does_not_fire_after_window():
    planets = [
        _planet(0, owner=0, ships=20, x=10.0, y=10.0),
        _planet(1, owner=-1, ships=5, prod=2, x=70.0, y=10.0),
    ]
    w = _world(planets, step=OPENING_WINDOW + 1)
    out = propose_opening_missions(w, _model(w))
    assert out == []


# ---------------------------------------------------------------------------
# Source/target filters
# ---------------------------------------------------------------------------


def test_skips_owned_planet_below_min_garrison():
    """8 ships is the boundary; only > 8 fires."""
    planets = [
        _planet(0, owner=0, ships=MIN_LAUNCH_GARRISON, x=10.0, y=10.0),
        _planet(1, owner=-1, ships=2, prod=2, x=70.0, y=10.0),
    ]
    w = _world(planets, step=0)
    out = propose_opening_missions(w, _model(w))
    assert out == []


def test_skips_comet_targets_in_opening_window():
    """Comets don't spawn at step 0, but defensively the proposer ignores
    them in case a synthetic test puts one there."""
    planets = [
        _planet(0, owner=0, ships=30, x=10.0, y=10.0),
        _planet(1, owner=-1, ships=2, prod=2, x=70.0, y=10.0),
    ]
    w = _world(planets, step=0, comet_ids=[1])
    out = propose_opening_missions(w, _model(w))
    assert out == []


def test_skips_uncapturable_target():
    """Source has 10 ships; target has 50 ships (would need 51 to capture)."""
    planets = [
        _planet(0, owner=0, ships=10, x=10.0, y=10.0),
        _planet(1, owner=-1, ships=50, prod=2, x=70.0, y=10.0),
    ]
    w = _world(planets, step=0)
    out = propose_opening_missions(w, _model(w))
    assert out == []


def test_drains_source_to_opening_reserve():
    """Bowwowforeach archetype: 10-ship home sends ~8 ships at step 0,
    leaving OPENING_RESERVE=2 garrison at the source."""
    planets = [
        _planet(0, owner=0, ships=10, x=10.0, y=10.0),
        _planet(1, owner=-1, ships=0, prod=2, x=30.0, y=10.0),  # easy neutral
    ]
    w = _world(planets, step=0)
    out = propose_opening_missions(w, _model(w))
    assert len(out) == 1
    assert out[0].ships == 10 - OPENING_RESERVE  # = 8
    # Sanity: post-launch garrison equals OPENING_RESERVE.
    assert 10 - out[0].ships == OPENING_RESERVE


def test_sizing_caps_at_source_garrison():
    """Source has 30 ships; target needs only 3 to capture.
    With drain-to-reserve sizing, we send 28 (not capped down to 3)."""
    planets = [
        _planet(0, owner=0, ships=30, x=10.0, y=10.0),
        _planet(1, owner=-1, ships=2, prod=2, x=30.0, y=10.0),
    ]
    w = _world(planets, step=0)
    out = propose_opening_missions(w, _model(w))
    assert len(out) == 1
    assert out[0].ships == 30 - OPENING_RESERVE  # = 28


def test_sizing_respects_target_min_when_larger():
    """Source has 20 ships, target needs 19 to capture (ships=18).
    target_min=19 > src-reserve=18, so we send 19 (capture cost wins)."""
    planets = [
        _planet(0, owner=0, ships=20, x=10.0, y=10.0),
        _planet(1, owner=-1, ships=18, prod=2, x=30.0, y=10.0),
    ]
    w = _world(planets, step=0)
    out = propose_opening_missions(w, _model(w))
    assert len(out) == 1
    assert out[0].ships == 19  # max(target_min=19, src-reserve=18)


def test_skips_enemy_target():
    planets = [
        _planet(0, owner=0, ships=30, x=10.0, y=10.0),
        _planet(1, owner=1, ships=2, prod=2, x=70.0, y=10.0),  # enemy
    ]
    w = _world(planets, step=0)
    out = propose_opening_missions(w, _model(w))
    assert out == []


def test_skips_our_own_target():
    planets = [
        _planet(0, owner=0, ships=30, x=10.0, y=10.0),
        _planet(1, owner=0, ships=2, prod=2, x=70.0, y=10.0),  # ours
    ]
    w = _world(planets, step=0)
    out = propose_opening_missions(w, _model(w))
    assert out == []


# ---------------------------------------------------------------------------
# Scoring math
# ---------------------------------------------------------------------------


def test_score_uses_front_loaded_value():
    """value = production × (500 - step - eta)^1.5; score = value / (d + 1).

    With bowwowforeach-style sizing (drain to OPENING_RESERVE), a 30-
    ship source sends max(target_min, 30 - 2) = 28 ships, not the
    minimum-viable 3.
    """
    planets = [
        _planet(0, owner=0, ships=30, x=10.0, y=10.0),
        _planet(1, owner=-1, ships=2, prod=3, x=70.0, y=10.0),
    ]
    w = _world(planets, step=0)
    out = propose_opening_missions(w, _model(w))
    assert len(out) == 1
    m = out[0]
    d = 60.0
    base_ships = max(3, 30 - OPENING_RESERVE)  # = 28
    assert m.ships == base_ships, f"expected ships={base_ships}, got {m.ships}"
    v = fleet_speed(base_ships)
    eta = int(math.ceil(d / v))
    remaining = EPISODE_STEPS - 0 - eta
    expected = (3.0 * remaining ** 1.5) / (d + 1.0)
    assert abs(m.score - expected) < 1e-6, f"got {m.score}, expected {expected}"


def test_score_prefers_nearer_target_at_equal_production():
    planets = [
        _planet(0, owner=0, ships=30, x=10.0, y=10.0),
        _planet(1, owner=-1, ships=2, prod=2, x=30.0, y=10.0),  # near
        _planet(2, owner=-1, ships=2, prod=2, x=80.0, y=10.0),  # far
    ]
    w = _world(planets, step=0)
    out = propose_opening_missions(w, _model(w))
    assert len(out) == 2
    near = next(m for m in out if m.target_id == 1)
    far = next(m for m in out if m.target_id == 2)
    assert near.score > far.score


def test_returns_one_mission_per_source_target_pair():
    """3 sources × 2 neutrals = 6 missions (settle_plan picks one per source)."""
    planets = [
        _planet(0, owner=0, ships=30, x=10.0, y=10.0),
        _planet(3, owner=0, ships=30, x=10.0, y=90.0),
        _planet(4, owner=0, ships=30, x=90.0, y=10.0),
        _planet(1, owner=-1, ships=2, prod=2, x=50.0, y=50.0),
        _planet(2, owner=-1, ships=2, prod=2, x=70.0, y=70.0),
    ]
    w = _world(planets, step=0)
    out = propose_opening_missions(w, _model(w))
    # 3 owned planets each have 2 candidate targets = 6 missions.
    # But (10,10) → (50,50) is 56.6; with ships=3 fleet speed is ~1.0,
    # eta ~57 — comfortably inside the 500-step horizon. All pairs survive.
    assert len(out) == 3 * 2
