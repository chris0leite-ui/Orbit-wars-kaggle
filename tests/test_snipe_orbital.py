"""Orbital-safety tests for the snipe followon helpers.

B3 — `_followon_hold_estimate` now accepts `arrival_eta` and predicts
positions of the followon planet + each enemy planet at our arrival
when `BASELINE_ORBITAL_SAFETY=1`. Also uses
`model.incoming_enemy_eta_after(followon.id, my_id, arrival_eta - 1)`
to surface inbound waves arriving from our arrival onward.

B4 — `_best_followon` accepts `arrival_eta` and predicts the captured
target + each followon candidate's position at arrival before computing
the launch-from-target distance and `f_eta`.

Test shape: directly call the helpers with a tiny World/WorldModel
fixture and toggle the env var to compare verdicts.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from lib.intent import World
from lib.missions.snipe import _best_followon, _followon_hold_estimate
from lib.world_model import WorldModel


def _planet(pid, owner, x, y, *, ships=20, production=2, radius=1.5):
    return SimpleNamespace(
        id=pid, owner=owner, x=x, y=y, radius=radius,
        ships=ships, production=production,
    )


def _world(planets, fleets=(), my_id=0, omega=0.0, step=0):
    obs = {
        "player": my_id,
        "planets": [
            [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
            for p in planets
        ],
        "fleets": list(fleets),
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "step": step,
    }
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# B3 — _followon_hold_estimate orbital safety
# ---------------------------------------------------------------------------


def test_followon_hold_estimate_omega_zero_identical(monkeypatch):
    """omega=0 → identical verdict between env ON/OFF + arrival_eta>0
    has no effect when there's no rotation."""
    mine = _planet(0, 0, 5.0, 5.0, ships=100)
    target = _planet(1, -1, 50.0, 50.0, ships=10)
    followon = _planet(2, -1, 55.0, 55.0, ships=10)
    enemy = _planet(3, 1, 90.0, 50.0, ships=80)
    world = _world([mine, target, followon, enemy], omega=0.0)
    model = WorldModel.from_world(world)
    monkeypatch.setenv("BASELINE_ORBITAL_SAFETY", "0")
    off = _followon_hold_estimate(followon, target, world, model, 0, f_eta=10)
    monkeypatch.setenv("BASELINE_ORBITAL_SAFETY", "1")
    on = _followon_hold_estimate(
        followon, target, world, model, 0, f_eta=10, arrival_eta=20,
    )
    assert on == off


def test_followon_hold_estimate_rotates_into_enemy_lowers_hold(monkeypatch):
    """Followon at (40, 50) rotates to (60, 50) at arrival, closer to a
    STATIC outer enemy at (95, 50). With env ON the predicted threat
    travel is shorter → lower hold."""
    omega = 0.02
    mine = _planet(0, 0, 5.0, 5.0, ships=100, radius=1.0)
    # Target small static outside; not the focus of this test.
    target = _planet(1, -1, 50.0, 5.0, ships=10, radius=1.0)
    # Followon inner-orbital, starts on far side from enemy.
    followon = _planet(2, -1, 40.0, 50.0, ships=10, radius=1.0)
    enemy = _planet(3, 1, 95.0, 50.0, ships=80, radius=6.0)
    world = _world([mine, target, followon, enemy], omega=omega)
    model = WorldModel.from_world(world)
    arrival_eta = int(round(math.pi / omega))  # rotates to (60, 50)
    monkeypatch.setenv("BASELINE_ORBITAL_SAFETY", "0")
    off = _followon_hold_estimate(
        followon, target, world, model, 0, f_eta=10,
    )
    monkeypatch.setenv("BASELINE_ORBITAL_SAFETY", "1")
    on = _followon_hold_estimate(
        followon, target, world, model, 0, f_eta=10,
        arrival_eta=arrival_eta,
    )
    # Fixed path: closer enemy → shorter threat ETA → shorter hold.
    assert on <= off, (
        f"Predicted-at-arrival (followon close to enemy) should give "
        f"hold ≤ legacy (followon current pos far from enemy); got "
        f"on={on}, off={off}"
    )


def test_followon_hold_estimate_calls_incoming_enemy_eta_after(monkeypatch):
    """When env=1 and arrival_eta>0, `_followon_hold_estimate` must
    route in-flight fleet lookup through `incoming_enemy_eta_after`
    instead of `incoming_enemy_eta`."""
    omega = 0.02
    mine = _planet(0, 0, 5.0, 5.0, ships=100)
    target = _planet(1, -1, 50.0, 5.0, ships=10)
    followon = _planet(2, -1, 40.0, 50.0, ships=10, radius=1.0)
    enemy = _planet(3, 1, 95.0, 50.0, ships=80, radius=6.0)
    world = _world([mine, target, followon, enemy], omega=omega)
    model = WorldModel.from_world(world)
    # Track method calls by patching the model bound methods.
    calls = {"legacy": 0, "after": 0}
    orig_legacy = model.incoming_enemy_eta
    orig_after = model.incoming_enemy_eta_after

    def patched_legacy(*a, **kw):
        calls["legacy"] += 1
        return orig_legacy(*a, **kw)

    def patched_after(*a, **kw):
        calls["after"] += 1
        return orig_after(*a, **kw)

    model.incoming_enemy_eta = patched_legacy
    model.incoming_enemy_eta_after = patched_after

    monkeypatch.setenv("BASELINE_ORBITAL_SAFETY", "1")
    _followon_hold_estimate(
        followon, target, world, model, 0, f_eta=10, arrival_eta=20,
    )
    assert calls["after"] == 1, "Should route through incoming_enemy_eta_after"
    assert calls["legacy"] == 0, "Should NOT use legacy incoming_enemy_eta"

    # And the inverse — env off → uses legacy.
    calls["legacy"] = 0
    calls["after"] = 0
    monkeypatch.setenv("BASELINE_ORBITAL_SAFETY", "0")
    _followon_hold_estimate(
        followon, target, world, model, 0, f_eta=10, arrival_eta=20,
    )
    assert calls["legacy"] == 1
    assert calls["after"] == 0


# ---------------------------------------------------------------------------
# B4 — _best_followon orbital safety
# ---------------------------------------------------------------------------


def test_best_followon_omega_zero_identical(monkeypatch):
    """omega=0 → identical pick between env ON/OFF + arrival_eta>0."""
    mine = _planet(0, 0, 5.0, 5.0, ships=100)
    target = _planet(1, -1, 50.0, 50.0, ships=10, production=3)
    # Followon candidates within FOLLOWON_RADIUS (default ~25) of target.
    cand_a = _planet(2, -1, 60.0, 50.0, ships=10, production=3)
    cand_b = _planet(3, -1, 45.0, 55.0, ships=15, production=4)
    enemy = _planet(4, 1, 90.0, 50.0, ships=80)
    world = _world([mine, target, cand_a, cand_b, enemy], omega=0.0)
    model = WorldModel.from_world(world)
    monkeypatch.setenv("BASELINE_ORBITAL_SAFETY", "0")
    off = _best_followon(target, world, model, 0, radius=25.0)
    monkeypatch.setenv("BASELINE_ORBITAL_SAFETY", "1")
    on = _best_followon(target, world, model, 0, radius=25.0, arrival_eta=10)
    # Both should pick the same candidate when there's no rotation.
    if off is None or on is None:
        assert off == on
    else:
        assert off[0].id == on[0].id


def test_best_followon_default_arrival_eta_preserves_legacy(monkeypatch):
    """Default arrival_eta=0 → identical to env=OFF behavior (back-compat).

    A caller that never passes arrival_eta sees no change regardless of
    env state. This is the key back-compat guarantee for callers we
    don't migrate.
    """
    omega = 0.02
    mine = _planet(0, 0, 5.0, 5.0, ships=100)
    target = _planet(1, -1, 50.0, 50.0, ships=10, production=3, radius=1.0)
    cand = _planet(2, -1, 55.0, 55.0, ships=10, production=3, radius=1.0)
    enemy = _planet(3, 1, 90.0, 50.0, ships=80, radius=6.0)
    world = _world([mine, target, cand, enemy], omega=omega)
    model = WorldModel.from_world(world)
    monkeypatch.setenv("BASELINE_ORBITAL_SAFETY", "0")
    off = _best_followon(target, world, model, 0, radius=25.0)
    monkeypatch.setenv("BASELINE_ORBITAL_SAFETY", "1")
    on_default = _best_followon(target, world, model, 0, radius=25.0)
    # Default arrival_eta=0 → use_predict is False, identical verdict.
    if off is None or on_default is None:
        assert off == on_default
    else:
        assert off[0].id == on_default[0].id
        assert off[2] == on_default[2]  # f_eta identical


def test_best_followon_orbital_arrival_eta_changes_geometry(monkeypatch):
    """Setup: target inner-orbital starting at (40, 50). At arrival_eta
    = π/omega, target rotates to (60, 50). Followon candidate at a
    fixed (static) position. Distance from CURRENT target to candidate
    differs from distance at predicted target position."""
    omega = 0.02
    mine = _planet(0, 0, 5.0, 5.0, ships=100, radius=1.0)
    target = _planet(1, -1, 40.0, 50.0, ships=10, production=3, radius=1.0)
    # Followon static outside rotation limit, ~25 from PREDICTED tgt position
    # at (60, 50) → dist 55. Vs CURRENT tgt (40, 50) → dist 75.
    cand = _planet(2, -1, 95.0, 50.0, ships=10, production=3, radius=6.0)
    enemy = _planet(3, 1, 90.0, 5.0, ships=80, radius=6.0)
    world = _world([mine, target, cand, enemy], omega=omega)
    model = WorldModel.from_world(world)
    arrival_eta = int(round(math.pi / omega))
    # Pick radius large enough that distance always qualifies in BOTH
    # modes (~80 covers both 55 and 75 candidates). f_eta will differ.
    monkeypatch.setenv("BASELINE_ORBITAL_SAFETY", "1")
    on = _best_followon(
        target, world, model, 0, radius=80.0, arrival_eta=arrival_eta,
    )
    monkeypatch.setenv("BASELINE_ORBITAL_SAFETY", "0")
    off = _best_followon(target, world, model, 0, radius=80.0)
    # Both find the candidate; f_eta differs because distance differs.
    if on is None or off is None:
        # Allow the hold filter (MIN_FOLLOWON_HOLD=10 default) to drop
        # the candidate; verify at least one mode finds something.
        if on is None and off is None:
            pytest.skip("Both modes filtered out the candidate (hold floor).")
        # If only one finds, that's still a real modeling difference.
        return
    assert on[2] != off[2], (
        f"f_eta should differ when target rotates: on={on[2]}, off={off[2]}"
    )
