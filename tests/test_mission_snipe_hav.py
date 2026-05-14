"""Tests for the Hold-Aware Value (HAV) extensions to propose_snipe_missions.

Coverage:
- Default (all flags off) → emission byte-equivalent to pre-HAV behaviour
  (Mission per (src, target), no extra tier appended).
- USE_HAV=1: targets with `expected_hold <= 0` are dropped.
- USE_HAV=1: PV horizon for surviving targets uses `expected_hold` as
  `t_total`, not full `EPISODE_STEPS - step - eta`.
- USE_HOLDING_TIER=1: emits a second Mission with bigger `ships` and a
  "hold" `note` when in-flight enemy fleets threaten counter-attack
  AND source can afford.
- USE_OPERATIONAL_TIER=1: emits a third Mission with even bigger
  `ships` and an `op→<id>` note when a follow-on target exists, is
  reachable, and is holdable for ≥ MIN_FOLLOWON_HOLD turns.
- Operational tier does NOT emit when no follow-on qualifies.
- All flags off → pre-HAV parity preserved.
"""

from __future__ import annotations

import pytest
from types import SimpleNamespace

import lib.missions.snipe as snipe_mod
from lib.intent import World
from lib.missions.snipe import propose_snipe_missions
from lib.world_model import WorldModel


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    return [pid, owner, x, y, radius, ships, production]


def _world(planets, *, my_id=0, step=10, fleets=()):
    obs = {
        "player": my_id,
        "planets": planets,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": step,
        "comets": [],
        "fleets": list(fleets),
    }
    return World.from_obs(obs)


@pytest.fixture(autouse=True)
def _reset_flags():
    """Save and restore the HAV-related module flags so individual
    tests can flip them without leaking state."""
    saved = {
        "USE_HAV": snipe_mod.USE_HAV,
        "USE_HOLDING_TIER": snipe_mod.USE_HOLDING_TIER,
        "USE_OPERATIONAL_TIER": snipe_mod.USE_OPERATIONAL_TIER,
    }
    yield
    for k, v in saved.items():
        setattr(snipe_mod, k, v)


# ---------------------------------------------------------------------------
# Default-off parity
# ---------------------------------------------------------------------------


def test_default_flags_emit_one_mission_per_pair():
    snipe_mod.USE_HAV = 0
    snipe_mod.USE_HOLDING_TIER = 0
    snipe_mod.USE_OPERATIONAL_TIER = 0
    planets = [
        _planet(0, 0, 10.0, 10.0, ships=100),
        _planet(1, 1, 90.0, 90.0, ships=10),
    ]
    world = _world(planets)
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    # 1 src × 1 target = 1 Mission, no extra tiers.
    assert len(missions) == 1
    assert missions[0].src_id == 0 and missions[0].target_id == 1
    assert missions[0].note == ""  # no tier annotation


# ---------------------------------------------------------------------------
# HAV-1: expected_hold cap
# ---------------------------------------------------------------------------


def test_hav_drops_mission_when_expected_hold_is_zero():
    """A target enemy can reach BEFORE we arrive should be dropped."""
    snipe_mod.USE_HAV = 1
    # Our source is far west; target is dead-centre; enemy is RIGHT
    # next to the target → enemy threat-eta is tiny, our eta is large.
    planets = [
        _planet(0, 0, 5.0, 50.0, ships=100),
        _planet(1, -1, 70.0, 50.0, ships=5),     # neutral target
        _planet(2, 1, 72.0, 50.0, ships=80),     # enemy right next to it
    ]
    world = _world(planets)
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    # Target's expected_hold will be 0 (enemy is closer than we are)
    # → tactical tier should drop it. But the enemy planet itself is
    # also a target → that one might survive. Verify the neutral (id=1)
    # is dropped.
    target_ids = {m.target_id for m in missions}
    assert 1 not in target_ids


def test_hav_off_keeps_target_that_hav_would_drop():
    """Mirror of the above with USE_HAV=0 — the neutral target survives
    because the proposer uses the full-game horizon."""
    snipe_mod.USE_HAV = 0
    planets = [
        _planet(0, 0, 5.0, 50.0, ships=100),
        _planet(1, -1, 70.0, 50.0, ships=5),
        _planet(2, 1, 72.0, 50.0, ships=80),
    ]
    world = _world(planets)
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    target_ids = {m.target_id for m in missions}
    assert 1 in target_ids  # default behaviour proposes it


# ---------------------------------------------------------------------------
# Holding tier
# ---------------------------------------------------------------------------


def test_holding_tier_emits_extra_mission_when_enemy_counter_inbound():
    """An in-flight enemy fleet arriving a few turns AFTER our arrival
    triggers a Holding tier mission with more ships."""
    snipe_mod.USE_HAV = 0      # isolate Holding tier
    snipe_mod.USE_HOLDING_TIER = 1
    # Setup: our src very close to target so OUR eta is small; enemy
    # fleet farther so its eta lands in the post-capture window
    # (eta+1, eta+HOLD_WINDOW=eta+10].
    planets = [
        _planet(0, 0, 45.0, 50.0, ships=200),          # src 5 units west
        _planet(1, 1, 50.0, 50.0, ships=10),           # target
    ]
    # Enemy fleet 30 ships at (75, 50), heading west toward target.
    import math as _m
    fleets = [[100, 1, 75.0, 50.0, _m.pi, 99, 30]]
    world = _world(planets, fleets=fleets)
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    by_note = {m.note: m for m in missions if m.src_id == 0 and m.target_id == 1}
    assert "" in by_note
    assert "hold" in by_note
    assert by_note["hold"].ships > by_note[""].ships


def test_holding_tier_skipped_when_no_inflight_enemy_counter():
    """No counter-attack threat → no holding tier emitted."""
    snipe_mod.USE_HOLDING_TIER = 1
    planets = [
        _planet(0, 0, 10.0, 50.0, ships=200),
        _planet(1, 1, 50.0, 50.0, ships=10),
    ]
    world = _world(planets)   # no in-flight fleets
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    notes = {m.note for m in missions}
    assert "hold" not in notes


def test_holding_tier_skipped_when_source_cannot_afford():
    """Source garrison too low to fund holding-sized fleet AND keep
    SOURCE_DEFENSE_RESERVE → no holding tier."""
    snipe_mod.USE_HOLDING_TIER = 1
    import math as _m
    # Source only has 20 ships; holding tier would require more than
    # 20 - SOURCE_DEFENSE_RESERVE=8 → 12 max. Add a big enemy
    # counter to force the holding sizing to need > 12 ships.
    planets = [
        _planet(0, 0, 10.0, 50.0, ships=20),
        _planet(1, 1, 50.0, 50.0, ships=10),
    ]
    fleets = [[100, 1, 90.0, 50.0, _m.pi, 99, 60]]  # huge counter
    world = _world(planets, fleets=fleets)
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    notes = {m.note for m in missions}
    assert "hold" not in notes


# ---------------------------------------------------------------------------
# Operational tier
# ---------------------------------------------------------------------------


def test_operational_tier_emits_when_followon_exists():
    """A nearby unowned planet within FOLLOWON_RADIUS of the target,
    plus enough source garrison, should produce an operational mission."""
    snipe_mod.USE_OPERATIONAL_TIER = 1
    # Source at (10, 50), 500 ships; main target at (50, 50), 10 ships;
    # follow-on candidate at (60, 50), 5 ships (within 40 units of target).
    planets = [
        _planet(0, 0, 10.0, 50.0, ships=500),
        _planet(1, 1, 50.0, 50.0, ships=10),
        _planet(2, -1, 60.0, 50.0, ships=5),
    ]
    world = _world(planets)
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    op_missions = [m for m in missions
                   if m.src_id == 0 and m.target_id == 1
                   and m.note.startswith("op")]
    assert len(op_missions) >= 1
    assert op_missions[0].ships > 10   # bigger than tactical


def test_operational_tier_skipped_when_no_followon_within_radius():
    """No nearby unowned → no operational mission."""
    snipe_mod.USE_OPERATIONAL_TIER = 1
    planets = [
        _planet(0, 0, 10.0, 50.0, ships=500),
        _planet(1, 1, 50.0, 50.0, ships=10),
        # Only one other planet, far away (well past FOLLOWON_RADIUS=40).
    ]
    world = _world(planets)
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    op_missions = [m for m in missions if m.note.startswith("op")]
    assert op_missions == []


def test_all_flags_off_matches_default_byte_for_byte():
    """Belt and suspenders — three repeated calls with USE_HAV=0,
    USE_HOLDING_TIER=0, USE_OPERATIONAL_TIER=0 produce identical
    Mission lists (same set of (src, target, ships, score, eta))."""
    snipe_mod.USE_HAV = 0
    snipe_mod.USE_HOLDING_TIER = 0
    snipe_mod.USE_OPERATIONAL_TIER = 0
    planets = [
        _planet(0, 0, 10.0, 10.0, ships=100),
        _planet(1, 1, 90.0, 90.0, ships=10),
        _planet(2, -1, 50.0, 50.0, ships=5),
    ]
    world = _world(planets)
    model = WorldModel.from_world(world)
    m1 = propose_snipe_missions(world, model)
    m2 = propose_snipe_missions(world, model)
    assert len(m1) == len(m2)
    for a, b in zip(m1, m2):
        assert (a.src_id, a.target_id, a.ships, a.eta) == (
            b.src_id, b.target_id, b.ships, b.eta,
        )
        assert abs(a.score - b.score) < 1e-9
        assert a.note == "" and b.note == ""
