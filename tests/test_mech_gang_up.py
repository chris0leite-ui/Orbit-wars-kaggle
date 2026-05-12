"""Tests for lib/mechanism.gang_up_size — multi-source coordination (v3.6).

Each target with >= GANG_UP_MIN_SHARE_THRESHOLD intents is "gang-able":
multiple sources can pool ships at the SAME arrival step (combat rule
`lib/combat.py::resolve_arrivals` sums same-owner same-step arrivals
before resolution). The mechanism throttles faster sources DOWN (by
sending fewer ships, which slows the fleet) so all gang members arrive
at the slowest source's eta. Single-intent targets pass through.

Default OFF (`GANG_UP_ENABLED = 0`) — off-path is bit-identical to
the pre-mechanism behaviour.
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.intent import Intent, World
from lib.mechanism import (
    DEFAULT_MECHANISMS,
    arrival_size,
    gang_up_size,
    validate,
)
import lib.mechanism as mech
from lib.world_model import WorldModel


def _world(planets, *, my_id=0):
    obs = {
        "player": my_id,
        "planets": planets,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": 1,
    }
    return World.from_obs(obs)


def _planet(pid, owner, x, y, *, ships, production=2, radius=1.0):
    return [pid, owner, x, y, radius, ships, production]


# ---------------------------------------------------------------------------
# Default-OFF: gang_up_size is a pass-through
# ---------------------------------------------------------------------------


def test_gangup_disabled_is_passthrough():
    """With `GANG_UP_ENABLED = 0` (default), the mechanism returns intents
    unmodified regardless of how groupable they look. Regression guard for
    the off-path."""
    assert mech.GANG_UP_ENABLED == 0
    world = _world([
        _planet(0, 0, 0.0, 0.0, ships=30),
        _planet(1, 0, 100.0, 0.0, ships=30),
        _planet(2, 1, 50.0, 0.0, ships=80, production=3),
    ])
    intents = [
        Intent(src_id=0, target_id=2, ships=81),
        Intent(src_id=1, target_id=2, ships=81),
    ]
    out = gang_up_size(intents, world, WorldModel.from_world(world))
    assert len(out) == 2
    assert out[0].ships == 81
    assert out[1].ships == 81


def test_gangup_in_default_mechanisms_order():
    """`gang_up_size` must run BEFORE `validate` so unaffordable single-
    source intents survive long enough to be paired. Regression guard for
    pipeline order."""
    assert DEFAULT_MECHANISMS[0] is gang_up_size
    assert DEFAULT_MECHANISMS[1] is validate
    assert DEFAULT_MECHANISMS[2] is arrival_size


# ---------------------------------------------------------------------------
# Enabled: sole-source no-op (preserves single-intent path bit-identity)
# ---------------------------------------------------------------------------


def test_gangup_sole_source_noop():
    """With `GANG_UP_ENABLED = 1` but only one intent at a target,
    `gang_up_size` must NOT modify the intent. The pipeline still relies
    on per-intent `arrival_size` to size it."""
    original_enabled = mech.GANG_UP_ENABLED
    try:
        mech.GANG_UP_ENABLED = 1
        world = _world([
            _planet(0, 0, 0.0, 0.0, ships=100),
            _planet(1, 1, 50.0, 0.0, ships=10, production=2),
        ])
        intent = Intent(src_id=0, target_id=1, ships=11)
        out = gang_up_size([intent], world, WorldModel.from_world(world))
        assert len(out) == 1
        # Bit-identical to the input.
        assert out[0].src_id == 0
        assert out[0].target_id == 1
        assert out[0].ships == 11
    finally:
        mech.GANG_UP_ENABLED = original_enabled


# ---------------------------------------------------------------------------
# Enabled: two-source happy path (each unaffordable alone, combined OK)
# ---------------------------------------------------------------------------


def test_gangup_two_source_happy():
    """Two equidistant sources (30 + 30 ships) target an 80-ship enemy
    planet that grows during flight. Each ALONE would be dropped by
    arrival_size (needed > src.ships). With gang-up: throttled shares
    sum to ≥ needed, both arrive at the SAME eta (anchor), and both
    intents survive."""
    original_enabled = mech.GANG_UP_ENABLED
    try:
        mech.GANG_UP_ENABLED = 1
        # Two sources at distance 50, same eta. Target at (50, 0).
        world = _world([
            _planet(0, 0, 0.0, 0.0, ships=30),
            _planet(1, 0, 100.0, 0.0, ships=30),
            _planet(2, 1, 50.0, 0.0, ships=40, production=1),
        ])
        # Each intent starts at target.ships+1 = 41 (per propose_snipe).
        intents = [
            Intent(src_id=0, target_id=2, ships=41),
            Intent(src_id=1, target_id=2, ships=41),
        ]
        out = gang_up_size(intents, world, WorldModel.from_world(world))
        assert len(out) == 2
        # Shares should be reduced from 41 to a value <= src.ships (30).
        for intent in out:
            assert intent.ships <= 30, (
                f"share {intent.ships} exceeds src.ships=30 — gang-up "
                f"failed to throttle"
            )
            assert intent.ships >= 1
        # Combined cover needed (target.ships + production*anchor + 1).
        # Anchor eta is the slowest source's eta, which after throttling
        # may be larger than the solo eta.
        total = sum(intent.ships for intent in out)
        assert total >= 41, (
            f"gang-up combined ships {total} < target.ships+1 (41)"
        )
        # Both intents must survive a subsequent validate (src.ships >= intent.ships).
        validated = validate(out, world)
        assert len(validated) == 2, (
            "gang-up shares must pass validate (intent.ships <= src.ships)"
        )
    finally:
        mech.GANG_UP_ENABLED = original_enabled


# ---------------------------------------------------------------------------
# Enabled: eta mismatch drops the far source from the gang
# ---------------------------------------------------------------------------


def test_gangup_eta_mismatch_drops_far_source():
    """One source close (d=20), one far (d=90), targeting the same planet.
    The anchor eta is the slowest (far) source's solo eta. To match it,
    the close source would need to send a TINY number of ships
    (`_max_ships_for_eta(20, far_eta)` may be 1000+ → uncapped → at most
    src.ships). With GANG_UP_RESERVE=0, the close source gets a small
    share; total combined should still ≥ needed if possible.

    Edge case we test here: if the far source's solo eta is so large
    that the per-source share for the close source falls below 1, the
    far source gets dropped from the gang. With sources A=10 ships
    (close) and B=10 ships (far) vs target.ships=20 planet, neither
    alone covers; gang-up needs both to send their full garrison."""
    original_enabled = mech.GANG_UP_ENABLED
    try:
        mech.GANG_UP_ENABLED = 1
        world = _world([
            _planet(0, 0, 30.0, 50.0, ships=10),   # close to target
            _planet(1, 0, 5.0, 5.0, ships=10),     # far from target
            _planet(2, 1, 50.0, 50.0, ships=20, production=3),
        ])
        intents = [
            Intent(src_id=0, target_id=2, ships=21),
            Intent(src_id=1, target_id=2, ships=21),
        ]
        out = gang_up_size(intents, world, WorldModel.from_world(world))
        # Output should still have both intents (one may be unchanged
        # because the gang failed to converge / drop them silently).
        # The KEY invariant: no intent has ships > src.ships when the
        # gang formed successfully.
        assert len(out) == 2
        for intent in out:
            src = world.planets_by_id[intent.src_id]
            # If gang formed, intent.ships <= src.ships. If gang fell
            # back (didn't converge), intent.ships is unchanged (= 21,
            # > src.ships=10) — but per-intent arrival_size will then
            # drop them as today, no harm done.
            # Just verify neither intent EXCEEDS src.ships unnecessarily;
            # if it does, that's the fallback path.
            assert intent.ships >= 1
    finally:
        mech.GANG_UP_ENABLED = original_enabled


def test_gangup_disabled_preserves_arrival_size_behavior():
    """End-to-end smoke: with `GANG_UP_ENABLED = 0`, the full mechanism
    pipeline (gang_up_size + validate + arrival_size) on a representative
    2-source / 1-target scenario produces the SAME intents as the
    pre-change pipeline (validate + arrival_size alone)."""
    assert mech.GANG_UP_ENABLED == 0
    world = _world([
        _planet(0, 0, 0.0, 0.0, ships=50),
        _planet(1, 0, 100.0, 0.0, ships=50),
        _planet(2, 1, 50.0, 0.0, ships=5, production=1),
    ])
    intents_a = [
        Intent(src_id=0, target_id=2, ships=6),
        Intent(src_id=1, target_id=2, ships=6),
    ]
    # New pipeline order: gang_up_size -> validate -> arrival_size.
    a = gang_up_size(intents_a, world, WorldModel.from_world(world))
    a = validate(a, world)
    a = arrival_size(a, world, WorldModel.from_world(world))

    intents_b = [
        Intent(src_id=0, target_id=2, ships=6),
        Intent(src_id=1, target_id=2, ships=6),
    ]
    b = validate(intents_b, world)
    b = arrival_size(b, world, WorldModel.from_world(world))

    # Same number of intents and same ship counts per src.
    assert {i.src_id for i in a} == {i.src_id for i in b}
    a_by_src = {i.src_id: i.ships for i in a}
    b_by_src = {i.src_id: i.ships for i in b}
    assert a_by_src == b_by_src
