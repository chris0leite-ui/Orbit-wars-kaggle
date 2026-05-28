"""Tests for lib/missions/macro.py — 2P macro state machine."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from lib.geometry import BOARD_SIZE
from lib.missions.macro import (
    MacroEmit,
    MacroState,
    _angular_distance,
    _pick_forward_lateral,
    _polar_angle,
    determine_macro_state,
)


# ---------------------------------------------------------------------------
# Test helpers — minimal mocks for World + WorldModel
# ---------------------------------------------------------------------------


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    """Build a Planet-shape object usable by macro.py.

    macro.py reads only .id, .owner, .x, .y, .ships, .production from
    planets, so a plain SimpleNamespace is enough.
    """
    return SimpleNamespace(
        id=pid, owner=owner, x=float(x), y=float(y),
        ships=int(ships), production=int(production), radius=float(radius),
    )


def _mk_world(planets):
    """Build a minimal World-shape object."""
    return SimpleNamespace(
        planets_by_id={p.id: p for p in planets},
    )


def _initial_planets(planets):
    """Convert planet objects to the [id, owner, x, y, r, ships, prod] tuple
    shape that lib.mirror.build_bijection expects."""
    return [
        [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
        for p in planets
    ]


@dataclass
class FakeModel:
    """Mock WorldModel — only `owner_at` is consumed by macro.py."""
    owner_map: dict = field(default_factory=dict)  # (planet_id, step) -> owner

    def owner_at(self, planet_id, step):
        return self.owner_map.get((int(planet_id), int(step)))


def _build_2p_board(*, home_owner=0, opp_owner=1,
                    home_ships=100, lat_fwd_ships=5,
                    lat_back_ships=5, opp_home_ships=20,
                    home_xy=(20.0, 25.0)):
    """Build the canonical 2P home group: home, opp_home (180-rot), and
    two laterals at the other two corners of the 4-fold symmetric group.

    Returns (planets, world, initial_planets_raw).
    """
    hx, hy = home_xy
    home = _planet(0, home_owner, hx, hy, ships=home_ships)
    opp_home = _planet(3, opp_owner, BOARD_SIZE - hx, BOARD_SIZE - hy,
                       ships=opp_home_ships)
    # Determine which lateral is "forward" (home_angle + pi/2). With
    # home at upper-left (20, 25), forward angle goes through the
    # upper-right or lower-left depending on the rotation direction.
    lat_a = _planet(1, -1, BOARD_SIZE - hx, hy, ships=lat_fwd_ships)
    lat_b = _planet(2, -1, hx, BOARD_SIZE - hy, ships=lat_back_ships)
    planets = [home, lat_a, lat_b, opp_home]
    return planets, _mk_world(planets), _initial_planets(planets)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def test_polar_angle_quadrants():
    # Centre is (50, 50). Points east, north (y down), west, south.
    assert _polar_angle(60, 50) == pytest.approx(0.0)
    # y > centre means polar angle in (0, pi); since y=60 is below centre
    # in screen coords (y-down), atan2(10, 0) = pi/2.
    assert _polar_angle(50, 60) == pytest.approx(math.pi / 2)
    assert _polar_angle(40, 50) == pytest.approx(math.pi)


def test_angular_distance_wraparound():
    # Distance between 0.1 rad and 2*pi - 0.1 rad should be 0.2, not 2*pi - 0.2.
    eps = 0.1
    assert _angular_distance(eps, 2 * math.pi - eps) == pytest.approx(2 * eps)


def test_lateral_picker_quadrants():
    """The forward lateral is +pi/2 ahead of home in polar angle.

    With home in each of the 4 quadrants, the picker should return the
    lateral whose angle is closest to (home_angle + pi/2) mod 2*pi.
    """
    for hx, hy in [(20, 25), (80, 25), (80, 75), (20, 75)]:
        home = _planet(0, 0, hx, hy)
        lat_a = _planet(1, -1, BOARD_SIZE - hx, hy)
        lat_b = _planet(2, -1, hx, BOARD_SIZE - hy)
        chosen = _pick_forward_lateral([lat_a, lat_b], home)
        # The chosen lateral must be one of the two; verify which by
        # computing forward direction explicitly.
        home_angle = _polar_angle(hx, hy)
        forward = (home_angle + math.pi / 2) % (2 * math.pi)
        d_a = _angular_distance(_polar_angle(lat_a.x, lat_a.y), forward)
        d_b = _angular_distance(_polar_angle(lat_b.x, lat_b.y), forward)
        expected = lat_a if d_a < d_b else lat_b
        assert chosen.id == expected.id, (
            f"home=({hx},{hy}): expected {expected.id}, got {chosen.id}"
        )


# ---------------------------------------------------------------------------
# State-machine transitions
# ---------------------------------------------------------------------------


def test_expand_emits_bundled_launch_when_affordable():
    planets, world, init = _build_2p_board(
        home_ships=200, lat_fwd_ships=5, lat_back_ships=5,
    )
    model = FakeModel()
    state = determine_macro_state(world, model, me=0, num_seats=2,
                                  omega=0.04, initial_planets=init)
    assert state.phase == "EXPAND", state.reason
    assert state.emit is not None
    # Source must be home (id 0), target must be the chosen lateral.
    assert state.emit.src_id == 0
    assert state.emit.tgt_id == state.chosen_lateral_id
    # Ships sent must cover the capture (5 + 1 + margin 2 = 8 minimum).
    assert state.emit.ships >= 8


def test_expand_accumulates_when_below_threshold():
    """Home has only 3 ships — can't afford lateral's 5+1+2 = 8 needed.

    With EXPAND_HOME_MIN default = 0 (opening-aggressive), the only
    constraint is having enough ships for the capture itself.
    """
    planets, world, init = _build_2p_board(
        home_ships=3, lat_fwd_ships=5, lat_back_ships=5,
        opp_home_ships=20,
    )
    model = FakeModel()
    state = determine_macro_state(world, model, me=0, num_seats=2,
                                  omega=0.04, initial_planets=init)
    assert state.phase == "EXPAND"
    assert state.emit is None
    assert "accumulating" in state.reason


def test_stockpile_reserves_chosen_lateral_when_we_own_it():
    """We already own the forward lateral with 50 ships; strike threshold
    against opp_home (20 ships, prod 2) is well above 50."""
    planets, world, init = _build_2p_board(
        home_ships=100, lat_fwd_ships=50, lat_back_ships=5,
        opp_home_ships=20,
    )
    # We own the forward lateral.
    home_angle = _polar_angle(20, 25)
    forward = (home_angle + math.pi / 2) % (2 * math.pi)
    lat_a = world.planets_by_id[1]
    lat_b = world.planets_by_id[2]
    d_a = _angular_distance(_polar_angle(lat_a.x, lat_a.y), forward)
    chosen_id = 1 if d_a < _angular_distance(_polar_angle(lat_b.x, lat_b.y), forward) else 2
    world.planets_by_id[chosen_id].owner = 0
    model = FakeModel()
    state = determine_macro_state(world, model, me=0, num_seats=2,
                                  omega=0.04, initial_planets=init)
    assert state.phase == "STOCKPILE", state.reason
    assert state.hold_src == chosen_id
    assert state.emit is None


def test_strike_emits_bundled_launch_at_opp_home():
    """We own forward lateral with a large stockpile; STRIKE fires."""
    planets, world, init = _build_2p_board(
        home_ships=50, lat_fwd_ships=500, lat_back_ships=5,
        opp_home_ships=20,
    )
    home_angle = _polar_angle(20, 25)
    forward = (home_angle + math.pi / 2) % (2 * math.pi)
    lat_a = world.planets_by_id[1]
    lat_b = world.planets_by_id[2]
    d_a = _angular_distance(_polar_angle(lat_a.x, lat_a.y), forward)
    chosen_id = 1 if d_a < _angular_distance(_polar_angle(lat_b.x, lat_b.y), forward) else 2
    world.planets_by_id[chosen_id].owner = 0
    model = FakeModel()
    state = determine_macro_state(world, model, me=0, num_seats=2,
                                  omega=0.04, initial_planets=init)
    assert state.phase == "STRIKE", state.reason
    assert state.emit is not None
    assert state.emit.src_id == chosen_id
    assert state.emit.tgt_id == 3  # opp home
    # Most of the stockpile is shipped (leaving STRIKE_RESERVE behind).
    assert state.emit.ships >= 400


def test_defend_overrides_expand_when_home_flips_predicted():
    """Model predicts home flips to enemy within DEFEND_HORIZON — DEFEND."""
    planets, world, init = _build_2p_board(home_ships=200)
    # owner_at(home, 10) returns the opponent → home flips at step 10.
    model = FakeModel(owner_map={(0, 10): 1})
    state = determine_macro_state(world, model, me=0, num_seats=2,
                                  omega=0.04, initial_planets=init)
    assert state.phase == "DEFEND"
    assert state.emit is None


def test_disabled_in_4p():
    planets, world, init = _build_2p_board()
    model = FakeModel()
    state = determine_macro_state(world, model, me=0, num_seats=4,
                                  omega=0.04, initial_planets=init)
    assert state.phase == "DISABLED"
    assert state.emit is None
    assert state.hold_src is None


# ---------------------------------------------------------------------------
# Byte-parity OFF
# ---------------------------------------------------------------------------


def test_macro_off_preserves_agent_module_import():
    """With BASELINE_MACRO unset, the agent module imports unchanged.

    This is the byte-parity guarantee: the only change in agent behaviour
    when the flag is off is that a `macro_state` is computed and the
    resulting `macro_reserved` is an empty set, `macro_moves` is empty.
    The chooser, post-passes, and emit logic are byte-identical.
    """
    import os
    # Save + clear the env var to simulate "macro disabled".
    saved = os.environ.pop("BASELINE_MACRO", None)
    try:
        # Re-import the agent module fresh to pick up the unset env var.
        import importlib
        import agents.baseline.main as agent_main
        importlib.reload(agent_main)
        assert agent_main.MACRO_ENABLED is False
    finally:
        if saved is not None:
            os.environ["BASELINE_MACRO"] = saved
            import importlib
            import agents.baseline.main as agent_main
            importlib.reload(agent_main)
