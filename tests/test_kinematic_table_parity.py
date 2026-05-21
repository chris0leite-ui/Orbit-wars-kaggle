"""Layer 1 parity pin tests for `lib.kinematic_table.KinematicTable`.

Phase α of /root/.claude/plans/do-it-thoroughly-consider-tingly-fox.md.

These tests lock the 100% accuracy gate: every lookup MUST be bit-
identical to what the inline call site (`predict_relative` for orbital,
raw obs (x, y) for static, `path[path_index + lead]` for active comets,
sentinel `OFF_BOARD` for expired comets) would have computed.

`==` is used as the assertion, NOT `pytest.approx` or tolerance —
the table is built by calling the SAME `predict_relative` function the
inline site calls with the SAME planet tuple. Any divergence is a
contract bug, not a precision artefact.
"""

from __future__ import annotations

import math
import random

import pytest

from lib.intent import Planet, World
from lib.kinematic_table import (
    DEFAULT_MAX_LEAD,
    OFF_BOARD,
    KinematicTable,
)
from lib.orbit import is_orbiting, predict_relative


# ---------------------------------------------------------------------------
# World-builder helpers (adapted from test_trajectory_comet_handling.py
# so the test isolates kinematic_table behaviour).
# ---------------------------------------------------------------------------


def _make_obs(planets, *, omega, step=0, comet_groups=None, comet_ids=None):
    return {
        "player": 0,
        "planets": planets,
        "fleets": [],
        "angular_velocity": float(omega),
        "initial_planets": [],
        "comet_planet_ids": list(comet_ids or []),
        "comets": list(comet_groups or []),
        "step": int(step),
    }


def _static_planet_row(pid, x, y, radius=5.0):
    return [pid, -1, float(x), float(y), float(radius), 0, 1]


def _orbital_planet_row(pid, x, y, radius=2.0):
    return [pid, -1, float(x), float(y), float(radius), 0, 1]


# ---------------------------------------------------------------------------
# Static planets — lookup returns exact (p.x, p.y).
# ---------------------------------------------------------------------------


def test_static_planet_lookup_returns_exact_position():
    """Static planet (orb_r + radius >= ROTATION_RADIUS_LIMIT) is not
    rotated; lookup must return (p.x, p.y) byte-identically for any lead.
    """
    # orb_r at (10, 10) ≈ 56.57; +5 = 61.57 > 50 → static.
    obs = _make_obs(
        [_static_planet_row(0, 10.0, 10.0, radius=5.0)],
        omega=0.05, step=42,
    )
    world = World.from_obs(obs)
    table = KinematicTable()
    table.begin_turn(world)

    for lead in [0, 1, 5, 17, 100, 200, 249]:
        x, y = table.lookup_relative(0, lead)
        assert x == 10.0
        assert y == 10.0
    assert table.kind(0) == "static"


def test_static_planet_omega_zero_treated_as_static():
    """Even an inner-orbit planet (would orbit) is static when omega=0."""
    obs = _make_obs(
        [_orbital_planet_row(0, 60.0, 50.0, radius=2.0)],  # orb_r=10, inner
        omega=0.0, step=0,
    )
    world = World.from_obs(obs)
    table = KinematicTable()
    table.begin_turn(world)

    for lead in [0, 1, 50, 100]:
        x, y = table.lookup_relative(0, lead)
        assert x == 60.0
        assert y == 50.0
    assert table.kind(0) == "static"


# ---------------------------------------------------------------------------
# Orbital planets — bit-parity with predict_relative.
# ---------------------------------------------------------------------------


def test_orbital_planet_bit_parity_50_random_leads():
    """Orbital planet (omega != 0, inside rotation limit): lookup must
    equal predict_relative(p_tuple, omega, lead) BIT-IDENTICALLY for 50
    randomly-chosen (lead) values across [0, max_lead].

    The table builds positions by calling predict_relative with the
    same planet tuple, so equality is by construction. This test pins
    that the build is correct and lookup_relative routes back to the
    cached entry verbatim.
    """
    # Inner planet at (60, 50): orb_r = 10, radius=2, orb_r+r=12 < 50.
    obs = _make_obs(
        [_orbital_planet_row(0, 60.0, 50.0, radius=2.0)],
        omega=0.05, step=0,
    )
    world = World.from_obs(obs)
    table = KinematicTable()
    table.begin_turn(world)
    assert table.kind(0) == "orbital"

    p = world.planets_by_id[0]
    p_tuple = [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
    omega = world.omega

    rng = random.Random(42)
    leads = [0, 1, 2, DEFAULT_MAX_LEAD] + [rng.randint(0, DEFAULT_MAX_LEAD) for _ in range(46)]
    for lead in leads:
        expected = predict_relative(p_tuple, omega, lead)
        got = table.lookup_relative(0, lead)
        assert got == expected, (
            f"lead={lead}: table got {got}, expected {expected}; "
            f"diff_x={got[0] - expected[0]!r}, diff_y={got[1] - expected[1]!r}"
        )


def test_orbital_lookup_past_max_lead_raises():
    """Out-of-range orbital lookups MUST raise rather than silently
    answer with re-derived (possibly drifted) values. Surface contract
    violations loudly."""
    obs = _make_obs(
        [_orbital_planet_row(0, 60.0, 50.0, radius=2.0)],
        omega=0.05, step=0,
    )
    world = World.from_obs(obs)
    table = KinematicTable(max_lead=10)
    table.begin_turn(world)

    # Within range: fine.
    table.lookup_relative(0, 10)
    # Past range: must raise.
    with pytest.raises(IndexError):
        table.lookup_relative(0, 11)


def test_orbital_multiple_planets_parity():
    """Several orbital planets simultaneously; each lookup matches
    predict_relative for THAT planet's tuple."""
    rows = [
        _orbital_planet_row(0, 60.0, 50.0, radius=2.0),
        _orbital_planet_row(1, 50.0, 60.0, radius=2.5),
        _orbital_planet_row(2, 40.0, 50.0, radius=1.5),
        _orbital_planet_row(3, 50.0, 40.0, radius=3.0),
    ]
    obs = _make_obs(rows, omega=0.05, step=0)
    world = World.from_obs(obs)
    table = KinematicTable()
    table.begin_turn(world)

    for pid in (0, 1, 2, 3):
        p = world.planets_by_id[pid]
        p_tuple = [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
        for lead in (0, 1, 7, 23, 100, DEFAULT_MAX_LEAD):
            expected = predict_relative(p_tuple, world.omega, lead)
            got = table.lookup_relative(pid, lead)
            assert got == expected


# ---------------------------------------------------------------------------
# Comet handling — path lookup + expiry sentinel.
# ---------------------------------------------------------------------------


def _comet_group(pid, path, path_index=0):
    """Single-comet group, matching env's obs["comets"] schema."""
    return {
        "planet_ids": [pid],
        "paths": [list(path)],
        "path_index": int(path_index),
    }


def test_comet_active_lookup_returns_path_entries():
    """Active comet (path_index in range, lead within remaining path):
    lookup returns path[path_index + lead] exactly."""
    path = [[90.0, 30.0], [70.0, 30.0], [50.0, 30.0], [30.0, 30.0], [10.0, 30.0]]
    obs = _make_obs(
        [_static_planet_row(20, 90.0, 30.0, radius=2.0)],
        omega=0.0, step=0,
        comet_groups=[_comet_group(20, path, path_index=0)],
        comet_ids=[20],
    )
    world = World.from_obs(obs)
    table = KinematicTable()
    table.begin_turn(world)
    assert table.kind(20) == "comet"

    for t in range(5):
        x, y = table.lookup_relative(20, t)
        assert x == float(path[t][0])
        assert y == float(path[t][1])


def test_comet_expiry_boundary_returns_off_board():
    """At lead == len(path) - path_index (one past the last valid
    index), lookup MUST return OFF_BOARD. This is the exact contract
    the trajectory.py:214-218 expiry guard relies on."""
    path = [[90.0, 30.0], [70.0, 30.0], [50.0, 30.0]]
    obs = _make_obs(
        [_static_planet_row(20, 90.0, 30.0, radius=2.0)],
        omega=0.0, step=0,
        comet_groups=[_comet_group(20, path, path_index=0)],
        comet_ids=[20],
    )
    world = World.from_obs(obs)
    table = KinematicTable()
    table.begin_turn(world)

    # In range.
    assert table.lookup_relative(20, 0) == (90.0, 30.0)
    assert table.lookup_relative(20, 1) == (70.0, 30.0)
    assert table.lookup_relative(20, 2) == (50.0, 30.0)
    # First out-of-range tick → sentinel.
    assert table.lookup_relative(20, 3) == OFF_BOARD
    assert table.lookup_relative(20, 4) == OFF_BOARD
    assert table.lookup_relative(20, 100) == OFF_BOARD


def test_comet_with_offset_path_index():
    """path_index > 0: lookup at lead=0 returns path[path_index]."""
    path = [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0], [50.0, 50.0]]
    obs = _make_obs(
        [_static_planet_row(20, 30.0, 30.0, radius=2.0)],  # current pos = path[2]
        omega=0.0, step=2,
        comet_groups=[_comet_group(20, path, path_index=2)],
        comet_ids=[20],
    )
    world = World.from_obs(obs)
    table = KinematicTable()
    table.begin_turn(world)

    # path_index=2, lead=0 → path[2]; lead=1 → path[3]; lead=2 → path[4]; lead=3 → OFF_BOARD.
    assert table.lookup_relative(20, 0) == (30.0, 30.0)
    assert table.lookup_relative(20, 1) == (40.0, 40.0)
    assert table.lookup_relative(20, 2) == (50.0, 50.0)
    assert table.lookup_relative(20, 3) == OFF_BOARD


def test_comet_past_expiry_all_sentinel():
    """path_index >= len(path): every lookup returns OFF_BOARD.
    (In practice the env removes such a comet from obs["comets"] before
    this is observable, but the table must be defensive.)"""
    path = [[10.0, 10.0], [20.0, 20.0]]
    obs = _make_obs(
        [_static_planet_row(20, 10.0, 10.0, radius=2.0)],
        omega=0.0, step=5,
        # path_index BEYOND end-of-path (defensive test only).
        comet_groups=[_comet_group(20, path, path_index=5)],
        comet_ids=[20],
    )
    world = World.from_obs(obs)
    table = KinematicTable()
    table.begin_turn(world)

    for lead in (0, 1, 2, 10, 50):
        assert table.lookup_relative(20, lead) == OFF_BOARD


# ---------------------------------------------------------------------------
# Fingerprint / lifecycle.
# ---------------------------------------------------------------------------


def test_begin_turn_idempotent_within_same_turn():
    """Two begin_turn calls on the same world: second is a no-op."""
    obs = _make_obs(
        [_orbital_planet_row(0, 60.0, 50.0, radius=2.0)],
        omega=0.05, step=10,
    )
    world = World.from_obs(obs)
    table = KinematicTable()

    fired1 = table.begin_turn(world)
    fired2 = table.begin_turn(world)
    assert fired1 is True
    assert fired2 is False


def test_begin_turn_rebuilds_on_new_step():
    """Different step (new turn) → rebuild, even if planet ids match."""
    obs1 = _make_obs(
        [_orbital_planet_row(0, 60.0, 50.0, radius=2.0)],
        omega=0.05, step=10,
    )
    obs2 = _make_obs(
        # Same planet id, but new turn means new Planet() identity AND step++.
        [_orbital_planet_row(0, 60.0, 50.0, radius=2.0)],
        omega=0.05, step=11,
    )
    world1 = World.from_obs(obs1)
    world2 = World.from_obs(obs2)
    table = KinematicTable()

    fired1 = table.begin_turn(world1)
    fired2 = table.begin_turn(world2)
    assert fired1 is True
    assert fired2 is True


def test_game_boundary_wipe():
    """Different planet set entirely (new game): wipe + rebuild.

    This mirrors PendingSchedule's fingerprint-reset pattern; the
    table must not retain stale entries across games."""
    obs_game_a = _make_obs(
        [_orbital_planet_row(0, 60.0, 50.0, radius=2.0)],
        omega=0.05, step=200,
    )
    obs_game_b = _make_obs(
        [_orbital_planet_row(5, 55.0, 50.0, radius=2.5)],  # different pid
        omega=0.07, step=0,  # step drops back to 0, omega differs
    )
    world_a = World.from_obs(obs_game_a)
    world_b = World.from_obs(obs_game_b)

    table = KinematicTable()
    table.begin_turn(world_a)
    assert table.has(0)
    assert not table.has(5)

    table.begin_turn(world_b)
    assert not table.has(0)
    assert table.has(5)


def test_reset_clears_state():
    obs = _make_obs(
        [_orbital_planet_row(0, 60.0, 50.0, radius=2.0)],
        omega=0.05, step=0,
    )
    world = World.from_obs(obs)
    table = KinematicTable()
    table.begin_turn(world)
    assert table.has(0)

    table.reset()
    assert not table.has(0)
    assert table.stats()["n_planets"] == 0


# ---------------------------------------------------------------------------
# window() — the predict_fleet_fate integration shape.
# ---------------------------------------------------------------------------


def test_window_matches_inline_planet_positions_build():
    """`table.window(pids, wait_N, max_steps+1)` must produce a dict
    identical to what `lib/trajectory.py:137-159`'s inline build produces.

    This is the critical Phase γ swap: the inline code becomes a
    one-liner. If this test passes, predict_fleet_fate's inner loop
    sees the same data structure either way.
    """
    rows = [
        _orbital_planet_row(0, 60.0, 50.0, radius=2.0),
        _orbital_planet_row(1, 50.0, 60.0, radius=2.5),
        _static_planet_row(7, 10.0, 10.0, radius=5.0),
    ]
    obs = _make_obs(rows, omega=0.05, step=0)
    world = World.from_obs(obs)
    table = KinematicTable()
    table.begin_turn(world)

    wait_N = 3
    max_steps = 50
    length = max_steps + 1
    pids = list(world.planets_by_id.keys())

    # Inline build (verbatim from trajectory.py:137-159, comet path
    # excluded since there are no comets here).
    expected: dict[int, list[tuple[float, float]]] = {}
    for pid, p in world.planets_by_id.items():
        p_tuple = [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
        if is_orbiting(p_tuple) and world.omega != 0.0:
            expected[pid] = [
                predict_relative(p_tuple, world.omega, wait_N + t)
                for t in range(length)
            ]
        else:
            expected[pid] = [(p.x, p.y)] * length

    got = table.window(pids, start_offset=wait_N, length=length)

    assert set(got.keys()) == set(expected.keys())
    for pid in expected:
        assert got[pid] == expected[pid], (
            f"pid={pid}: window output differs from inline build "
            f"(first divergence: "
            f"{next((i, g, e) for i, (g, e) in enumerate(zip(got[pid], expected[pid])) if g != e)})"
        )


def test_window_with_comets_matches_inline():
    """Same as above but with a comet in the mix — the table's comet
    section must produce the same list-of-(x,y) as the inline build
    at trajectory.py:138-150 (which handles comets specially)."""
    path = [[90.0, 30.0], [70.0, 30.0], [50.0, 30.0], [30.0, 30.0]]
    rows = [
        _orbital_planet_row(0, 60.0, 50.0, radius=2.0),
        _static_planet_row(7, 10.0, 10.0, radius=5.0),
        _static_planet_row(20, 90.0, 30.0, radius=2.0),  # comet current pos
    ]
    obs = _make_obs(
        rows, omega=0.05, step=0,
        comet_groups=[_comet_group(20, path, path_index=0)],
        comet_ids=[20],
    )
    world = World.from_obs(obs)
    table = KinematicTable()
    table.begin_turn(world)

    wait_N = 0
    max_steps = 6
    length = max_steps + 1
    pids = list(world.planets_by_id.keys())

    # Build expected dict mirroring trajectory.py:137-159 verbatim.
    from lib.world_model import _comet_paths_by_id
    comet_paths = _comet_paths_by_id(world)
    expected: dict[int, list[tuple[float, float]]] = {}
    for pid, p in world.planets_by_id.items():
        if int(pid) in comet_paths:
            cpath, c_idx = comet_paths[int(pid)]
            positions: list[tuple[float, float]] = []
            for t in range(length):
                path_t = int(c_idx) + int(wait_N) + t
                if 0 <= path_t < len(cpath):
                    pt = cpath[path_t]
                    positions.append((float(pt[0]), float(pt[1])))
                else:
                    positions.append(OFF_BOARD)
            expected[pid] = positions
            continue
        p_tuple = [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
        if is_orbiting(p_tuple) and world.omega != 0.0:
            expected[pid] = [
                predict_relative(p_tuple, world.omega, wait_N + t)
                for t in range(length)
            ]
        else:
            expected[pid] = [(p.x, p.y)] * length

    got = table.window(pids, start_offset=wait_N, length=length)
    assert got == expected


# ---------------------------------------------------------------------------
# comet_paths_view — schema parity with _comet_paths_by_id.
# ---------------------------------------------------------------------------


def test_comet_paths_view_matches_world_model_helper():
    """The table's `comet_paths_view()` must return the same dict shape
    as `lib.world_model._comet_paths_by_id(world)`. Phase γ will swap
    that function's body to read from here."""
    from lib.world_model import _comet_paths_by_id

    path_a = [[90.0, 30.0], [70.0, 30.0], [50.0, 30.0]]
    path_b = [[10.0, 80.0], [20.0, 80.0]]
    obs = _make_obs(
        [
            _static_planet_row(20, 90.0, 30.0, radius=2.0),
            _static_planet_row(21, 10.0, 80.0, radius=2.0),
        ],
        omega=0.0, step=0,
        comet_groups=[
            _comet_group(20, path_a, path_index=0),
            _comet_group(21, path_b, path_index=0),
        ],
        comet_ids=[20, 21],
    )
    world = World.from_obs(obs)
    table = KinematicTable()
    table.begin_turn(world)

    got = table.comet_paths_view()
    expected = _comet_paths_by_id(world)
    assert got.keys() == expected.keys()
    for pid in expected:
        ep, ei = expected[pid]
        gp, gi = got[pid]
        assert gi == ei
        # Compare path entry-by-entry (lists of [x, y]).
        assert len(gp) == len(ep)
        for j in range(len(ep)):
            assert tuple(gp[j]) == tuple(ep[j])


# ---------------------------------------------------------------------------
# Module-level singleton sanity.
# ---------------------------------------------------------------------------


def test_module_singleton_wraps_class():
    """The module-level functions delegate to the singleton; reset()
    wipes it; begin_turn rebuilds."""
    from lib import kinematic_table as kt

    obs = _make_obs(
        [_orbital_planet_row(0, 60.0, 50.0, radius=2.0)],
        omega=0.05, step=0,
    )
    world = World.from_obs(obs)
    kt.clear()
    assert kt.get_default().stats()["n_planets"] == 0
    kt.begin_turn(world)
    assert kt.get_default().has(0)

    p = world.planets_by_id[0]
    p_tuple = [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
    for lead in (0, 5, 23):
        assert kt.lookup_relative(0, lead) == predict_relative(p_tuple, world.omega, lead)

    kt.clear()
    assert kt.get_default().stats()["n_planets"] == 0
