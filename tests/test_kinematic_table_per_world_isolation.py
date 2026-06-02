"""Per-World isolation test for `lib.kinematic_table`.

This is the Rule 38 fix-verification anchor for the singleton refactor
(2026-06-02). The original failure mode lived in
`audit/2026-05-29-postmortem-three-abs-headroom-empty.md`: two seats in
one Python process called `kt.begin_turn(world)` on DIFFERENT worlds,
and the second call clobbered the first because the singleton at
`lib/kinematic_table.py:_DEFAULT` is process-global. The first seat
then read the second seat's planet positions and a ~9 pp phantom
regression appeared in in-process A/Bs.

The fix attaches a per-World `KinematicTable` instance via
`lib.kinematic_table.attach(world)` / `for_world(world)`. Distinct
worlds → distinct tables → no contamination. This test reproduces the
failing state, applies the new path, and asserts the failure mode is
gone.

The test deliberately AVOIDS using `get_default()` for its assertions —
that's the legacy path that's still contaminated and exists only for
transitional back-compat with `test_kinematic_table_baseline_wiring`.
Reading via `for_world(...)` is what production code should do after
this refactor.
"""

from __future__ import annotations

import pytest

import lib.kinematic_table as kt
from lib.intent import World
from lib.kinematic_table import (
    KinematicTable,
    attach,
    begin_turn,
    for_world,
    get_default,
)


# ---------------------------------------------------------------------------
# World-builder helpers — mirror tests/test_kinematic_table_parity.py
# ---------------------------------------------------------------------------


def _orbital_row(pid, x, y, radius=2.0):
    """Inner-orbit planet (will rotate when omega != 0)."""
    return [pid, -1, float(x), float(y), float(radius), 0, 1]


def _make_obs(planets, *, omega, step=0):
    return {
        "player": 0,
        "planets": planets,
        "fleets": [],
        "angular_velocity": float(omega),
        "initial_planets": [],
        "comet_planet_ids": [],
        "comets": [],
        "step": int(step),
    }


def _world_seat_a():
    """Seat A's world view: one orbital planet at (60, 50)."""
    obs = _make_obs([_orbital_row(0, 60.0, 50.0)], omega=0.05, step=10)
    return World.from_obs(obs)


def _world_seat_b():
    """Seat B's world view: SAME planet id, but at a DIFFERENT position
    (mirrored across the board). With shared singleton state, seat B's
    `begin_turn` would clobber seat A's positions.
    """
    obs = _make_obs([_orbital_row(0, 40.0, 50.0)], omega=0.05, step=10)
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# Isolation guarantees — the load-bearing assertions for this refactor.
# ---------------------------------------------------------------------------


def test_two_worlds_get_independent_tables():
    """The core guarantee: `attach(A)` and `attach(B)` return distinct
    `KinematicTable` instances even when called in the same process.
    """
    kt.clear()  # paranoia — reset the legacy singleton too
    a = _world_seat_a()
    b = _world_seat_b()

    table_a = attach(a)
    table_b = attach(b)

    assert table_a is not table_b, (
        "attach(a) and attach(b) returned the SAME KinematicTable — "
        "per-World isolation is broken"
    )
    # Neither per-World table is the legacy singleton.
    assert table_a is not get_default()
    assert table_b is not get_default()


def test_attach_is_idempotent_per_world():
    """`attach(world)` repeated on the same world returns the same
    instance. Required so callers can hand World around without
    accidentally replacing its table.
    """
    a = _world_seat_a()
    first = attach(a)
    second = attach(a)
    third = for_world(a)
    assert first is second is third


def test_begin_turn_primes_per_world_first():
    """After `begin_turn(world)`, `for_world(world)` must return a
    populated, this-turn table — not None, not a stale instance.
    """
    a = _world_seat_a()
    rebuilt = begin_turn(a)
    assert rebuilt is True
    table = for_world(a)
    assert table is not None
    assert table.n_planets == 1
    assert table.step == 10


def test_postmortem_scenario_seat_a_unpolluted_by_seat_b():
    """The bug, reproduced and fixed.

    SCENARIO: seat A and seat B share a process (in-process A/B
    harness). Seat A primes its world, then seat B primes ITS world.
    With the legacy singleton, seat A's subsequent read would return
    seat B's positions. With per-World attachment, A's reads stay
    consistent.
    """
    kt.clear()
    a = _world_seat_a()
    b = _world_seat_b()

    # 1. Seat A primes its world.
    begin_turn(a)
    a_pos_before_b = for_world(a).lookup_relative(0, 5)

    # 2. Seat B primes its world — this is the contamination trigger
    #    under the old singleton design.
    begin_turn(b)

    # 3. Seat A re-reads. Per-World guarantees: same result as before.
    a_pos_after_b = for_world(a).lookup_relative(0, 5)
    assert a_pos_after_b == a_pos_before_b, (
        f"seat A's position read changed after seat B primed its world "
        f"(before={a_pos_before_b}, after={a_pos_after_b}) — "
        f"per-World isolation is broken"
    )

    # 4. Seat A and seat B should hold different positions because
    #    their input worlds differ at planet 0 (60,50) vs (40,50).
    b_pos = for_world(b).lookup_relative(0, 5)
    assert a_pos_after_b != b_pos, (
        f"seat A and seat B read the SAME position for planet 0 "
        f"({a_pos_after_b}) — their input worlds differed, so reads "
        f"must differ; tables are aliased or shared"
    )


def test_for_world_returns_none_for_unprimed_world():
    """`for_world(world)` returns None when the world has never had
    `attach` or `begin_turn` called on it. This is the signal that
    callers (`lib/trajectory._table_window_or_none`) use to take the
    inline fallback — they MUST NOT fall back to the singleton.
    """
    fresh = _world_seat_a()
    assert for_world(fresh) is None


def test_for_world_independent_of_singleton_state():
    """Mutating the legacy singleton must NOT affect any per-World
    table's reads. This is the structural guarantee that closes the
    contamination class.
    """
    a = _world_seat_a()
    begin_turn(a)
    a_pos = for_world(a).lookup_relative(0, 5)

    # Pollute the legacy singleton with a totally different world.
    other_world = World.from_obs(_make_obs(
        [_orbital_row(0, 25.0, 25.0)], omega=0.05, step=99,
    ))
    get_default().begin_turn(other_world)

    # Per-World table for A is unaffected.
    assert for_world(a).lookup_relative(0, 5) == a_pos


# ---------------------------------------------------------------------------
# Back-compat — the transitional double-prime keeps existing wiring
# tests green. Lock that contract here so a future "drop the double-
# prime" refactor catches the deprecation handoff explicitly.
# ---------------------------------------------------------------------------


def test_begin_turn_still_primes_legacy_singleton_transitional():
    """During the transition, `begin_turn(world)` ALSO primes the
    module-global singleton so `test_kinematic_table_baseline_wiring`
    can keep reading via `get_default()`. When that test migrates to
    `for_world(world)`, this contract can be dropped and this test
    deleted with it.
    """
    kt.clear()
    assert get_default().stats()["n_planets"] == 0

    a = _world_seat_a()
    begin_turn(a)

    # Per-World path is the one we care about.
    assert for_world(a) is not None
    # Singleton path also populated for back-compat with wiring test.
    assert get_default().stats()["n_planets"] == 1
