"""Phase 8 tests for lib.trajectory_layer — `World.snapshot_at(t)`.

`snapshot_at(t)` re-anchors a World so its turn 0 = parent's turn t.
The trajectory layer's per-planet `ownership_at` already SCORES at
a horizon; `snapshot_at` is the complement — produce a fresh World
the caller can chain another `BundleSearch` from. Pinned invariants:

1. `snapshot_at(0)` is the identity.
2. Negative `t` raises ValueError.
3. `snapshot_at(t).ownership_at(p, 0) == self.ownership_at(p, t)`
   for every planet p (the LOAD-BEARING invariant; enables chained
   search to see the same world state).
4. `planet_position(p, 0)` on the snapshot matches `planet_position(
   p, t)` on the parent.
5. Fleets that arrived by t are dropped; in-flight fleets are
   carried forward at their projected position; future-scheduled
   fleets keep their relative-to-new-anchor spawn_turn.
6. Comet paths that exhaust within t are dropped from the new World.
7. Caches on the returned World are EMPTY (queries re-build from the
   new anchor).
8. A chained `BundleSearch.search(snapshot_at(t))` returns a valid
   Bundle without exception.
"""

from __future__ import annotations

import math

import pytest

from lib.trajectory_layer import (
    Bundle,
    BundleEvaluator,
    BundleSearch,
    LaunchSpec,
    World,
)


def _toy_world(planets: list, fleets: list, *,
               my_id: int = 0, step: int = 0,
               angular_velocity: float = 0.0,
               ) -> World:
    obs = {
        "step": step,
        "player": my_id,
        "angular_velocity": angular_velocity,
        "planets": planets,
        "initial_planets": planets,
        "fleets": fleets,
        "comet_planet_ids": [],
        "comets": [],
        "next_fleet_id": max((f[0] for f in fleets), default=-1) + 1,
    }
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# Trivial cases
# ---------------------------------------------------------------------------


def test_snapshot_at_zero_returns_self():
    """t=0 is the identity; the parent is returned unchanged."""
    world = _toy_world(
        planets=[[0, 0, 30.0, 80.0, 2.0, 50, 1]],
        fleets=[],
    )
    snap = world.snapshot_at(0)
    assert snap is world


def test_snapshot_at_negative_raises():
    """Snapshotting into the past is not a sensible operation."""
    world = _toy_world(
        planets=[[0, 0, 30.0, 80.0, 2.0, 50, 1]],
        fleets=[],
    )
    with pytest.raises(ValueError):
        world.snapshot_at(-1)


def test_snapshot_at_advances_step():
    """The new World's `step` is parent.step + t."""
    world = _toy_world(
        planets=[[0, 0, 30.0, 80.0, 2.0, 50, 1]],
        fleets=[],
        step=10,
    )
    snap = world.snapshot_at(5)
    assert snap.step == 15


# ---------------------------------------------------------------------------
# Ownership / position invariants — the load-bearing contract
# ---------------------------------------------------------------------------


def test_snapshot_at_ownership_invariant_static_world():
    """For a static no-fleet world, ownership_at on the snapshot at
    turn 0 must match the parent's ownership_at at turn t."""
    world = _toy_world(
        planets=[
            [0, 0, 30.0, 80.0, 2.0, 50, 1],   # ours, accruing 1/turn
            [1, 1, 70.0, 80.0, 2.0, 50, 1],   # enemy, accruing 1/turn
            [2, -1, 50.0, 50.0, 1.0, 0, 0],   # neutral (sun-edge), flat
        ],
        fleets=[],
    )
    for t in (1, 5, 10, 30):
        snap = world.snapshot_at(t)
        for p in world.planets:
            parent = world.ownership_at(p.id, t)
            child = snap.ownership_at(p.id, 0)
            assert child[0] == parent[0], (
                f"owner mismatch t={t} pid={p.id}: parent={parent} child={child}"
            )
            assert math.isclose(child[1], parent[1], abs_tol=1e-6), (
                f"ships mismatch t={t} pid={p.id}: parent={parent} child={child}"
            )


def test_snapshot_at_ownership_invariant_with_inbound_fleet():
    """Inbound fleet that arrives in the [0, t] window — its capture
    effect must already be baked into the snapshot's t=0 state.
    Geometry off the sun axis to avoid sun-kill confounds."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],   # source
            [1, -1, 40.0, 80.0, 1.0, 3, 0],   # weak neutral target
        ],
        # Fleet of 10 ships heading from src at (20, 80) toward (40, 80).
        fleets=[[0, 0, 22.0, 80.0, 0.0, 0, 10]],
    )
    # Pick t large enough that the fleet has arrived.
    t = 30
    snap = world.snapshot_at(t)
    parent_owner = world.ownership_at(1, t)[0]
    child_owner = snap.ownership_at(1, 0)[0]
    assert child_owner == parent_owner, (
        f"capture-by-arrival not baked into snapshot: "
        f"parent_owner_at_{t}={parent_owner} child_owner_at_0={child_owner}"
    )


def test_snapshot_at_planet_position_invariant_orbital():
    """For a rotating planet, `planet_position(p, 0)` on the snapshot
    matches `planet_position(p, t)` on the parent."""
    world = _toy_world(
        planets=[
            # Inner planet (within ROTATION_RADIUS_LIMIT=50) so it
            # rotates. Position offset slightly from CENTER=50.
            [0, 0, 60.0, 50.0, 1.0, 10, 1],
        ],
        fleets=[],
        angular_velocity=0.05,
    )
    for t in (1, 5, 20):
        snap = world.snapshot_at(t)
        parent_pos = world.planet_position(0, t)
        child_pos = snap.planet_position(0, 0)
        assert math.isclose(child_pos[0], parent_pos[0], abs_tol=1e-6)
        assert math.isclose(child_pos[1], parent_pos[1], abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Fleet rebuild
# ---------------------------------------------------------------------------


def test_snapshot_at_drops_already_arrived_fleet():
    """A fleet that has arrived by t should not appear in the new
    World's `fleets` (its effect is in the new planet state).
    Geometry off the sun axis."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, -1, 40.0, 80.0, 1.0, 3, 0],
        ],
        fleets=[[0, 0, 22.0, 80.0, 0.0, 0, 10]],
    )
    snap = world.snapshot_at(50)
    # Fleet should have arrived within 50 turns; not in new fleets.
    assert all(f.id != 0 for f in snap.fleets), (
        f"arrived fleet still in snapshot: ids={[f.id for f in snap.fleets]}"
    )


def test_snapshot_at_carries_future_scheduled_launch():
    """A future-scheduled launch (`with_candidate` at launch_turn=20)
    that hasn't fired by t=5 should be carried over with shifted
    spawn_turn / launch_turn."""
    world = _toy_world(
        planets=[
            [0, 0, 30.0, 50.0, 2.0, 50, 1],
            [1, -1, 50.0, 50.0, 1.0, 3, 0],
        ],
        fleets=[],
    )
    overlay = world.with_candidate(
        LaunchSpec(src_id=0, aim_angle=0.0, ships=10, owner=0,
                   launch_turn=20),
    )
    snap = overlay.snapshot_at(5)
    # The outgoing-launch entry should be shifted from launch_turn=20
    # to launch_turn=15.
    assert any(lt == 15 for (_src, lt, _ships)
               in snap._outgoing_launches), (
        f"future launch not carried with shifted turn: "
        f"_outgoing_launches={snap._outgoing_launches}"
    )
    # The synthetic fleet from the with_candidate is in overlay.fleets
    # with spawn_turn=20. After snapshot_at(5) it should have
    # spawn_turn=15.
    virtual = [f for f in snap.fleets if f.id < 0]
    assert virtual, "future-launch synthetic fleet missing after snapshot"
    assert virtual[0].spawn_turn == 15, (
        f"future fleet spawn_turn not shifted: {virtual[0].spawn_turn}"
    )


# ---------------------------------------------------------------------------
# Cache semantics
# ---------------------------------------------------------------------------


def test_snapshot_at_clears_caches():
    """Caches on the returned World are empty; queries on the
    snapshot rebuild from the new anchor."""
    world = _toy_world(
        planets=[[0, 0, 30.0, 80.0, 2.0, 50, 1]],
        fleets=[],
    )
    # Warm parent caches.
    _ = world.ownership_at(0, 10)
    assert world._timeline_cache, "parent timeline cache should be populated"
    snap = world.snapshot_at(5)
    assert snap._timeline_cache == {}, (
        f"snapshot caches not cleared: {snap._timeline_cache}"
    )
    assert snap._ledger_cache == {}
    assert snap._combat_log_cache == {}


# ---------------------------------------------------------------------------
# Chained search
# ---------------------------------------------------------------------------


def test_snapshot_at_chained_search_returns_valid_bundle():
    """A `BundleSearch.search()` on a snapshot_at output runs to
    completion and returns a Bundle (possibly empty)."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, 0, 25.0, 65.0, 2.0, 25, 1],
            [2, 1, 80.0, 20.0, 2.0, 30, 1],
            [3, 1, 75.0, 30.0, 2.0, 20, 1],
            [4, -1, 50.0, 30.0, 2.0, 10, 0],
            [5, -1, 50.0, 70.0, 2.0, 10, 0],
        ],
        fleets=[],
    )
    snap = world.snapshot_at(10)
    search = BundleSearch(
        evaluator=BundleEvaluator(horizon=20),
        max_depth=1, beam_width=2, candidates_per_source=2,
    )
    bundle = search.search(snap, my_id=0)
    # Bundle is a Bundle instance (possibly empty); search shouldn't
    # crash on a rolled-forward world.
    assert isinstance(bundle, Bundle)


def test_snapshot_at_is_idempotent_under_recomposition():
    """`world.snapshot_at(5)` then `.snapshot_at(3)` should give
    ownership equivalent to `world.snapshot_at(8)` (both at t=0)."""
    world = _toy_world(
        planets=[
            [0, 0, 30.0, 50.0, 2.0, 50, 1],
            [1, 1, 70.0, 50.0, 2.0, 50, 1],
            [2, -1, 50.0, 80.0, 1.0, 3, 0],
        ],
        fleets=[],
    )
    direct = world.snapshot_at(8)
    composed = world.snapshot_at(5).snapshot_at(3)
    for pid in (0, 1, 2):
        d = direct.ownership_at(pid, 0)
        c = composed.ownership_at(pid, 0)
        assert d[0] == c[0], f"owner mismatch pid={pid}: direct={d} composed={c}"
        assert math.isclose(d[1], c[1], abs_tol=1e-6), (
            f"ships mismatch pid={pid}: direct={d} composed={c}"
        )
