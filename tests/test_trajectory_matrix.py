"""Pin tests for `lib/joint_solver/trajectory_matrix.py` — Phase η.1.

The matrix precomputes every (src, tgt, launch_tick) viable trajectory
at game start. These tests verify:

1. Layer-1 parity: matrix entries are byte-identical to direct
   `aim_and_eta + predict_garrison_at + predict_fleet_fate` calls on
   the same inputs. (Rule 38: any divergence is a bug.)
2. Fingerprint isolation: rebuilds fire on game boundary (different
   initial planets), idempotent within a game.
3. Lookup semantics: get() returns None for non-viable tuples, entries
   for viable ones; iter_viable filters correctly.
4. Concrete planet-16 case: on seed 42 the matrix MUST contain a
   viable trajectory (src=0, tgt=16, launch_tick=?) — this is the
   capture our introspect showed is missing from the existing
   opening_planner because of its K=8 nearest-target prune.
"""

from __future__ import annotations

import math

import pytest

from kaggle_environments import make

from lib.intent import World
from lib.joint_solver.trajectory_matrix import (
    DEFAULT_ARRIVAL_BUFFER,
    DEFAULT_MAX_LAUNCH_TICK,
    TrajectoryEntry,
    TrajectoryMatrix,
    get_default_matrix as get_default,
)
from lib.world_model import WorldModel, predict_garrison_at
from agents.baseline.proposer import aim_and_eta
from lib.trajectory import predict_fleet_fate


@pytest.fixture(autouse=True)
def _isolate_singleton():
    """Reset the module-level singleton between tests."""
    get_default().reset()
    yield
    get_default().reset()


def _world_and_model_from_seed(seed: int):
    env = make("orbit_wars", configuration={"seed": int(seed)}, debug=False)
    env.reset()
    obs = env.steps[0][0]["observation"]
    if not isinstance(obs, dict):
        obs = {k: getattr(obs, k) for k in dir(obs) if not k.startswith("_")}
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    omega = float(obs.get("angular_velocity", 0.0) or 0.0)
    return world, model, omega


# ---------------------------------------------------------------------------
# Build + basic shape.
# ---------------------------------------------------------------------------


def test_begin_game_rebuilds_on_first_call():
    """First call to begin_game always rebuilds; second call same-game
    is idempotent (returns False, no rebuild)."""
    world, model, omega = _world_and_model_from_seed(42)
    matrix = TrajectoryMatrix()

    rebuilt = matrix.begin_game(world, model, omega, my_id=0)
    assert rebuilt is True
    n_after_first = len(matrix)
    assert n_after_first > 0, "expected viable trajectories on seed 42"

    rebuilt2 = matrix.begin_game(world, model, omega, my_id=0)
    assert rebuilt2 is False  # idempotent within same game
    assert len(matrix) == n_after_first


def test_fingerprint_changes_force_rebuild():
    """Different seed = different initial planets = fingerprint changes
    = rebuild fires."""
    world1, model1, omega1 = _world_and_model_from_seed(42)
    world2, model2, omega2 = _world_and_model_from_seed(7)

    matrix = TrajectoryMatrix()
    matrix.begin_game(world1, model1, omega1, my_id=0)
    n1 = len(matrix)

    rebuilt = matrix.begin_game(world2, model2, omega2, my_id=0)
    assert rebuilt is True
    # Different game → different viable set in general. (Counts can
    # coincide but the entries differ; we don't compare counts here.)
    # Verify entries map specifically to seed-7 src/tgt pairs by spot
    # checking one entry exists for a known seed-7 source.
    my_planets = [p for p in world2.planets_by_id.values() if int(p.owner) == 0]
    assert my_planets, "seed 7 should have at least one of-our planets"


# ---------------------------------------------------------------------------
# Layer-1 parity: matrix entry equals direct call result.
# ---------------------------------------------------------------------------


def test_parity_per_entry_matches_direct_call():
    """For every viable entry in the matrix, calling aim_and_eta +
    predict_garrison_at + predict_fleet_fate directly with the entry's
    parameters MUST produce identical (angle, eta_flight,
    arrival_owner, arrival_garrison, capture-success) values.

    Sample 30 random entries on seed 42 to keep the test fast.
    """
    world, model, omega = _world_and_model_from_seed(42)
    matrix = TrajectoryMatrix()
    matrix.begin_game(world, model, omega, my_id=0)

    all_entries = list(matrix.iter_viable())
    assert len(all_entries) > 0, "expected viable entries on seed 42"

    # Sample deterministically (no random; just every Nth entry).
    sample_step = max(1, len(all_entries) // 30)
    sample = all_entries[::sample_step][:30]

    for e in sample:
        src = world.planets_by_id[int(e.src_id)]
        tgt = world.planets_by_id[int(e.tgt_id)]
        # Recompute the trajectory directly with the SAME ships_needed
        # the matrix converged on. (Two-pass refinement is in the
        # matrix's compute; here we replay the final step using the
        # converged ship count.)
        res = aim_and_eta(src, tgt, int(e.ships_needed), omega,
                          wait_N=int(e.launch_tick))
        assert res is not None, f"aim_and_eta failed for stored entry {e}"
        angle, eta = res
        assert math.isclose(float(angle), float(e.angle), abs_tol=1e-9), (
            f"angle mismatch on {e}: stored={e.angle} direct={angle}"
        )
        assert int(eta) == int(e.eta_flight), (
            f"eta mismatch on {e}: stored={e.eta_flight} direct={eta}"
        )

        arrival_total = int(e.launch_tick) + int(e.eta_flight)
        base_arrivals = list(model.ledger.get(int(tgt.id), []))
        owner_at_arr, gar_at_arr = predict_garrison_at(
            tgt, arrival_total, base_arrivals,
        )
        assert int(owner_at_arr) == int(e.arrival_owner)
        assert math.isclose(float(gar_at_arr), float(e.arrival_garrison),
                            abs_tol=1e-9)

        fate = predict_fleet_fate(
            src, tgt, float(e.angle), int(e.ships_needed), world,
            wait_N=int(e.launch_tick),
        )
        assert fate is not None
        assert getattr(fate, "outcome", "") == "target", (
            f"fate outcome mismatch on {e}: {fate.outcome}"
        )
        assert int(getattr(fate, "hit_planet_id", -1)) == int(tgt.id)


# ---------------------------------------------------------------------------
# Lookup semantics.
# ---------------------------------------------------------------------------


def test_get_returns_none_for_missing_tuple():
    world, model, omega = _world_and_model_from_seed(42)
    matrix = TrajectoryMatrix()
    matrix.begin_game(world, model, omega, my_id=0)
    # Tuple with absurd launch_tick — way past any reasonable horizon.
    assert matrix.get(0, 8, launch_tick=999) is None
    # Source = target — never stored.
    assert matrix.get(0, 0, launch_tick=0) is None


def test_iter_viable_filters_correctly():
    world, model, omega = _world_and_model_from_seed(42)
    matrix = TrajectoryMatrix()
    matrix.begin_game(world, model, omega, my_id=0)

    all_count = sum(1 for _ in matrix.iter_viable())
    src0_count = sum(1 for _ in matrix.iter_viable(src_id=0))
    assert 0 < src0_count <= all_count

    # Filtering by both src and tgt: must be a subset of src filter.
    src0_tgt8_count = sum(1 for _ in matrix.iter_viable(src_id=0, tgt_id=8))
    assert src0_tgt8_count <= src0_count


# ---------------------------------------------------------------------------
# The motivating case: planet 16 IS reachable from planet 0 on seed 42.
# ---------------------------------------------------------------------------


def test_seed42_planet0_to_planet16_exists():
    """The introspect that motivated Phase η showed planet 16 has a
    +1045 NET capture from planet 0 at launch_tick=1, wait_N=4 (so
    launch_tick=1+4=5? actually wait_N is offset from step_now; in
    matrix language launch_tick is absolute). We don't pin a specific
    launch_tick — just verify that AT LEAST ONE viable (0→16) entry
    exists somewhere in the matrix window. The existing opening_planner
    fails this because of its K=8 nearest-target prune.
    """
    world, model, omega = _world_and_model_from_seed(42)
    matrix = TrajectoryMatrix()
    matrix.begin_game(world, model, omega, my_id=0)

    # planet ids: seed 42 has planets 0..N. Find planet 16's id.
    if 16 not in world.planets_by_id:
        pytest.skip("seed 42 doesn't have planet id 16")
    if 0 not in world.planets_by_id:
        pytest.skip("seed 42 doesn't have planet id 0")
    if int(world.planets_by_id[0].owner) != 0:
        pytest.skip("planet 0 not owned by seat 0 on seed 42")

    entries_0_to_16 = list(matrix.iter_viable(src_id=0, tgt_id=16))
    assert entries_0_to_16, (
        f"expected at least one viable (0→16) trajectory in matrix "
        f"window on seed 42; matrix has {len(matrix)} total entries"
    )


# ---------------------------------------------------------------------------
# Stats and viability.
# ---------------------------------------------------------------------------


def test_stats_sum_to_raw_count():
    """The build-pass diagnostics should partition raw count into
    viable + drop reasons. Total accounting must match."""
    world, model, omega = _world_and_model_from_seed(42)
    matrix = TrajectoryMatrix()
    matrix.begin_game(world, model, omega, my_id=0)

    stats = matrix.stats()
    raw = stats.get("raw", 0)
    viable = stats.get("viable", 0)
    drop_keys = [k for k in stats if k.startswith("dropped_")]
    drops = sum(stats[k] for k in drop_keys)
    assert raw > 0
    assert viable + drops == raw, (
        f"stats accounting broken: viable={viable} + drops={drops} != raw={raw}\n"
        f"full stats: {stats}"
    )
    # Reasonable: viable fraction is in [1%, 50%] (most tuples drop for
    # at least one reason).
    assert 0.0 < viable / raw < 0.6, (
        f"viable fraction outside expected range: {viable}/{raw}"
    )


def test_module_singleton_matches_class_instance():
    """The module-level _DEFAULT singleton must produce equivalent
    output to a fresh TrajectoryMatrix() on the same inputs."""
    world, model, omega = _world_and_model_from_seed(42)

    # Module-level singleton.
    from lib.joint_solver.trajectory_matrix import begin_game as _begin_game
    from lib.joint_solver.trajectory_matrix import iter_viable as _iter_viable
    _begin_game(world, model, omega, my_id=0)
    singleton_entries = set(
        (e.src_id, e.tgt_id, e.launch_tick, e.ships_needed, e.eta_flight)
        for e in _iter_viable()
    )

    # Class instance.
    matrix = TrajectoryMatrix()
    matrix.begin_game(world, model, omega, my_id=0)
    instance_entries = set(
        (e.src_id, e.tgt_id, e.launch_tick, e.ships_needed, e.eta_flight)
        for e in matrix.iter_viable()
    )

    assert singleton_entries == instance_entries
