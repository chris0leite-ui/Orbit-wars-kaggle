"""Day 1 unit tests for agents.coord.enumerate_attack_bundles."""
from __future__ import annotations

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from agents.coord.main import (
    ARRIVAL_WINDOW_SLACK,
    Bundle,
    BundleKind,
    Leg,
    MAX_BUNDLE_SIZE,
    NEAREST_SOURCES_PER_TARGET,
    _cluster_arrival_windows,
    _emit_subsets,
    enumerate_attack_bundles,
)
from lib.intent import World
from lib.world_model import WorldModel


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    return Planet(pid, owner, x, y, radius, ships, production)


def _world(my_id, planets, *, step=0, omega=0.0):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": [],
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "step": step,
    }
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# Window clustering — pure-logic tests.
# ---------------------------------------------------------------------------

def test_cluster_empty():
    assert _cluster_arrival_windows([]) == []


def test_cluster_single_leg_one_window():
    leg = Leg(src_id=0, ships=10, angle=0.0, wait_N=0, eta=5)
    out = _cluster_arrival_windows([leg], slack=2)
    assert len(out) == 1
    assert out[0] == [leg]


def test_cluster_two_close_legs_one_anchor_window_contains_both():
    a = Leg(0, 10, 0.0, 0, 5)
    b = Leg(1, 10, 0.0, 0, 6)  # arrival_step within slack
    out = _cluster_arrival_windows([a, b], slack=2)
    # Anchor a's window contains both; anchor b's window contains just b.
    assert any(set(w) == {a, b} for w in out)


def test_cluster_two_far_legs_no_window_contains_both():
    a = Leg(0, 10, 0.0, 0, 5)
    b = Leg(1, 10, 0.0, 0, 15)
    out = _cluster_arrival_windows([a, b], slack=2)
    # No window should contain both — they're 10 ticks apart, slack=2.
    for w in out:
        assert not (a in w and b in w)


# ---------------------------------------------------------------------------
# Subset enumeration — pure-logic tests.
# ---------------------------------------------------------------------------

def test_emit_subsets_size_caps():
    legs = [Leg(i, 10, 0.0, 0, 5) for i in range(4)]
    out = _emit_subsets(legs, max_size=3)
    sizes = {len(s) for s in out}
    assert sizes == {1, 2, 3}
    # No subset has size 4.
    assert max(len(s) for s in out) == 3


def test_emit_subsets_no_source_repeats():
    a = Leg(src_id=0, ships=10, angle=0.0, wait_N=0, eta=5)
    b = Leg(src_id=0, ships=20, angle=0.0, wait_N=0, eta=6)  # same src!
    c = Leg(src_id=1, ships=10, angle=0.0, wait_N=0, eta=5)
    out = _emit_subsets([a, b, c], max_size=3)
    # The (a, b) subset must NOT appear — same src_id.
    for subset in out:
        srcs = [L.src_id for L in subset]
        assert len(srcs) == len(set(srcs)), f"duplicate src in {subset}"


def test_emit_subsets_singletons_always_emitted():
    a = Leg(0, 10, 0.0, 0, 5)
    b = Leg(1, 10, 0.0, 0, 7)
    out = _emit_subsets([a, b], max_size=3)
    singletons = [s for s in out if len(s) == 1]
    assert len(singletons) == 2


# ---------------------------------------------------------------------------
# End-to-end enumerate_attack_bundles — synthetic world tests.
# ---------------------------------------------------------------------------

def test_enumerate_no_my_planets_returns_empty():
    tgt = _planet(0, 1, 50.0, 50.0, ships=5)
    world = _world(0, [tgt])
    model = WorldModel.from_world(world)
    bundles = enumerate_attack_bundles([], [tgt], world, model, me=0, omega=0.0)
    assert bundles == []


def test_enumerate_no_targets_returns_empty():
    src = _planet(0, 0, 10.0, 50.0, ships=20)
    world = _world(0, [src])
    model = WorldModel.from_world(world)
    bundles = enumerate_attack_bundles([src], [], world, model, me=0, omega=0.0)
    assert bundles == []


def test_enumerate_own_targets_excluded():
    src = _planet(0, 0, 10.0, 50.0, ships=20)
    own_tgt = _planet(1, 0, 12.0, 50.0, ships=5)
    world = _world(0, [src, own_tgt])
    model = WorldModel.from_world(world)
    bundles = enumerate_attack_bundles(
        [src], [own_tgt], world, model, me=0, omega=0.0,
    )
    assert bundles == []  # own_tgt filtered out


def test_enumerate_single_source_produces_singletons():
    src = _planet(0, 0, 10.0, 50.0, ships=50, production=2)
    tgt = _planet(1, -1, 14.0, 50.0, ships=3, production=1)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    bundles = enumerate_attack_bundles(
        [src], [tgt], world, model, me=0, omega=0.0,
    )
    # All bundles should be singletons (only one source).
    assert bundles, "expected at least one singleton bundle"
    for b in bundles:
        assert len(b.legs) == 1
        assert b.kind == BundleKind.ATTACK
        assert b.target_id == 1
        assert b.legs[0].src_id == 0


def test_enumerate_two_sources_one_target_produces_pair():
    # Two reachable sources, one target — should get singletons AND a pair.
    src_a = _planet(0, 0, 10.0, 50.0, ships=30, production=2)
    src_b = _planet(1, 0, 10.0, 60.0, ships=30, production=2)
    tgt = _planet(2, -1, 14.0, 55.0, ships=3, production=1)
    world = _world(0, [src_a, src_b, tgt])
    model = WorldModel.from_world(world)
    bundles = enumerate_attack_bundles(
        [src_a, src_b], [tgt], world, model, me=0, omega=0.0,
    )
    sizes = {len(b.legs) for b in bundles}
    assert 1 in sizes, "expected singletons"
    assert 2 in sizes, "expected at least one 2-source pair"


def test_enumerate_three_sources_one_target_can_produce_triple():
    src_a = _planet(0, 0, 10.0, 50.0, ships=30, production=2)
    src_b = _planet(1, 0, 10.0, 60.0, ships=30, production=2)
    src_c = _planet(2, 0, 10.0, 40.0, ships=30, production=2)
    tgt = _planet(3, -1, 14.0, 50.0, ships=5, production=1)
    world = _world(0, [src_a, src_b, src_c, tgt])
    model = WorldModel.from_world(world)
    bundles = enumerate_attack_bundles(
        [src_a, src_b, src_c], [tgt], world, model, me=0, omega=0.0,
    )
    sizes = {len(b.legs) for b in bundles}
    # We may or may not get a triple depending on arrival-window alignment,
    # but max bundle size must never exceed MAX_BUNDLE_SIZE.
    assert max(sizes) <= MAX_BUNDLE_SIZE


def test_enumerate_respects_bundle_size_cap():
    # Five sources — bundles must never exceed MAX_BUNDLE_SIZE=3.
    sources = [
        _planet(i, 0, 10.0 + (i % 2), 50.0 + 2 * i, ships=20, production=2)
        for i in range(5)
    ]
    tgt = _planet(99, -1, 14.0, 55.0, ships=3, production=1)
    world = _world(0, sources + [tgt])
    model = WorldModel.from_world(world)
    bundles = enumerate_attack_bundles(
        sources, [tgt], world, model, me=0, omega=0.0,
    )
    for b in bundles:
        assert len(b.legs) <= MAX_BUNDLE_SIZE


def test_enumerate_respects_arrival_window():
    # Two sources at very different distances — should NOT be paired
    # because their arrival_steps fall outside ARRIVAL_WINDOW_SLACK.
    near = _planet(0, 0, 10.0, 50.0, ships=30, production=2)
    far = _planet(1, 0, 80.0, 50.0, ships=30, production=2)
    tgt = _planet(2, -1, 14.0, 50.0, ships=3, production=1)
    world = _world(0, [near, far, tgt])
    model = WorldModel.from_world(world)
    bundles = enumerate_attack_bundles(
        [near, far], [tgt], world, model, me=0, omega=0.0,
    )
    # Singletons exist; pair-with-both should NOT (arrival gap >> slack).
    for b in bundles:
        if len(b.legs) >= 2:
            arrivals = [L.arrival_step for L in b.legs]
            assert max(arrivals) - min(arrivals) <= ARRIVAL_WINDOW_SLACK


def test_enumerate_no_source_repeats_in_bundle():
    src = _planet(0, 0, 10.0, 50.0, ships=50, production=2)
    tgt = _planet(1, -1, 14.0, 50.0, ships=3, production=1)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    bundles = enumerate_attack_bundles(
        [src], [tgt], world, model, me=0, omega=0.0,
    )
    for b in bundles:
        srcs = [L.src_id for L in b.legs]
        assert len(srcs) == len(set(srcs))


def test_enumerate_unreachable_source_filtered():
    # Source too small to launch (ships < MIN_FLEET_SIZE) — excluded.
    too_small = _planet(0, 0, 10.0, 50.0, ships=1, production=2)
    tgt = _planet(1, -1, 14.0, 50.0, ships=3, production=1)
    world = _world(0, [too_small, tgt])
    model = WorldModel.from_world(world)
    bundles = enumerate_attack_bundles(
        [too_small], [tgt], world, model, me=0, omega=0.0,
    )
    assert bundles == []


def test_bundle_arrival_step_is_max_over_legs():
    src_a = _planet(0, 0, 10.0, 50.0, ships=30, production=2)
    src_b = _planet(1, 0, 10.0, 60.0, ships=30, production=2)
    tgt = _planet(2, -1, 14.0, 55.0, ships=3, production=1)
    world = _world(0, [src_a, src_b, tgt])
    model = WorldModel.from_world(world)
    bundles = enumerate_attack_bundles(
        [src_a, src_b], [tgt], world, model, me=0, omega=0.0,
    )
    for b in bundles:
        expected = max(L.arrival_step for L in b.legs)
        assert b.arrival_step == expected
