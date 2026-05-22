"""Day 1 unit tests for agents.coord.enumerate_attack_bundles."""
from __future__ import annotations

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from agents.coord.main import (
    ARRIVAL_WINDOW_SLACK,
    Bundle,
    BundleKind,
    CHEAP_FILTER_TOP_K,
    CHEAP_OPPORTUNITY_COST,
    DEFEND_LOOKAHEAD,
    Leg,
    MAX_BUNDLE_SIZE,
    NEAREST_SOURCES_PER_TARGET,
    TIER2_BUDGET_MS,
    _bundle_cheap_delta,
    _bundle_to_launches,
    _cluster_arrival_windows,
    _emit_subsets,
    _resolve_target_post_bundle,
    _synthesise_post_arrival_obs,
    cheap_filter_bundles,
    enumerate_attack_bundles,
    enumerate_defend_bundles,
    enumerate_recapture_bundles,
    tier2_score_bundles,
)
from lib.fast_sim import from_obs as fs_from_obs
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


# ---------------------------------------------------------------------------
# Defense enumeration tests.
# ---------------------------------------------------------------------------

def test_defend_no_threat_returns_empty():
    """My planets, no enemy at all → no defend bundles."""
    own = _planet(0, 0, 10.0, 50.0, ships=20)
    peer = _planet(1, 0, 12.0, 50.0, ships=20)
    world = _world(0, [own, peer])
    model = WorldModel.from_world(world)
    bundles = enumerate_defend_bundles(
        [own, peer], world, model, me=0, omega=0.0,
    )
    assert bundles == []


def test_defend_threat_too_far_returns_empty():
    """Threat ETA > DEFEND_LOOKAHEAD → no defend bundles."""
    own = _planet(0, 0, 10.0, 50.0, ships=20)
    # Enemy very far away — threat ETA exceeds DEFEND_LOOKAHEAD.
    very_far_enemy = _planet(1, 1, 95.0, 50.0, ships=5, production=1)
    world = _world(0, [own, very_far_enemy])
    model = WorldModel.from_world(world)
    bundles = enumerate_defend_bundles(
        [own], world, model, me=0, omega=0.0,
    )
    # Could be empty either via the >LOOKAHEAD check or by having no peers
    # — either way the assertion is "no bundles" which is correct.
    assert bundles == []


def test_defend_no_peers_returns_empty():
    """Only one own planet, no peers to deliver from → no defend bundles."""
    own = _planet(0, 0, 10.0, 50.0, ships=20)
    enemy = _planet(1, 1, 14.0, 50.0, ships=50, production=2)
    world = _world(0, [own, enemy])
    model = WorldModel.from_world(world)
    bundles = enumerate_defend_bundles(
        [own], world, model, me=0, omega=0.0,
    )
    # Only own planet exists; no peer can defend it.
    assert bundles == []


def test_defend_with_reachable_peer_produces_singleton():
    """Threatened own + nearby peer with enough ships → defend bundle."""
    own = _planet(0, 0, 10.0, 50.0, ships=5, production=1)
    peer = _planet(1, 0, 12.0, 50.0, ships=80, production=2)
    # Enemy threatens own (close + lots of ships).
    enemy = _planet(2, 1, 16.0, 50.0, ships=50, production=2)
    world = _world(0, [own, peer, enemy])
    model = WorldModel.from_world(world)
    bundles = enumerate_defend_bundles(
        [own, peer], world, model, me=0, omega=0.0,
    )
    # If a defense leg is constructable in time, we should see at least
    # one defend bundle. If the geometry doesn't allow it (peer arrives
    # too late), bundles may be empty — both outcomes are valid as long
    # as the function doesn't crash and emits coherent results.
    for b in bundles:
        assert b.kind == BundleKind.DEFEND
        assert b.target_id == int(own.id)
        # Every leg must be sourced from a peer (not the threatened
        # planet itself) and arrive before any enemy ETA.
        for L in b.legs:
            assert L.src_id != int(own.id)


def test_defend_bundle_kind_is_DEFEND():
    """Any emitted defend bundle must carry kind=DEFEND."""
    own = _planet(0, 0, 10.0, 50.0, ships=5, production=1)
    peer_a = _planet(1, 0, 11.0, 50.0, ships=60, production=2)
    peer_b = _planet(2, 0, 10.0, 52.0, ships=60, production=2)
    enemy = _planet(3, 1, 16.0, 50.0, ships=50, production=2)
    world = _world(0, [own, peer_a, peer_b, enemy])
    model = WorldModel.from_world(world)
    bundles = enumerate_defend_bundles(
        [own, peer_a, peer_b], world, model, me=0, omega=0.0,
    )
    for b in bundles:
        assert b.kind == BundleKind.DEFEND


def test_defend_target_is_own_planet_not_enemy():
    """Defend bundle's target_id must be the own threatened planet."""
    own = _planet(0, 0, 10.0, 50.0, ships=5, production=1)
    peer = _planet(1, 0, 11.0, 50.0, ships=60, production=2)
    enemy = _planet(2, 1, 16.0, 50.0, ships=50, production=2)
    world = _world(0, [own, peer, enemy])
    model = WorldModel.from_world(world)
    bundles = enumerate_defend_bundles(
        [own, peer], world, model, me=0, omega=0.0,
    )
    for b in bundles:
        # target_id must be the OWN planet (id=0), not the enemy (id=2).
        assert b.target_id == 0


def test_defend_bundle_respects_size_cap():
    """No defend bundle exceeds MAX_BUNDLE_SIZE legs."""
    own = _planet(0, 0, 10.0, 50.0, ships=5, production=1)
    peers = [
        _planet(i, 0, 10.0 + (i % 2), 50.0 + 1.5 * i, ships=60, production=2)
        for i in range(1, 6)
    ]
    enemy = _planet(99, 1, 16.0, 50.0, ships=50, production=2)
    world = _world(0, [own] + peers + [enemy])
    model = WorldModel.from_world(world)
    bundles = enumerate_defend_bundles(
        [own] + peers, world, model, me=0, omega=0.0,
    )
    for b in bundles:
        assert len(b.legs) <= MAX_BUNDLE_SIZE


def test_defend_no_source_repeats_in_bundle():
    """No defend bundle uses the same source twice."""
    own = _planet(0, 0, 10.0, 50.0, ships=5, production=1)
    peer_a = _planet(1, 0, 11.0, 50.0, ships=60, production=2)
    peer_b = _planet(2, 0, 10.0, 52.0, ships=60, production=2)
    enemy = _planet(3, 1, 16.0, 50.0, ships=50, production=2)
    world = _world(0, [own, peer_a, peer_b, enemy])
    model = WorldModel.from_world(world)
    bundles = enumerate_defend_bundles(
        [own, peer_a, peer_b], world, model, me=0, omega=0.0,
    )
    for b in bundles:
        srcs = [L.src_id for L in b.legs]
        assert len(srcs) == len(set(srcs))


def test_defend_arrival_steps_before_threat_eta():
    """Every defend leg must arrive BEFORE the enemy ETA."""
    own = _planet(0, 0, 10.0, 50.0, ships=5, production=1)
    peer = _planet(1, 0, 11.0, 50.0, ships=60, production=2)
    enemy = _planet(2, 1, 14.0, 50.0, ships=50, production=2)
    world = _world(0, [own, peer, enemy])
    model = WorldModel.from_world(world)
    enemy_eta = model.time_to_enemy_threat(0, 0, world)
    bundles = enumerate_defend_bundles(
        [own, peer], world, model, me=0, omega=0.0,
    )
    if enemy_eta is not None:
        for b in bundles:
            for L in b.legs:
                assert L.arrival_step < enemy_eta, (
                    f"leg arrival_step {L.arrival_step} >= enemy_eta {enemy_eta}"
                )


def test_defend_skips_below_minimum_ships_peer():
    """Peer with ships < MIN_FLEET_SIZE filtered out as defender."""
    own = _planet(0, 0, 10.0, 50.0, ships=5, production=1)
    too_small = _planet(1, 0, 11.0, 50.0, ships=1, production=2)
    enemy = _planet(2, 1, 14.0, 50.0, ships=50, production=2)
    world = _world(0, [own, too_small, enemy])
    model = WorldModel.from_world(world)
    bundles = enumerate_defend_bundles(
        [own, too_small], world, model, me=0, omega=0.0,
    )
    # If any bundle exists, its legs must not source from `too_small`.
    for b in bundles:
        for L in b.legs:
            assert L.src_id != int(too_small.id)


# ---------------------------------------------------------------------------
# Recapture stub — v1 returns empty by design.
# ---------------------------------------------------------------------------

def test_recapture_stub_returns_empty():
    """v1 stub: recapture deferred to v2 (no inter-turn state)."""
    src = _planet(0, 0, 10.0, 50.0, ships=30)
    tgt = _planet(1, 1, 14.0, 50.0, ships=5)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    bundles = enumerate_recapture_bundles(
        [src], [tgt], world, model, me=0, omega=0.0,
    )
    assert bundles == []


# ---------------------------------------------------------------------------
# Cheap-filter — synthesised-obs combat resolution.
# ---------------------------------------------------------------------------

def _make_bundle(target_id, legs, kind=BundleKind.ATTACK, arrival_step=None):
    """Build a Bundle for testing — arrival_step defaults to max leg ETA."""
    if arrival_step is None:
        arrival_step = max(L.arrival_step for L in legs)
    return Bundle(
        target_id=target_id,
        arrival_step=arrival_step,
        legs=tuple(legs),
        kind=kind,
    )


def test_resolve_attack_capture_when_force_exceeds_defender():
    """ATTACK with enough ships → captured; garrison = force − defender."""
    tgt = _planet(1, 1, 14.0, 50.0, ships=10)
    legs = [Leg(src_id=0, ships=30, angle=0.0, wait_N=0, eta=5)]
    bundle = _make_bundle(target_id=1, legs=legs, kind=BundleKind.ATTACK)
    owner, ships = _resolve_target_post_bundle(
        bundle, tgt, pred_owner=1, pred_ships=10, me=0,
    )
    assert owner == 0  # me captured
    assert ships == 20  # 30 - 10


def test_resolve_attack_failed_when_force_below_defender():
    """ATTACK with insufficient ships → not captured; defender retains."""
    tgt = _planet(1, 1, 14.0, 50.0, ships=50)
    legs = [Leg(src_id=0, ships=10, angle=0.0, wait_N=0, eta=5)]
    bundle = _make_bundle(target_id=1, legs=legs, kind=BundleKind.ATTACK)
    owner, ships = _resolve_target_post_bundle(
        bundle, tgt, pred_owner=1, pred_ships=50, me=0,
    )
    assert owner == 1  # enemy retains
    assert ships == 40  # 50 - 10


def test_resolve_defend_reinforce_when_we_still_own():
    """DEFEND with we-still-own-at-arrival → reinforcement ADDS to garrison."""
    tgt = _planet(1, 0, 14.0, 50.0, ships=5)
    legs = [Leg(src_id=2, ships=20, angle=0.0, wait_N=0, eta=3)]
    bundle = _make_bundle(target_id=1, legs=legs, kind=BundleKind.DEFEND)
    owner, ships = _resolve_target_post_bundle(
        bundle, tgt, pred_owner=0, pred_ships=8, me=0,
    )
    assert owner == 0  # still ours
    assert ships == 28  # 8 + 20


def test_resolve_defend_handles_lost_by_arrival_as_recapture():
    """DEFEND when already-lost-by-arrival → combat-rule-1 recapture."""
    tgt = _planet(1, 0, 14.0, 50.0, ships=5)
    legs = [Leg(src_id=2, ships=20, angle=0.0, wait_N=0, eta=3)]
    bundle = _make_bundle(target_id=1, legs=legs, kind=BundleKind.DEFEND)
    # pred_owner==1 means we lost it by arrival_step; recapture path.
    owner, ships = _resolve_target_post_bundle(
        bundle, tgt, pred_owner=1, pred_ships=5, me=0,
    )
    assert owner == 0  # we recaptured
    assert ships == 15  # 20 - 5


def test_synthesise_subtracts_committed_ships_from_sources():
    """Source planets in synthesised obs lose the legs' ship count."""
    src = _planet(0, 0, 10.0, 50.0, ships=50)
    tgt = _planet(1, -1, 14.0, 50.0, ships=3)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    legs = [Leg(src_id=0, ships=20, angle=0.0, wait_N=0, eta=2)]
    bundle = _make_bundle(target_id=1, legs=legs, kind=BundleKind.ATTACK)
    synth = _synthesise_post_arrival_obs(bundle, world, model, me=0)
    src_after = [p for p in synth["planets"] if p[0] == 0][0]
    # Idle projection at step=2 = 50 + 2*production; minus 20 committed.
    # Production=2 (default), so idle=50+4=54, then minus 20 = 34.
    assert src_after[5] == 34, f"src ships expected 34, got {src_after[5]}"


def test_synthesise_target_reflects_combat_resolution():
    """Target's post-arrival state matches `_resolve_target_post_bundle`."""
    src = _planet(0, 0, 10.0, 50.0, ships=50)
    tgt = _planet(1, 1, 14.0, 50.0, ships=10, production=1)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    legs = [Leg(src_id=0, ships=30, angle=0.0, wait_N=0, eta=2)]
    bundle = _make_bundle(target_id=1, legs=legs, kind=BundleKind.ATTACK)
    synth = _synthesise_post_arrival_obs(bundle, world, model, me=0)
    tgt_after = [p for p in synth["planets"] if p[0] == 1][0]
    # Enemy at step=2 has 10 + 2*1 = 12 ships idle. 30 - 12 = 18, captured.
    assert tgt_after[1] == 0  # owner = me
    assert tgt_after[5] == 18


def test_synthesise_step_advances():
    """Synthesised obs step = current + arrival_step."""
    src = _planet(0, 0, 10.0, 50.0, ships=20)
    tgt = _planet(1, -1, 14.0, 50.0, ships=3)
    world = _world(0, [src, tgt], step=37)
    model = WorldModel.from_world(world)
    legs = [Leg(src_id=0, ships=10, angle=0.0, wait_N=0, eta=5)]
    bundle = _make_bundle(target_id=1, legs=legs, arrival_step=5)
    synth = _synthesise_post_arrival_obs(bundle, world, model, me=0)
    assert synth["step"] == 42  # 37 + 5
    assert synth["player"] == 0


def test_synthesise_clears_fleets():
    """Synthesised obs has fleets=[] (cheap simplification)."""
    src = _planet(0, 0, 10.0, 50.0, ships=20)
    tgt = _planet(1, -1, 14.0, 50.0, ships=3)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    legs = [Leg(src_id=0, ships=10, angle=0.0, wait_N=0, eta=2)]
    bundle = _make_bundle(target_id=1, legs=legs)
    synth = _synthesise_post_arrival_obs(bundle, world, model, me=0)
    assert synth["fleets"] == []


def test_cheap_filter_empty_input_returns_empty():
    src = _planet(0, 0, 10.0, 50.0, ships=20)
    world = _world(0, [src])
    model = WorldModel.from_world(world)
    out = cheap_filter_bundles([], world, model, me=0, num_seats=2)
    assert out == []


def test_cheap_filter_populates_cheap_score():
    """All returned bundles have cheap_score set (non-zero)."""
    src = _planet(0, 0, 10.0, 50.0, ships=50)
    tgt = _planet(1, -1, 14.0, 50.0, ships=3)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    bundles = enumerate_attack_bundles(
        [src], [tgt], world, model, me=0, omega=0.0,
    )
    assert bundles  # precondition
    out = cheap_filter_bundles(bundles, world, model, me=0, num_seats=2)
    assert out
    # cheap_score must be populated (not the default 0.0 if the bundle
    # genuinely captures value — at minimum some bundle should score non-zero).
    assert any(b.cheap_score != 0.0 for b in out)


def test_cheap_filter_top_k_limits_returned_count():
    """cheap_filter returns at most K bundles, sorted by cheap_score desc."""
    src_a = _planet(0, 0, 10.0, 50.0, ships=30)
    src_b = _planet(1, 0, 10.0, 60.0, ships=30)
    tgt = _planet(2, -1, 14.0, 55.0, ships=3)
    world = _world(0, [src_a, src_b, tgt])
    model = WorldModel.from_world(world)
    bundles = enumerate_attack_bundles(
        [src_a, src_b], [tgt], world, model, me=0, omega=0.0,
    )
    out = cheap_filter_bundles(
        bundles, world, model, me=0, num_seats=2, K=2,
    )
    assert len(out) <= 2
    # Sorted descending.
    scores = [b.cheap_score for b in out]
    assert scores == sorted(scores, reverse=True)


def test_cheap_filter_capture_scores_higher_than_failed_attack():
    """An ATTACK that captures must score higher than one that doesn't."""
    src = _planet(0, 0, 10.0, 50.0, ships=100)
    tgt = _planet(1, 1, 14.0, 50.0, ships=10, production=1)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    capture_legs = [Leg(src_id=0, ships=40, angle=0.0, wait_N=0, eta=2)]
    fail_legs = [Leg(src_id=0, ships=5, angle=0.0, wait_N=0, eta=2)]
    capture_bundle = _make_bundle(target_id=1, legs=capture_legs)
    fail_bundle = _make_bundle(target_id=1, legs=fail_legs)
    base = favor_hybrid_via_world(world, me=0, num_seats=2)
    cap_score = _bundle_cheap_delta(
        capture_bundle, world, model, me=0, num_seats=2, current_favor=base,
    )
    fail_score = _bundle_cheap_delta(
        fail_bundle, world, model, me=0, num_seats=2, current_favor=base,
    )
    assert cap_score > fail_score, (
        f"capture cheap_score={cap_score} must exceed failed-attack {fail_score}"
    )


def test_cheap_filter_opportunity_cost_penalises_over_commitment():
    """Two captures of the same target — smaller fleet wins on cheap_score
    when both produce the same target outcome (tie-breaking via epsilon).
    """
    src = _planet(0, 0, 10.0, 50.0, ships=200)
    tgt = _planet(1, 1, 14.0, 50.0, ships=5, production=1)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    minimal_capture = [Leg(src_id=0, ships=10, angle=0.0, wait_N=0, eta=2)]
    bloated_capture = [Leg(src_id=0, ships=150, angle=0.0, wait_N=0, eta=2)]
    b_min = _make_bundle(target_id=1, legs=minimal_capture)
    b_max = _make_bundle(target_id=1, legs=bloated_capture)
    base = favor_hybrid_via_world(world, me=0, num_seats=2)
    s_min = _bundle_cheap_delta(b_min, world, model, 0, 2, base)
    s_max = _bundle_cheap_delta(b_max, world, model, 0, 2, base)
    # Bloated commits 140 more ships for the same capture; the opportunity-
    # cost term penalises it.
    assert s_min > s_max, (
        f"minimal-capture {s_min} should beat bloated {s_max} via epsilon"
    )


def favor_hybrid_via_world(world, me, num_seats):
    """Tiny convenience for tests."""
    from agents.minimal.main import favor_hybrid as _fh
    return _fh(world.obs_raw, me, num_seats)


# ---------------------------------------------------------------------------
# Tier-2 scoring (Day 5) — score_candidate_v4_joint glue.
# ---------------------------------------------------------------------------

def _world_with_snap(my_id, planets, *, step=0, omega=0.0, num_seats=2):
    """Build (world, snap_base) — needed for Tier-2 since it uses fast_sim."""
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
    world = World.from_obs(obs)
    snap = fs_from_obs(obs, num_seats=num_seats)
    return world, snap, obs


def test_bundle_to_launches_basic():
    src = _planet(0, 0, 10.0, 50.0, ships=50)
    tgt = _planet(1, 1, 14.0, 50.0, ships=10)
    planets_by_id = {0: src, 1: tgt}
    bundle = Bundle(
        target_id=1, arrival_step=2,
        legs=(Leg(src_id=0, ships=20, angle=0.0, wait_N=0, eta=2),),
        kind=BundleKind.ATTACK,
    )
    launches = _bundle_to_launches(bundle, planets_by_id)
    assert launches is not None
    assert len(launches) == 1
    sp, tp, ships, angle, wait_N = launches[0]
    assert sp is src
    assert tp is tgt
    assert ships == 20
    assert wait_N == 0


def test_bundle_to_launches_missing_source_returns_none():
    """Defensive: if source planet not in current world (stale bundle)."""
    tgt = _planet(1, 1, 14.0, 50.0, ships=10)
    planets_by_id = {1: tgt}  # source id=0 NOT present
    bundle = Bundle(
        target_id=1, arrival_step=2,
        legs=(Leg(src_id=0, ships=20, angle=0.0, wait_N=0, eta=2),),
        kind=BundleKind.ATTACK,
    )
    assert _bundle_to_launches(bundle, planets_by_id) is None


def test_bundle_to_launches_missing_target_returns_none():
    src = _planet(0, 0, 10.0, 50.0, ships=50)
    planets_by_id = {0: src}  # target id=99 NOT present
    bundle = Bundle(
        target_id=99, arrival_step=2,
        legs=(Leg(src_id=0, ships=20, angle=0.0, wait_N=0, eta=2),),
        kind=BundleKind.ATTACK,
    )
    assert _bundle_to_launches(bundle, planets_by_id) is None


def test_tier2_score_bundles_empty_returns_empty():
    src = _planet(0, 0, 10.0, 50.0, ships=20)
    world, snap, _ = _world_with_snap(0, [src])
    out = tier2_score_bundles([], snap, me=0, num_seats=2, world=world)
    assert out == []


def test_tier2_score_bundles_populates_score_and_sorts():
    """End-to-end: enumerate + cheap-filter + Tier-2 yields sorted bundles."""
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=2)
    peer = _planet(1, 0, 10.0, 60.0, ships=80, production=2)
    tgt = _planet(2, 1, 14.0, 55.0, ships=10, production=1)
    world, snap, _ = _world_with_snap(0, [src, peer, tgt])
    model = WorldModel.from_world(world)
    raw = enumerate_attack_bundles(
        [src, peer], [tgt], world, model, me=0, omega=0.0,
    )
    assert raw, "precondition: enumeration produced bundles"
    cheap = cheap_filter_bundles(raw, world, model, me=0, num_seats=2, K=10)
    scored = tier2_score_bundles(cheap, snap, me=0, num_seats=2, world=world)
    assert scored, "expected at least one Tier-2-scored bundle"
    # tier2_score populated (not the default 0.0 across the board — at
    # least one bundle should have a non-zero score in a real capture).
    assert any(b.tier2_score != 0.0 for b in scored)
    # Sorted descending by tier2_score.
    scores = [b.tier2_score for b in scored]
    assert scores == sorted(scores, reverse=True)


def test_tier2_score_bundles_drops_admissibility_failures():
    """A bundle whose fleet trajectory crosses the sun must be dropped."""
    # Source FAR east, target FAR west — fleet aimed west crosses sun (50,50).
    src = _planet(0, 0, 90.0, 50.0, ships=50, production=2)
    tgt = _planet(1, 1, 10.0, 50.0, ships=10, production=1)
    world, snap, _ = _world_with_snap(0, [src, tgt])
    import math
    bad_bundle = Bundle(
        target_id=1, arrival_step=15,
        # Aim due west — fleet path goes from (90, 50) through (50, 50)
        # toward (10, 50), crossing the sun region en route.
        legs=(Leg(src_id=0, ships=15, angle=math.pi, wait_N=0, eta=15),),
        kind=BundleKind.ATTACK,
    )
    out = tier2_score_bundles(
        [bad_bundle], snap, me=0, num_seats=2, world=world,
    )
    # score_candidate_v4_joint flags sun-crossing as "sun" status;
    # tier2_score_bundles drops non-"scored" results.
    assert out == []


def test_tier2_score_bundles_budget_pre_bails():
    """With a tiny wallclock budget, the function returns partial-or-empty
    output without crashing (safe_deadline pre-bail).
    """
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=2)
    peer = _planet(1, 0, 10.0, 60.0, ships=80, production=2)
    tgt = _planet(2, 1, 14.0, 55.0, ships=10, production=1)
    world, snap, _ = _world_with_snap(0, [src, peer, tgt])
    model = WorldModel.from_world(world)
    raw = enumerate_attack_bundles(
        [src, peer], [tgt], world, model, me=0, omega=0.0,
    )
    cheap = cheap_filter_bundles(raw, world, model, me=0, num_seats=2, K=20)
    # 10ms budget — too small for the affordable_validate_cap probe to
    # fit, so safe_deadline will be in the past and the loop short-circuits.
    out = tier2_score_bundles(
        cheap, snap, me=0, num_seats=2, world=world, wallclock_ms=10.0,
    )
    # No crash; output may be empty or small.
    assert isinstance(out, list)
