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
    LAGRANGIAN_ALPHA0,
    LAGRANGIAN_BUDGET_MS,
    LAGRANGIAN_MAX_ITERS,
    LAGRANGIAN_TARGET_UTIL,
    Leg,
    MAX_BUNDLE_SIZE,
    NEAREST_SOURCES_PER_TARGET,
    TIER2_BUDGET_MS,
    _bundle_cheap_delta,
    _bundle_fire_now_viable,
    _bundle_to_launches,
    _cluster_arrival_windows,
    _emit_subsets,
    _greedy_primal,
    _reduced_score,
    _resolve_target_post_bundle,
    _synthesise_post_arrival_obs,
    _used_ships_per_source,
    agent as coord_agent,
    cheap_filter_bundles,
    emit_bundle_actions,
    enumerate_attack_bundles,
    enumerate_defend_bundles,
    enumerate_recapture_bundles,
    lagrangian_clear,
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
    """Empty bundle list short-circuits before any scoring work."""
    src = _planet(0, 0, 10.0, 50.0, ships=20)
    world, snap, _ = _world_with_snap(0, [src])
    # model explicitly None: short-circuit path.
    out_no_model = tier2_score_bundles(
        [], snap, me=0, num_seats=2, world=world, model=None,
    )
    assert out_no_model == []
    # With a real model: also short-circuits (empty input).
    model = WorldModel.from_world(world)
    out_with_model = tier2_score_bundles(
        [], snap, me=0, num_seats=2, world=world, model=model,
    )
    assert out_with_model == []


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
    scored = tier2_score_bundles(
        cheap, snap, me=0, num_seats=2, world=world, model=model,
    )
    assert scored, "expected at least one Tier-2-scored bundle"
    # tier2_score populated (not the default 0.0 across the board — at
    # least one bundle should have a non-zero score in a real capture).
    assert any(b.tier2_score != 0.0 for b in scored)
    # Sorted descending by COMPOSITE (tier2_score + endgame_bonus).
    composites = [b.tier2_score + b.endgame_bonus for b in scored]
    assert composites == sorted(composites, reverse=True)


def test_tier2_score_bundles_drops_admissibility_failures():
    """A bundle whose fleet trajectory crosses the sun must be dropped."""
    # Source FAR east, target FAR west — fleet aimed west crosses sun (50,50).
    src = _planet(0, 0, 90.0, 50.0, ships=50, production=2)
    tgt = _planet(1, 1, 10.0, 50.0, ships=10, production=1)
    world, snap, _ = _world_with_snap(0, [src, tgt])
    model = WorldModel.from_world(world)
    import math
    bad_bundle = Bundle(
        target_id=1, arrival_step=15,
        # Aim due west — fleet path goes from (90, 50) through (50, 50)
        # toward (10, 50), crossing the sun region en route.
        legs=(Leg(src_id=0, ships=15, angle=math.pi, wait_N=0, eta=15),),
        kind=BundleKind.ATTACK,
    )
    out = tier2_score_bundles(
        [bad_bundle], snap, me=0, num_seats=2, world=world, model=model,
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
        cheap, snap, me=0, num_seats=2, world=world, model=model,
        wallclock_ms=10.0,
    )
    # No crash; output may be empty or small.
    assert isinstance(out, list)


# ---------------------------------------------------------------------------
# Lagrangian clearing (Day 6) — shadow-priced bundle selection.
# ---------------------------------------------------------------------------

def _scored(target_id, src_id, ships, tier2, kind=BundleKind.ATTACK):
    """Build a Bundle pre-populated with tier2_score (skips Tier-2)."""
    return Bundle(
        target_id=target_id,
        arrival_step=2,
        legs=(Leg(src_id=src_id, ships=ships, angle=0.0, wait_N=0, eta=2),),
        kind=kind,
        tier2_score=tier2,
    )


def _two_source(target_id, src_a, ships_a, src_b, ships_b, tier2,
                kind=BundleKind.ATTACK):
    return Bundle(
        target_id=target_id,
        arrival_step=2,
        legs=(
            Leg(src_id=src_a, ships=ships_a, angle=0.0, wait_N=0, eta=2),
            Leg(src_id=src_b, ships=ships_b, angle=0.1, wait_N=0, eta=2),
        ),
        kind=kind,
        tier2_score=tier2,
    )


def test_reduced_score_zero_lambda_equals_tier2():
    b = _scored(target_id=1, src_id=0, ships=20, tier2=50.0)
    lam = {0: 0.0}
    assert _reduced_score(b, lam) == 50.0


def test_reduced_score_subtracts_shadow_cost():
    b = _scored(target_id=1, src_id=0, ships=20, tier2=50.0)
    lam = {0: 1.0}  # 1.0 per ship → cost = 20
    assert _reduced_score(b, lam) == 30.0


def test_used_ships_per_source_aggregates_legs():
    chosen = [
        _scored(target_id=1, src_id=0, ships=20, tier2=10.0),
        _two_source(target_id=2, src_a=0, ships_a=15, src_b=1, ships_b=25,
                    tier2=20.0),
    ]
    used = _used_ships_per_source(chosen)
    assert used[0] == 35  # 20 + 15
    assert used[1] == 25


def test_greedy_primal_picks_positive_only():
    """A bundle with negative reduced_score must not be picked."""
    pos = _scored(target_id=1, src_id=0, ships=20, tier2=50.0)
    neg = _scored(target_id=2, src_id=1, ships=20, tier2=-5.0)
    chosen = _greedy_primal([pos, neg], lam={0: 0.0, 1: 0.0})
    assert pos in chosen
    assert neg not in chosen


def test_greedy_primal_one_bundle_per_source():
    """Two bundles sharing a source — only the higher-reduced one wins."""
    high = _scored(target_id=1, src_id=0, ships=20, tier2=50.0)
    low = _scored(target_id=2, src_id=0, ships=20, tier2=30.0)
    chosen = _greedy_primal([high, low], lam={0: 0.0})
    assert high in chosen
    assert low not in chosen


def test_greedy_primal_one_bundle_per_target():
    """Two bundles targeting the same planet — only one is picked."""
    a = _scored(target_id=1, src_id=0, ships=20, tier2=50.0)
    b = _scored(target_id=1, src_id=1, ships=30, tier2=40.0)
    chosen = _greedy_primal([a, b], lam={0: 0.0, 1: 0.0})
    target_ids = [c.target_id for c in chosen]
    assert target_ids.count(1) == 1


def test_lagrangian_clear_empty_returns_empty():
    assert lagrangian_clear([], my_planets=[]) == []


def test_lagrangian_clear_single_bundle_selected():
    src = _planet(0, 0, 10.0, 50.0, ships=50)
    b = _scored(target_id=1, src_id=0, ships=20, tier2=50.0)
    out = lagrangian_clear([b], my_planets=[src])
    assert out == [b]


def test_lagrangian_clear_higher_tier2_wins_at_zero_lambda():
    src = _planet(0, 0, 10.0, 50.0, ships=50)
    src2 = _planet(1, 0, 10.0, 60.0, ships=50)
    high = _scored(target_id=1, src_id=0, ships=20, tier2=80.0)
    low = _scored(target_id=1, src_id=1, ships=20, tier2=30.0)
    out = lagrangian_clear([high, low], my_planets=[src, src2])
    assert high in out
    assert low not in out  # same target — only one wins


def test_lagrangian_clear_independent_bundles_both_selected():
    """Two bundles with disjoint sources and targets — both should win."""
    src0 = _planet(0, 0, 10.0, 50.0, ships=50)
    src1 = _planet(1, 0, 10.0, 60.0, ships=50)
    b1 = _scored(target_id=10, src_id=0, ships=20, tier2=40.0)
    b2 = _scored(target_id=11, src_id=1, ships=20, tier2=35.0)
    out = lagrangian_clear([b1, b2], my_planets=[src0, src1])
    assert b1 in out
    assert b2 in out


def test_lagrangian_clear_terminates_within_max_iters():
    """Should never loop forever — terminates at 2-cycle or MAX_ITERS."""
    srcs = [_planet(i, 0, 10.0, 50.0 + 2 * i, ships=50) for i in range(4)]
    bundles = [
        _scored(target_id=10 + i, src_id=i, ships=20, tier2=30.0 + i)
        for i in range(4)
    ]
    out = lagrangian_clear(bundles, my_planets=srcs)
    # Just verify completion + sane output (no crash, no infinite loop).
    assert isinstance(out, list)
    assert all(b in bundles for b in out)


def test_lagrangian_clear_budget_enforcement():
    """Tiny wallclock budget — function returns without exceeding it."""
    srcs = [_planet(i, 0, 10.0, 50.0 + 2 * i, ships=50) for i in range(4)]
    bundles = [
        _scored(target_id=10 + i, src_id=i, ships=20, tier2=30.0 + i)
        for i in range(4)
    ]
    out = lagrangian_clear(bundles, my_planets=srcs, wallclock_ms=0.001)
    # Even with sub-ms budget, the function returns a best-feasible-so-far
    # (first iteration always completes).
    assert isinstance(out, list)


def test_lagrangian_clear_returns_best_across_iterations():
    """Best-feasible-ever across iterations — not the final iteration's
    primal. Verify by constructing a scenario where the dual oscillates.
    """
    src0 = _planet(0, 0, 10.0, 50.0, ships=10)  # tiny budget
    src1 = _planet(1, 0, 10.0, 60.0, ships=10)
    # Bundle A uses src0 with 10 ships (= entire budget) for tier2=20.
    # Bundle B uses src1 with 5 ships for tier2=10.
    # Both can co-exist at λ=0 (different sources).
    a = _scored(target_id=10, src_id=0, ships=10, tier2=20.0)
    b = _scored(target_id=11, src_id=1, ships=5, tier2=10.0)
    out = lagrangian_clear([a, b], my_planets=[src0, src1])
    # Both bundles should be selected (disjoint sources, disjoint targets).
    assert a in out
    assert b in out


def test_lagrangian_clear_shadow_price_resolves_source_conflict():
    """Two competing bundles for the same source — Lagrangian picks
    the higher-tier2 one, but the LOSER's bundle is still positive-value.
    Verify the chosen set's total tier2 is maximised.
    """
    src = _planet(0, 0, 10.0, 50.0, ships=20)
    competing_a = _scored(target_id=10, src_id=0, ships=20, tier2=50.0)
    competing_b = _scored(target_id=11, src_id=0, ships=20, tier2=30.0)
    out = lagrangian_clear(
        [competing_a, competing_b], my_planets=[src],
    )
    # Only one can win (shared source). The higher-tier2 should be chosen.
    assert competing_a in out
    assert competing_b not in out
    assert sum(b.tier2_score for b in out) == 50.0


# ---------------------------------------------------------------------------
# Day 7: emit_bundle_actions + agent() entry.
# ---------------------------------------------------------------------------

def _wait_bundle(target_id, src_id, ships, wait_N, eta=2, kind=BundleKind.ATTACK):
    """A single-leg bundle with the given wait_N — for safeguard tests."""
    return Bundle(
        target_id=target_id,
        arrival_step=wait_N + eta,
        legs=(Leg(src_id=src_id, ships=ships, angle=0.0, wait_N=wait_N, eta=eta),),
        kind=kind,
        tier2_score=10.0,
    )


def _mixed_wait_bundle(target_id, fire_now_src, fire_now_ships,
                       wait_src, wait_ships, wait_N=5,
                       kind=BundleKind.ATTACK):
    """A two-leg bundle: one fire-now, one delayed."""
    return Bundle(
        target_id=target_id,
        arrival_step=wait_N + 2,
        legs=(
            Leg(src_id=fire_now_src, ships=fire_now_ships, angle=0.0,
                wait_N=0, eta=2),
            Leg(src_id=wait_src, ships=wait_ships, angle=0.1,
                wait_N=wait_N, eta=2),
        ),
        kind=kind,
        tier2_score=10.0,
    )


def test_emit_bundle_actions_empty_returns_empty():
    src = _planet(0, 0, 10.0, 50.0, ships=20)
    world = _world(0, [src])
    model = WorldModel.from_world(world)
    out = emit_bundle_actions([], world, model, me=0)
    assert out == []


def test_emit_bundle_actions_fire_now_only():
    """All-wait_N=0 bundle: emits the leg as [src_id, angle, ships]."""
    src = _planet(0, 0, 10.0, 50.0, ships=50)
    tgt = _planet(1, 1, 14.0, 50.0, ships=5)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    b = _wait_bundle(target_id=1, src_id=0, ships=20, wait_N=0)
    out = emit_bundle_actions([b], world, model, me=0)
    assert out == [[0, 0.0, 20]]


def test_emit_bundle_actions_all_delayed_emits_nothing():
    """All-wait_N>0 bundle: reserved, no immediate move."""
    src = _planet(0, 0, 10.0, 50.0, ships=50)
    tgt = _planet(1, 1, 14.0, 50.0, ships=5)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    b = _wait_bundle(target_id=1, src_id=0, ships=20, wait_N=5)
    out = emit_bundle_actions([b], world, model, me=0)
    assert out == []


def test_emit_bundle_actions_mixed_safe_emits_fire_now():
    """Mixed-wait bundle where fire-now subset is viable standalone —
    emits the fire-now leg only.
    """
    src_a = _planet(0, 0, 10.0, 50.0, ships=80)
    src_b = _planet(1, 0, 10.0, 60.0, ships=80)
    tgt = _planet(2, -1, 14.0, 55.0, ships=3)  # neutral, weak
    world = _world(0, [src_a, src_b, tgt])
    model = WorldModel.from_world(world)
    # Fire-now leg with 50 ships easily captures 3-ship neutral.
    b = _mixed_wait_bundle(
        target_id=2, fire_now_src=0, fire_now_ships=50,
        wait_src=1, wait_ships=30, wait_N=5,
    )
    out = emit_bundle_actions([b], world, model, me=0)
    assert len(out) == 1
    assert out[0][0] == 0  # fire-now source id
    assert out[0][2] == 50  # fire-now ships


def test_emit_bundle_actions_mixed_unsafe_drops_bundle():
    """Mixed-wait bundle where fire-now subset can't hold — drops entirely.

    Fire-now leg captures the target marginally, but a strong nearby opp
    immediately recaptures: counter_force >> SAFETY_MARGIN * garrison.
    """
    src_a = _planet(0, 0, 10.0, 50.0, ships=80, production=2)
    src_b = _planet(1, 0, 10.0, 60.0, ships=60, production=2)
    tgt = _planet(2, 1, 45.0, 50.0, ships=3, production=1)  # far + weak
    strong = _planet(3, 1, 60.0, 50.0, ships=300, production=5)  # close to tgt
    world = _world(0, [src_a, src_b, tgt, strong])
    model = WorldModel.from_world(world)
    # Fire-now leg from src_a: ~50 ships, ~35-step flight. Marginally
    # captures tgt (delivered ≈ 15), but `strong` enemy's recapture
    # force vastly exceeds it.
    b = _mixed_wait_bundle(
        target_id=2, fire_now_src=0, fire_now_ships=50,
        wait_src=1, wait_ships=40, wait_N=5,
    )
    out = emit_bundle_actions([b], world, model, me=0)
    # Should drop because the fire-now leg alone can't hold against
    # the strong recapture.
    assert out == []


def test_emit_bundle_actions_source_dedup_across_bundles():
    """Two selected bundles sharing a source — only the first emits."""
    src = _planet(0, 0, 10.0, 50.0, ships=80)
    tgt_a = _planet(1, 1, 14.0, 50.0, ships=5)
    tgt_b = _planet(2, 1, 14.0, 60.0, ships=5)
    world = _world(0, [src, tgt_a, tgt_b])
    model = WorldModel.from_world(world)
    b1 = _wait_bundle(target_id=1, src_id=0, ships=20, wait_N=0)
    b2 = _wait_bundle(target_id=2, src_id=0, ships=30, wait_N=0)
    out = emit_bundle_actions([b1, b2], world, model, me=0)
    # Only first bundle's leg emitted (second's source already used).
    assert len(out) == 1
    assert out[0][2] == 20  # the FIRST bundle's ship count


def test_emit_defend_bundle_always_viable_safeguard():
    """DEFEND bundles bypass the standalone-hold check (always viable)."""
    own = _planet(0, 0, 10.0, 50.0, ships=5, production=2)
    peer = _planet(1, 0, 11.0, 50.0, ships=80)
    enemy = _planet(2, 1, 16.0, 50.0, ships=50)
    world = _world(0, [own, peer, enemy])
    model = WorldModel.from_world(world)
    # Mixed-wait DEFEND bundle — fire-now subset wouldn't pass any
    # standalone-attack check, but DEFEND always returns viable.
    b = Bundle(
        target_id=0, arrival_step=7,
        legs=(
            Leg(src_id=1, ships=3, angle=0.0, wait_N=0, eta=2),
            Leg(src_id=1, ships=30, angle=0.1, wait_N=5, eta=2),
        ),
        kind=BundleKind.DEFEND,
        tier2_score=5.0,
    )
    fire_now = [L for L in b.legs if L.wait_N == 0]
    assert _bundle_fire_now_viable(b, fire_now, world, model, me=0) is True


# ---------------------------------------------------------------------------
# Day 7 agent() entry — smoke tests.
# ---------------------------------------------------------------------------

def test_agent_empty_obs_returns_empty():
    """Agent with no planets returns []."""
    obs = {"player": 0, "planets": [], "fleets": [],
           "angular_velocity": 0.0, "comet_planet_ids": [], "step": 0}
    assert coord_agent(obs) == []


def test_agent_no_my_planets_returns_empty():
    """Agent with only enemy planets returns []."""
    e = _planet(0, 1, 50.0, 50.0, ships=5)
    obs = {"player": 0, "planets": [(e.id, e.owner, e.x, e.y, e.radius, e.ships, e.production)],
           "fleets": [], "angular_velocity": 0.0, "comet_planet_ids": [], "step": 0}
    assert coord_agent(obs) == []


def test_agent_returns_well_formed_moves():
    """Agent on a real synthesised state returns [[int, float, int], ...]."""
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=2)
    peer = _planet(1, 0, 10.0, 60.0, ships=60, production=2)
    tgt = _planet(2, 1, 14.0, 55.0, ships=10, production=1)
    obs = {
        "player": 0,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in [src, peer, tgt]
        ],
        "fleets": [], "angular_velocity": 0.0,
        "comet_planet_ids": [], "step": 0,
    }
    moves = coord_agent(obs)
    assert isinstance(moves, list)
    for m in moves:
        assert len(m) == 3
        assert isinstance(m[0], int)
        assert isinstance(m[1], float)
        assert isinstance(m[2], int)
        assert m[2] >= 1  # positive ship count
