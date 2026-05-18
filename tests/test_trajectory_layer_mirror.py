"""Phase 8 tests for lib.trajectory_layer — opp_overlays + mirror-search.

Pins:

1. BundleEvaluator.score with `opp_overlays=None` matches the prior
   (Phase 7) behaviour exactly — regression guard.
2. An opp_overlay that captures one of our planets reduces our score
   relative to the passive-world score (counterplay shows up).
3. An infeasible opp_overlay is silently dropped (no exception).
4. `predict_opp_bundles_via_mirror_search` with `depth=0` returns {}.
5. With `depth=1` and one opp who has a clear capture, the predicted
   bundle is non-empty.
6. With no opps, returns {}.
7. End-to-end: BundleSearch.search with `opp_overlays=mirror_dict`
   produces a bundle whose evaluator score (with the same overlays)
   beats the empty-bundle score under the same overlays — the search
   isn't fooled into "do nothing" by counterplay.
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
    predict_opp_bundles_via_mirror_search,
)


def _toy_world(planets: list, fleets: list, *, my_id: int = 0,
               ) -> World:
    obs = {
        "step": 0,
        "player": my_id,
        "angular_velocity": 0.0,
        "planets": planets,
        "initial_planets": planets,
        "fleets": fleets,
        "comet_planet_ids": [],
        "comets": [],
        "next_fleet_id": max((f[0] for f in fleets), default=-1) + 1,
    }
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# BundleEvaluator opp_overlays
# ---------------------------------------------------------------------------


def test_evaluator_opp_overlays_none_matches_phase7():
    """`opp_overlays=None` must score identically to the prior
    Phase 7 path (regression guard so the new kwarg doesn't shift
    any caller's numbers)."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, 1, 80.0, 80.0, 2.0, 50, 1],
            [2, -1, 50.0, 90.0, 1.0, 3, 0],
        ],
        fleets=[],
    )
    bundle = Bundle(launches=(
        LaunchSpec(src_id=0, aim_angle=math.atan2(10.0, 30.0),
                   ships=5, owner=0, launch_turn=0),
    ))
    ev = BundleEvaluator(horizon=30)
    a = ev.score(world, bundle, my_id=0)
    b = ev.score(world, bundle, my_id=0, opp_overlays=None)
    assert a == b


def test_evaluator_opp_overlay_capture_lowers_my_score():
    """If the opp's overlay captures one of our planets at horizon,
    the score drops vs the passive-world score. Sized so the swing
    is dominated by planet_delta (the cancellation between
    planet/ship/prod deltas is the easy-to-write-broken pitfall)."""
    # Us at (10, 50) — small garrison, isolated. Opp at (30, 50) with
    # a massive garrison, very close (20 units). Opp's overlay
    # launches 200 ships at us → we lose our only planet.
    # Neutral spectator at (90, 90) far from both.
    world = _toy_world(
        planets=[
            [0, 0, 10.0, 50.0, 2.0, 10, 1],   # our only planet
            [1, 1, 30.0, 50.0, 2.0, 300, 1],  # opp, heavy
            [2, -1, 90.0, 90.0, 1.0, 5, 0],   # neutral far away
        ],
        fleets=[],
    )
    our_bundle = Bundle()  # no-op for clarity
    # Opp bundle: aim 180° (toward us at (10,50)) with 200 ships —
    # but src is (30,50), so toward us is angle = pi.
    opp_bundle = Bundle(launches=(
        LaunchSpec(src_id=1, aim_angle=math.pi,
                   ships=200, owner=1, launch_turn=0),
    ))
    ev = BundleEvaluator(horizon=30)
    passive = ev.score(world, our_bundle, my_id=0)
    with_opp = ev.score(world, our_bundle, my_id=0,
                        opp_overlays={1: opp_bundle})
    # In passive, planet_delta = 1 - 1 = 0 and we hold our planet.
    # With opp overlay, planet_delta = 0 - 2 = -2 (we're wiped).
    assert with_opp.planet_delta < passive.planet_delta, (
        f"opp capture of our planet should drop planet_delta: "
        f"passive={passive}, with_opp={with_opp}"
    )
    assert with_opp.total < passive.total, (
        f"opp counterplay should reduce score: passive={passive.total}, "
        f"with_opp={with_opp.total}"
    )


def test_evaluator_infeasible_opp_overlay_dropped_silently():
    """An opp_overlay whose source no longer belongs to the opp
    (because OUR bundle captured it first) must NOT raise — the
    bundle is dropped silently and the score reflects partial
    counterplay."""
    # Opp owns p2 at (50, 85). Our bundle nukes p2 from p0 (close).
    world = _toy_world(
        planets=[
            [0, 0, 45.0, 85.0, 2.0, 50, 1],  # very close, big garrison
            [1, 0, 25.0, 75.0, 2.0, 20, 1],
            [2, 1, 50.0, 85.0, 1.5, 3, 0],   # opp planet, will fall
        ],
        fleets=[],
    )
    dx, dy = 50.0 - 45.0, 85.0 - 85.0
    our_bundle = Bundle(launches=(
        LaunchSpec(src_id=0, aim_angle=math.atan2(dy, dx),
                   ships=20, owner=0, launch_turn=0),
    ))
    # Opp tries to launch from p2 at launch_turn=5 — but by then we
    # own p2 (or at least opp doesn't), so the launch is infeasible.
    opp_bundle = Bundle(launches=(
        LaunchSpec(src_id=2, aim_angle=0.0,
                   ships=2, owner=1, launch_turn=5),
    ))
    ev = BundleEvaluator(horizon=20)
    # Should NOT raise.
    score = ev.score(world, our_bundle, my_id=0,
                     opp_overlays={1: opp_bundle})
    assert score.total > 0  # we still capture; opp's overlay silently dropped


# ---------------------------------------------------------------------------
# predict_opp_bundles_via_mirror_search
# ---------------------------------------------------------------------------


def test_mirror_search_depth_zero_returns_empty():
    """Depth 0 short-circuits before enumerating opps."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, 1, 80.0, 80.0, 2.0, 50, 1],
            [2, -1, 50.0, 90.0, 1.0, 3, 0],
        ],
        fleets=[],
    )
    out = predict_opp_bundles_via_mirror_search(world, my_id=0, depth=0)
    assert out == {}


def test_mirror_search_no_opps_returns_empty():
    """Only us + neutrals → no opps to predict for."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, 0, 25.0, 75.0, 2.0, 50, 1],
            [2, -1, 50.0, 90.0, 1.0, 3, 0],
        ],
        fleets=[],
    )
    out = predict_opp_bundles_via_mirror_search(world, my_id=0)
    assert out == {}


def test_mirror_search_depth_one_predicts_opp_capture():
    """With one opp who has a clear capture (close source, weak
    neutral target nearby), the predicted bundle is non-empty and
    its launches carry the opp's owner id."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, 1, 78.0, 78.0, 2.0, 50, 1],
            [2, -1, 70.0, 82.0, 1.0, 3, 0],   # weak neutral near opp
        ],
        fleets=[],
    )
    out = predict_opp_bundles_via_mirror_search(
        world, my_id=0,
        search=BundleSearch(max_depth=2, candidates_per_source=3),
    )
    assert 1 in out, f"expected opp 1 in prediction, got {out}"
    opp_bundle = out[1]
    assert not opp_bundle.is_empty
    for spec in opp_bundle.launches:
        assert spec.owner == 1, (
            f"opp bundle's spec has wrong owner: {spec.owner} != 1"
        )


def test_mirror_search_depth_two_does_not_infinite_recurse():
    """Depth 2 finishes (the recursion break is per-depth, not
    cycle-detection on opp_ids). Also a cost-budget sanity check
    — two opps × depth-2 should be sub-second for a small world."""
    import time
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 40, 1],
            [1, 1, 80.0, 80.0, 2.0, 40, 1],
            [2, 2, 50.0, 15.0, 2.0, 40, 1],
            [3, -1, 50.0, 50.0, 1.0, 3, 0],
        ],
        fleets=[],
    )
    search = BundleSearch(max_depth=2, candidates_per_source=2)
    t0 = time.perf_counter()
    out = predict_opp_bundles_via_mirror_search(
        world, my_id=0, search=search, depth=2,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"depth=2 mirror took {elapsed*1000:.0f} ms"
    assert isinstance(out, dict)
    for opp_id, b in out.items():
        assert opp_id != 0
        for spec in b.launches:
            assert spec.owner == opp_id


# ---------------------------------------------------------------------------
# End-to-end: BundleSearch + mirror overlay
# ---------------------------------------------------------------------------


def test_bundle_search_with_opp_overlays_beats_no_op_under_overlays():
    """The chooser's invariant: under counterplay, the chosen bundle
    still scores >= the empty bundle (the empty-bundle floor is
    preserved even when overlays make our launches look worse).

    Strict-equality is allowed: if the opp's overlay neutralises
    every action, no-op may tie no-op-with-overlay.
    """
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 40, 1],
            [1, 1, 80.0, 80.0, 2.0, 40, 1],
            [2, -1, 50.0, 90.0, 1.5, 5, 1],
            [3, -1, 50.0, 70.0, 1.5, 5, 1],
        ],
        fleets=[],
    )
    search = BundleSearch(max_depth=3, candidates_per_source=3,
                          beam_width=4)
    mirror = predict_opp_bundles_via_mirror_search(
        world, my_id=0, search=search,
    )
    chosen = search.search(world, my_id=0, opp_overlays=mirror)
    ev = search.evaluator
    chosen_score = ev.score(world, chosen, my_id=0, opp_overlays=mirror).total
    empty_score = ev.score(world, Bundle(), my_id=0, opp_overlays=mirror).total
    assert chosen_score >= empty_score, (
        f"chooser violated the empty-bundle floor under overlays: "
        f"chosen={chosen_score}, empty={empty_score}"
    )


def test_bundle_search_opp_overlays_changes_choice_vs_passive():
    """The presence of opp_overlays should be allowed to change the
    chosen bundle vs the passive-world chooser. We don't require a
    change on every world (some choices coincide), only that the
    plumbing is wired such that a noticeable opp overlay reaches the
    score function. The proxy: empty-bundle score with mirror !=
    empty-bundle score without — meaning opp launches DID land in
    the overlay world and shift our planet/ship deltas.
    """
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 40, 1],
            [1, 1, 78.0, 78.0, 2.0, 40, 1],
            [2, -1, 70.0, 82.0, 1.0, 3, 0],
        ],
        fleets=[],
    )
    search = BundleSearch(max_depth=2, candidates_per_source=3)
    mirror = predict_opp_bundles_via_mirror_search(
        world, my_id=0, search=search,
    )
    ev = search.evaluator
    passive = ev.score(world, Bundle(), my_id=0).total
    with_mirror = ev.score(world, Bundle(), my_id=0, opp_overlays=mirror).total
    # Opp captures the weak neutral, so our planet_delta at horizon is
    # one lower under the mirror.
    assert with_mirror < passive, (
        f"mirror overlay had no observable effect on empty-bundle "
        f"score: passive={passive}, with_mirror={with_mirror}"
    )
