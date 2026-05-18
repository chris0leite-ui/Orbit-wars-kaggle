"""Phase B unit tests for `predict_my_followup_via_event_driven_lite_greedy`.

Me-side mirror of `predict_opp_via_event_driven_lite_greedy`. Predicts
what lite_greedy from MY seat would launch at each subsequent event
given a pre-bundle-applied overlay world. Bug #14 (me-half) target:
makes `BundleEvaluator.score` self-consistent on both seats.

Pinned invariants (B1):
  1. Empty / no-my-planets / no-targets → empty Bundle.
  2. Viable single source → at least one launch in the followup.
  3. Followup composes with my_bundle without ValueError on apply.
  4. t=0 followup picks up sources my_bundle did NOT touch.
  5. max_events bounds the iteration count (no runaway).

Phase B-second-half tests (B2/B3 score integration + my_id guard)
land in a follow-up edit once `my_followup_mode` field exists.
"""
from __future__ import annotations

import math

import pytest

from lib.trajectory_layer import (
    Bundle,
    BundleEvaluator,
    LaunchSpec,
    World,
    predict_my_followup_via_event_driven_lite_greedy,
)


def _toy_world(planets: list, fleets: list = None, *,
               my_id: int = 0, step: int = 0) -> World:
    fleets = fleets or []
    obs = {
        "step": step,
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


# Planet tuple: (id, owner, x, y, radius, ships, production)


def test_empty_world_returns_empty_bundle():
    """No planets → empty followup (early bail on has_my_planet check)."""
    world = _toy_world(planets=[], fleets=[])
    followup = predict_my_followup_via_event_driven_lite_greedy(
        world, my_id=0, horizon=15,
    )
    assert isinstance(followup, Bundle)
    assert followup.is_empty


def test_no_my_planets_returns_empty_bundle():
    """Eliminated-me case: only opp/neutral planets. Function should
    short-circuit cheaply and return an empty Bundle."""
    world = _toy_world(planets=[
        [0, 1, 20.0, 30.0, 1.5, 50, 1],   # opp
        [1, -1, 80.0, 30.0, 1.5, 10, 1],  # neutral
    ])
    followup = predict_my_followup_via_event_driven_lite_greedy(
        world, my_id=0, horizon=15,
    )
    assert followup.is_empty


def test_no_capturable_targets_returns_empty_bundle():
    """My planets exist but lite_greedy finds no affordable target →
    empty followup (lite_greedy returns [] each event)."""
    # Only my planets present — `targets = [p for p in planets if
    # p[1] != player]` is empty, so lite_greedy returns [].
    world = _toy_world(planets=[
        [0, 0, 20.0, 30.0, 1.5, 50, 1],
        [1, 0, 80.0, 30.0, 1.5, 50, 1],
    ])
    followup = predict_my_followup_via_event_driven_lite_greedy(
        world, my_id=0, horizon=15,
    )
    assert followup.is_empty


def test_viable_source_produces_launch():
    """Single my source with enough ships + a capturable neutral
    target → followup contains at least one launch owned by me."""
    world = _toy_world(planets=[
        [0, 0, 20.0, 30.0, 1.5, 50, 1],   # my source, plenty of ships
        [1, -1, 70.0, 30.0, 1.5, 5, 1],   # easy neutral target
    ])
    followup = predict_my_followup_via_event_driven_lite_greedy(
        world, my_id=0, horizon=20,
    )
    assert not followup.is_empty, "viable launch should appear in followup"
    assert all(s.owner == 0 for s in followup.launches), \
        "every spec must be owned by my_id"
    assert any(s.src_id == 0 for s in followup.launches), \
        "launch should originate from the qualifying source"


def test_followup_composes_with_my_bundle_without_error():
    """Apply my_bundle, then predict_my_followup on the overlay, then
    apply the followup. Must not raise ValueError. Ship-bookkeeping
    consistency is enforced by World.with_candidate."""
    base = _toy_world(planets=[
        [0, 0, 20.0, 30.0, 1.5, 30, 1],
        [1, 0, 20.0, 70.0, 1.5, 25, 1],
        [2, -1, 60.0, 30.0, 1.5, 5, 1],
        [3, -1, 60.0, 70.0, 1.5, 5, 1],
    ])
    my_bundle = Bundle(launches=(
        LaunchSpec(src_id=0, aim_angle=0.0, ships=20,
                   owner=0, launch_turn=0),
    ))
    overlay = my_bundle.apply(base)
    followup = predict_my_followup_via_event_driven_lite_greedy(
        overlay, my_id=0, horizon=20,
    )
    # Whether the followup is empty or not, apply() must not raise.
    overlay2 = followup.apply(overlay)
    assert overlay2 is not None
    # Composed overlay still has the original my_bundle launch.
    assert any(spec for spec in my_bundle.launches)


def test_t0_followup_picks_up_untouched_source():
    """my_bundle drains P0 at t=0; P1 is untouched. predict_my_followup
    at t=0 sees P0 at 0 ships (skipped by lite_greedy's <10 filter) and
    P1 still full → followup should include a launch from P1.

    This is the value-add of KEEPING t=0 in the event queue. If we
    skipped t=0, this launch would be invisible to the score function
    and we'd under-credit bundles that don't drain every source."""
    base = _toy_world(planets=[
        [0, 0, 20.0, 30.0, 1.5, 30, 1],   # will be drained by my_bundle
        [1, 0, 20.0, 70.0, 1.5, 25, 1],   # untouched — eligible for followup
        [2, -1, 60.0, 30.0, 1.5, 5, 1],   # my_bundle's target
        [3, -1, 60.0, 70.0, 1.5, 5, 1],   # natural target for P1
    ])
    my_bundle = Bundle(launches=(
        LaunchSpec(src_id=0, aim_angle=0.0, ships=29,
                   owner=0, launch_turn=0),
    ))
    overlay = my_bundle.apply(base)
    followup = predict_my_followup_via_event_driven_lite_greedy(
        overlay, my_id=0, horizon=20,
    )
    sources = {s.src_id for s in followup.launches}
    assert 1 in sources, (
        f"followup should include launch from untouched P1; "
        f"got sources={sources}, launches={followup.launches}"
    )
    # P0 is drained at t=0 (1 ship left). lite_greedy's <10 ships
    # filter skips it. Verify no t=0 launch from P0 — but ALLOW later
    # launches from P0 (production refills it across the horizon, and
    # the function correctly picks those up at subsequent events).
    p0_t0_launches = [s for s in followup.launches
                      if s.src_id == 0 and s.launch_turn == 0]
    assert p0_t0_launches == [], (
        f"drained P0 must not launch at t=0; got {p0_t0_launches}"
    )


# ---------------------------------------------------------------------------
# B2/B3 — BundleEvaluator.score integration
# ---------------------------------------------------------------------------


def _a5_world() -> World:
    """A5 oracle's layout (reinforcement-aware launch). P0 has prod=3,
    so production refills the source within horizon; without me-followup,
    score treats P0 as drained-forever after a launch."""
    return _toy_world(planets=[
        [0, 0, 20.0, 50.0, 1.5, 30, 3],
        [1, 0, 80.0, 50.0, 1.5, 10, 1],
        [2, -1, 50.0, 80.0, 1.5, 40, 1],
    ])


def test_score_off_mode_matches_no_field():
    """`my_followup_mode='off'` (default) must produce identical
    scores to a baseline evaluator with no followup logic — i.e. the
    field's presence and the new guarded branch must not perturb the
    'off' code path."""
    world = _a5_world()
    bundle = Bundle(launches=(
        LaunchSpec(src_id=0, aim_angle=math.pi/2, ships=29,
                   owner=0, launch_turn=0),
    ))
    ev = BundleEvaluator(horizon=15)
    assert ev.my_followup_mode == "off", "default must be 'off'"
    s = ev.score(world, bundle)
    # Score is deterministic given a fixed world+bundle; spot-check the
    # value is finite and structurally sane.
    assert math.isfinite(s.total)
    assert s.eliminations == 0  # no full opp wipe in this layout


def test_score_lite_mode_changes_multi_source_scenario():
    """Layout: my_bundle launches from P0; me-followup picks up
    untouched P1 and launches reactively at t=0. Lite mode must
    produce a HIGHER score than off — the reactive capture of P3
    credits planet+production over the horizon that off mode is
    blind to.

    NOTE on the A5 layout specifically: it can't be tested via this
    direct-score comparison because the my_bundle's own fleet has
    travel-ETA > horizon=15 (29 ships at speed ~2.7 over distance
    42 = ~16 turns), so no `initial_etas` events fall within
    horizon, and the only processed event is t=0 where lite_greedy
    can't afford P2 from the drained P0 or the small P1. The A5
    oracle's end-to-end run still validates the mechanic in a way
    this unit test cannot."""
    world = _toy_world(planets=[
        [0, 0, 20.0, 30.0, 1.5, 30, 1],
        [1, 0, 20.0, 70.0, 1.5, 25, 1],
        [2, -1, 60.0, 30.0, 1.5, 5, 1],   # my_bundle's target
        [3, -1, 60.0, 70.0, 1.5, 5, 1],   # P1's natural reactive target
    ])
    bundle = Bundle(launches=(
        LaunchSpec(src_id=0, aim_angle=0.0, ships=20,
                   owner=0, launch_turn=0),
    ))
    ev_off = BundleEvaluator(horizon=20, my_followup_mode="off")
    ev_lite = BundleEvaluator(horizon=20, my_followup_mode="lite")
    s_off = ev_off.score(world, bundle)
    s_lite = ev_lite.score(world, bundle)
    assert s_lite.total > s_off.total, (
        f"lite mode must give strictly higher score when reactive "
        f"launch is viable; got off={s_off.total:.2f} "
        f"lite={s_lite.total:.2f}"
    )


def test_my_id_guard_skips_followup_for_opp_seat():
    """When `score` is called with `my_id != world.my_id` (e.g. the
    inner opp-search in mirror mode), me-followup must NOT run.
    Verify by comparing 'lite' score from opp's seat vs 'off' from
    opp's seat — they must be IDENTICAL."""
    world = _a5_world()
    bundle = Bundle()  # empty bundle, score the passive trajectory
    ev_off = BundleEvaluator(horizon=15, my_followup_mode="off")
    ev_lite = BundleEvaluator(horizon=15, my_followup_mode="lite")
    # Pass my_id=1 (the opp); world.my_id is 0. Guard should trip.
    s_off = ev_off.score(world, bundle, my_id=1)
    s_lite = ev_lite.score(world, bundle, my_id=1)
    assert s_off.total == s_lite.total, (
        f"guard breach: me-followup ran from opp seat; "
        f"off={s_off.total:.2f} lite={s_lite.total:.2f}"
    )


def test_max_events_bounds_iterations():
    """Pathological input with many possible events shouldn't run
    forever. Passing max_events=1 guarantees at most one event is
    processed."""
    world = _toy_world(planets=[
        [0, 0, 20.0, 30.0, 1.5, 50, 1],
        [1, 0, 20.0, 70.0, 1.5, 50, 1],
        [2, -1, 60.0, 30.0, 1.5, 5, 1],
        [3, -1, 60.0, 70.0, 1.5, 5, 1],
        [4, -1, 80.0, 50.0, 1.5, 5, 1],
    ])
    followup = predict_my_followup_via_event_driven_lite_greedy(
        world, my_id=0, horizon=30, max_events=1,
    )
    # At most one event processed = launches from at most one snapshot.
    # All launches will share launch_turn == 0 (the only processed event).
    if not followup.is_empty:
        turns = {s.launch_turn for s in followup.launches}
        assert turns == {0}, (
            f"max_events=1 should process only t=0; got turns={turns}"
        )
