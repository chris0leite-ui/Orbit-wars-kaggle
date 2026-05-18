"""Phase 8b tests for the event-driven trajectory-native reactive
opp model — `predict_opp_via_event_driven_lite_greedy`.

The function walks the trajectory layer's natural event stream
(arrival ETAs from the ledger) and at each event calls
`lite_greedy_policy` for each opp, producing a per-opp Bundle of
launches scheduled across the rollout. Equivalent in shape to
`agents/baseline`'s per-step reactive opp, but built without
stepping a simulator.

Pinned invariants:

1. Empty world (no opps) → returns empty dict.
2. With opps present, every returned Bundle is owned by its opp_id.
3. Every returned LaunchSpec has launch_turn ≥ 0 and ≤ horizon.
4. Launches emitted at turn t are sourced from planets the opp
   ACTUALLY OWNS at turn t (validated via the trajectory layer's
   ownership_at at that turn).
5. The function is callable on a snapshot result without crashing
   (chained-search composition).
6. `world_to_obs` adapter survives `lite_greedy_policy` round-trip
   (the policy can parse it and return a valid action list).
"""
from __future__ import annotations

import math

import pytest

from lib.opp_model import lite_greedy_policy
from lib.trajectory_layer import (
    Bundle,
    BundleSearch,
    LaunchSpec,
    World,
    predict_opp_via_event_driven_lite_greedy,
    world_to_obs,
)


def _toy_world(planets: list, fleets: list, *, my_id: int = 0,
               step: int = 0) -> World:
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


# ---------------------------------------------------------------------------
# world_to_obs adapter
# ---------------------------------------------------------------------------


def test_world_to_obs_lite_greedy_roundtrip():
    """`world_to_obs(world, opp_id)` produces a dict that
    `lite_greedy_policy` can consume + return a valid action list.

    Note on setup: lite_greedy picks the SINGLE-BEST ROI target
    (production/distance). To exercise its launch path we need that
    target to be AFFORDABLE for the source. A nearby positive-prod
    neutral beats a far enemy-home (which the source can't capture
    anyway)."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 0],   # us, prod=0 → ROI 0
            [1, 1, 80.0, 80.0, 2.0, 50, 1],   # enemy source
            [2, -1, 60.0, 80.0, 1.0, 5, 3],   # high-prod capturable neutral
        ],
        fleets=[],
    )
    obs = world_to_obs(world, player_id=1)
    assert obs["player"] == 1
    assert len(obs["planets"]) == 3
    # Tuple shape: (id, owner, x, y, radius, ships, production).
    for p in obs["planets"]:
        assert len(p) == 7
    # lite_greedy should produce launches from opp's owned planet.
    actions = lite_greedy_policy(obs)
    assert isinstance(actions, list)
    # opp (player 1) owns planet 1 with 50 ships → expect ≥1 launch.
    assert len(actions) >= 1
    for src_id, angle, ships in actions:
        assert int(src_id) == 1   # opp's only owned source
        assert int(ships) > 0


def test_world_to_obs_excludes_synthetic_future_fleets():
    """Synthetic fleets from `with_candidate(launch_turn>0)` carry
    `spawn_turn>0` and shouldn't appear in the t=0 obs view (they
    don't exist at the snapshot's current time)."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, -1, 80.0, 80.0, 2.0, 5, 0],
        ],
        fleets=[],
    )
    overlay = world.with_candidate(LaunchSpec(
        src_id=0, aim_angle=0.0, ships=10, owner=0, launch_turn=5,
    ))
    obs = world_to_obs(overlay, player_id=0)
    assert obs["fleets"] == [], (
        f"future-spawn fleet leaked into t=0 obs: {obs['fleets']}"
    )


# ---------------------------------------------------------------------------
# predict_opp_via_event_driven_lite_greedy
# ---------------------------------------------------------------------------


def test_event_driven_empty_world_returns_empty():
    """No opps present (only our planets + neutrals) → empty dict."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, -1, 80.0, 80.0, 2.0, 5, 0],
        ],
        fleets=[],
    )
    result = predict_opp_via_event_driven_lite_greedy(world, my_id=0)
    assert result == {}


def test_event_driven_two_seat_predicts_opp_launches():
    """Standard 2P setup: opp with a strong source and an affordable
    high-prod neutral nearby. Event-driven prediction must emit ≥1
    launch owned by opp from opp's source."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 0],   # us, prod=0
            [1, 1, 80.0, 80.0, 2.0, 50, 1],   # opp source
            [2, -1, 60.0, 80.0, 1.0, 5, 3],   # high-prod neutral
        ],
        fleets=[],
    )
    result = predict_opp_via_event_driven_lite_greedy(
        world, my_id=0, horizon=30,
    )
    assert 1 in result, f"opp_id=1 missing from result: {result}"
    bundle = result[1]
    assert len(bundle.launches) >= 1
    for spec in bundle.launches:
        assert spec.owner == 1
        assert 0 <= spec.launch_turn <= 30
        # Source should be opp-owned at launch_turn.
        owner_at, _ = world.ownership_at(spec.src_id, spec.launch_turn)
        assert owner_at == 1, (
            f"launch from non-opp source: spec={spec} "
            f"owner_at_launch={owner_at}"
        )


def test_event_driven_horizon_truncates_late_events():
    """Launches with launch_turn > horizon should not appear in
    any returned bundle. (max_events also caps event count.)"""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, 1, 80.0, 80.0, 2.0, 50, 1],
            [2, -1, 60.0, 80.0, 1.0, 5, 0],
        ],
        fleets=[],
    )
    horizon = 10
    result = predict_opp_via_event_driven_lite_greedy(
        world, my_id=0, horizon=horizon,
    )
    for opp_id, bundle in result.items():
        for spec in bundle.launches:
            assert spec.launch_turn <= horizon, (
                f"launch beyond horizon: spec.launch_turn="
                f"{spec.launch_turn} > horizon={horizon}"
            )


def test_event_driven_no_opps_with_planets_returns_empty():
    """Opp player exists but has no owned planets (eliminated) →
    no launches predicted."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, -1, 80.0, 80.0, 2.0, 5, 0],
        ],
        fleets=[],
    )
    result = predict_opp_via_event_driven_lite_greedy(world, my_id=0)
    assert result == {}


def test_event_driven_reactive_to_inflight_arrivals():
    """An in-flight enemy fleet about to capture a neutral planet
    creates an event in the ledger. After the capture, opp now owns
    a NEW planet → opp's lite_greedy should consider launching from
    the newly-captured planet later in the rollout.

    Pin: with an arrival mid-horizon, the event-driven prediction
    emits ≥1 launch with launch_turn > 0 (i.e. evolves over the
    rollout, not just t=0)."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 30, 0],   # us, prod=0 → opp ignores
            [1, 1, 80.0, 80.0, 2.0, 30, 1],   # opp home
            [2, -1, 55.0, 80.0, 1.0, 3, 3],   # capturable high-prod neutral
            [3, -1, 40.0, 80.0, 1.0, 3, 3],   # 2nd capturable neutral (post-arrival opp target)
        ],
        # opp fleet about to capture neutral id=2: from (78, 80)
        # heading left, ~25-ship fleet.
        fleets=[[0, 1, 75.0, 80.0, math.pi, 1, 25]],
    )
    result = predict_opp_via_event_driven_lite_greedy(
        world, my_id=0, horizon=40,
    )
    assert 1 in result
    bundle = result[1]
    # At least SOME launches should be at t > 0 (post-arrival
    # opportunism). Otherwise event-driven adds no signal over a
    # one-shot lite_greedy at t=0.
    delayed = [s for s in bundle.launches if s.launch_turn > 0]
    assert delayed, (
        f"event-driven prediction yielded no t>0 launches: "
        f"{bundle.launches}"
    )


def test_event_driven_callable_after_snapshot_at():
    """The function should compose with `snapshot_at` — caller
    should be able to roll a world forward and predict opp from
    that rolled-forward state."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, 1, 80.0, 80.0, 2.0, 50, 1],
        ],
        fleets=[],
    )
    snap = world.snapshot_at(5)
    result = predict_opp_via_event_driven_lite_greedy(
        snap, my_id=0, horizon=20,
    )
    # Should run without exception.
    assert isinstance(result, dict)
    # Opp still owns planet 1 at snap.step=5 → expect a bundle.
    if 1 in result:
        for spec in result[1].launches:
            assert spec.owner == 1


def test_event_driven_integrates_with_bundle_search():
    """End-to-end: pass event-driven opp_overlays into BundleSearch.
    The search must run without exception and produce a Bundle."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, 1, 80.0, 80.0, 2.0, 50, 1],
            [2, -1, 50.0, 85.0, 1.0, 5, 0],
        ],
        fleets=[],
    )
    overlays = predict_opp_via_event_driven_lite_greedy(
        world, my_id=0, horizon=20,
    )
    search = BundleSearch(max_depth=1, beam_width=2,
                          candidates_per_source=3)
    bundle = search.search(world, my_id=0, opp_overlays=overlays)
    assert isinstance(bundle, Bundle)


def test_event_driven_opp_emits_lead_angles_under_omega():
    """Phase C+ symmetric world model: when world has non-zero omega,
    predict_opp's emitted specs use lead-aim (via lite_greedy_policy's
    omega path) rather than static atan2 to current target position.

    Without this, bundle's enumeration uses lead-aim but the opp
    prediction misses orbital targets — asymmetric world model that
    biases bundle's score to over-aggressive play."""
    import math
    from dataclasses import replace
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 30.0, 2.0, 5, 0],
            [1, 1, 80.0, 30.0, 2.0, 50, 1],
            [2, -1, 20.0, 70.0, 1.5, 5, 2],
        ],
        fleets=[],
    )
    # Override omega: _toy_world defaults angular_velocity to 0.
    world = replace(world, omega=0.04)
    result = predict_opp_via_event_driven_lite_greedy(
        world, my_id=0, horizon=20,
    )
    assert 1 in result, "opp should emit at least one launch"
    bundle = result[1]
    assert not bundle.is_empty

    static_angle = math.atan2(70.0 - 30.0, 20.0 - 80.0)
    matched = False
    for spec in bundle.launches:
        if spec.src_id != 1:
            continue
        delta = abs(spec.aim_angle - static_angle)
        delta = min(delta, abs(delta - 2 * math.pi),
                    abs(delta + 2 * math.pi))
        if delta > 0.02:
            matched = True
            break
    assert matched, (
        f"no spec from src=1 differs from static angle {static_angle:.4f} "
        f"by > 0.02 rad — lead-aim not active. bundle={bundle.launches}"
    )
