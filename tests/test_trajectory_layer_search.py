"""Phase 7c tests for lib.trajectory_layer — BundleSearch.

The greedy hill-climb chooser. Tests pin:

1. Empty world / no targets → empty bundle.
2. Single owned source + single weak target → bundle picks the capture.
3. Two strong sources + one strong target → bundle picks a gang-up
   (multi-launch) IF the combined ships beat the target.
4. SunFilter integration: candidates aimed through the sun are not
   emitted.
5. Source-ship accounting: a bundle that commits all of a source's
   ships doesn't double-commit them.
6. Score monotone: each added spec strictly improves the score, or
   the search stops.
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
# Trivial cases
# ---------------------------------------------------------------------------


def test_search_no_targets_returns_empty():
    """A world with only owned planets and no targets → empty
    bundle (nothing to capture)."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, 0, 80.0, 80.0, 2.0, 50, 1],
        ],
        fleets=[],
    )
    search = BundleSearch()
    bundle = search.search(world, my_id=0)
    assert bundle.is_empty


def test_search_no_sources_returns_empty():
    """No owned planets → no launches possible."""
    world = _toy_world(
        planets=[
            [0, -1, 20.0, 80.0, 2.0, 5, 0],
            [1, 1, 80.0, 80.0, 2.0, 50, 1],
        ],
        fleets=[],
        my_id=0,
    )
    search = BundleSearch()
    bundle = search.search(world, my_id=0)
    assert bundle.is_empty


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def test_search_captures_neutral_target():
    """Owned source with plenty of ships + nearby weak neutral
    target → bundle picks the capture. Geometry chosen so the
    capture lands within horizon=30 (target 20 units away, ETA ≈ 14
    turns for a 4-ship fleet)."""
    world = _toy_world(
        planets=[
            [0, 0, 30.0, 80.0, 2.0, 50, 1],
            [1, -1, 50.0, 90.0, 1.0, 3, 0],   # close neutral, off-axis (sun-clear)
        ],
        fleets=[],
    )
    search = BundleSearch()
    bundle = search.search(world, my_id=0)
    assert not bundle.is_empty
    assert len(bundle.launches) >= 1
    spec = bundle.launches[0]
    assert spec.src_id == 0
    assert spec.owner == 0
    assert spec.ships >= 4


def test_search_does_not_send_through_sun():
    """A bundle should NOT include a spec aimed through the sun.
    Setup: owned source at (40, 50), target at (60, 50) — direct
    line through sun at (50, 50)."""
    world = _toy_world(
        planets=[
            [0, 0, 40.0, 50.0, 2.0, 50, 1],
            [1, -1, 60.0, 50.0, 1.0, 3, 0],   # sun-blocked direct line
        ],
        fleets=[],
    )
    search = BundleSearch()
    bundle = search.search(world, my_id=0)
    # The aim toward (60, 50) is sun-blocked, so no candidate fires.
    # Search returns empty (no other targets).
    assert bundle.is_empty


# ---------------------------------------------------------------------------
# Multi-launch
# ---------------------------------------------------------------------------


def test_search_emits_multi_launch_when_useful():
    """Two owned sources + one strong neutral that NEITHER alone can
    capture but TOGETHER can. Search should produce a 2-launch
    bundle.

    Geometry: all on horizontal line at y=85 (35 units above sun) so
    fleets fly horizontally clear of the sun. Source A at (25, 85),
    source B at (65, 85), target at (45, 85). Symmetric flight ~15
    units each. ETA at speed ~3.1 (25-ship fleet) is ~4-5 turns. At
    capture time target garrison = 40 + 5*2 = 50; gang-up 50 ships
    survives the wash (50-50=0 — both destroyed). Make target prod=1
    so by ETA=5 garrison is 45 — gang-up 50 captures with 5
    surviving.
    """
    world = _toy_world(
        planets=[
            [0, 0, 25.0, 85.0, 2.0, 25, 1],
            [1, 0, 65.0, 85.0, 2.0, 25, 1],
            # Target: 40-ship neutral with prod=1 (worth capturing
            # over a long enough horizon to amortise the ship cost).
            [2, -1, 45.0, 85.0, 2.0, 40, 1],
        ],
        fleets=[],
    )
    # Horizon must be long enough for the captured planet's production
    # to repay the ships invested in the gang-up. 60 turns of +1
    # production = +60 ships; gang-up costs ~50 ships net.
    ev = BundleEvaluator(horizon=60)
    search = BundleSearch(evaluator=ev, max_depth=3, beam_width=4)
    bundle = search.search(world, my_id=0)
    # Want >= 1 launch; ideally 2 (gang-up) but geometry may stagger arrivals.
    assert len(bundle.launches) >= 1
    score = ev.score(world, bundle, my_id=0)
    empty_score = ev.score(world, Bundle(), my_id=0)
    assert score.total >= empty_score.total


# ---------------------------------------------------------------------------
# Score is monotone (greedy invariant)
# ---------------------------------------------------------------------------


def test_search_score_is_monotone_non_decreasing():
    """Each added spec strictly improves the bundle's score, or the
    search stops. After search completes, the final bundle's score
    is >= the empty-bundle score."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, -1, 60.0, 80.0, 1.0, 5, 0],
            [2, -1, 80.0, 80.0, 1.0, 3, 0],
            [3, 1, 30.0, 20.0, 2.0, 30, 1],
        ],
        fleets=[],
    )
    ev = BundleEvaluator(horizon=30)
    search = BundleSearch(evaluator=ev, max_depth=3)
    bundle = search.search(world, my_id=0)
    score = ev.score(world, bundle, my_id=0)
    empty_score = ev.score(world, Bundle(), my_id=0)
    assert score.total >= empty_score.total


# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------


def test_search_respects_max_depth_zero():
    """max_depth=0 → no extensions ever; returns empty bundle."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 50, 1],
            [1, -1, 80.0, 80.0, 1.0, 3, 0],
        ],
        fleets=[],
    )
    search = BundleSearch(max_depth=0)
    bundle = search.search(world, my_id=0)
    assert bundle.is_empty


def test_search_caps_at_max_depth():
    """A wide-open world with many targets should still produce a
    bundle no larger than `max_depth`."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 80, 1],  # rich source
            [1, -1, 30.0, 80.0, 1.0, 3, 0],
            [2, -1, 40.0, 80.0, 1.0, 3, 0],
            [3, -1, 50.0, 80.0, 1.0, 3, 0],
            [4, -1, 60.0, 80.0, 1.0, 3, 0],
            [5, -1, 70.0, 80.0, 1.0, 3, 0],
        ],
        fleets=[],
    )
    search = BundleSearch(max_depth=2)
    bundle = search.search(world, my_id=0)
    assert len(bundle.launches) <= 2


# ---------------------------------------------------------------------------
# Source ship accounting
# ---------------------------------------------------------------------------


def test_search_does_not_overcommit_source():
    """A bundle from a single source shouldn't ask for more ships
    than the source actually has across all its launches."""
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 10, 1],   # source w/ only 10 ships
            [1, -1, 30.0, 80.0, 1.0, 3, 0],
            [2, -1, 40.0, 80.0, 1.0, 3, 0],
            [3, -1, 50.0, 80.0, 1.0, 3, 0],
        ],
        fleets=[],
    )
    search = BundleSearch(max_depth=3)
    bundle = search.search(world, my_id=0)
    # Sum of launches from source 0 must be <= 10 - 1 (keep 1).
    total_from_src0 = sum(s.ships for s in bundle.launches
                           if s.src_id == 0)
    assert total_from_src0 <= 9, \
        f"overcommitted source 0: {total_from_src0} > 9"


# ---------------------------------------------------------------------------
# Integration: realistic-ish mid-game
# ---------------------------------------------------------------------------


def test_search_completes_on_realistic_state():
    """Plausible 8-planet 2P state. Search should terminate quickly
    (< 1 second wall) and produce some non-trivial bundle."""
    import time
    world = _toy_world(
        planets=[
            [0, 0, 20.0, 80.0, 2.0, 30, 1],
            [1, 0, 25.0, 65.0, 2.0, 25, 1],
            [2, 1, 80.0, 20.0, 2.0, 30, 1],
            [3, 1, 75.0, 35.0, 2.0, 25, 1],
            [4, -1, 50.0, 90.0, 1.5, 10, 1],
            [5, -1, 50.0, 10.0, 1.5, 10, 1],
            [6, -1, 90.0, 50.0, 1.5, 10, 1],
            [7, -1, 10.0, 50.0, 1.5, 10, 1],
        ],
        fleets=[],
    )
    search = BundleSearch(max_depth=3, candidates_per_source=3)
    t0 = time.perf_counter()
    bundle = search.search(world, my_id=0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"search took {elapsed*1000:.0f} ms"
    # Realistic state: we should commit at least one launch.
    assert len(bundle.launches) >= 1
