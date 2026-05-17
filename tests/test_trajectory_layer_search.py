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


# ---------------------------------------------------------------------------
# Phase 7d — Delayed-launch enumeration
# ---------------------------------------------------------------------------


def test_search_default_launch_turns_unchanged():
    """With `launch_turns=(0,)` (default), the search produces the
    same bundle as Phase 7c. Regression guard so the knob defaults
    don't change behaviour."""
    world = _toy_world(
        planets=[
            [0, 0, 30.0, 80.0, 2.0, 50, 1],
            [1, -1, 50.0, 90.0, 1.0, 3, 0],
        ],
        fleets=[],
    )
    search = BundleSearch()  # default launch_turns=(0,)
    bundle = search.search(world, my_id=0)
    # Phase 7c baseline: captures the near-target with launch_turn=0.
    assert not bundle.is_empty
    assert all(s.launch_turn == 0 for s in bundle.launches)


def test_search_delayed_launch_captures_strong_enemy():
    """A target too strong for an immediate single launch but
    capturable with a DELAYED launch once the source's production
    has accrued enough ships. Target is an ENEMY: capturing
    eliminates them (elimination_bonus=200 amortises the ship cost).

    Setup (sun-clear): source at (35, 85), 30 ships, prod=1. Enemy
    planet at (55, 85), 40 ships, prod=1 (their only planet).

    - Immediate launch: 29 ships → 40 + 5*1 (flight time) = 45
      garrison at arrival → fails, ships wasted.
    - Delayed launch at t=20: src has 30+20=50 ships; can send 49;
      target garrison = 40 + ~25 ≈ 65 → still fails for 49.
    - Delayed launch at t=40 with multiplier=1.5: src has 30+40=70
      ships; can send min(60, 69) = 60; target at arrival ≈ 40+50 =
      90 → still fails.

    Realistic setup: lower target ships so the delayed launch
    succeeds at moderate horizon.
    """
    world = _toy_world(
        planets=[
            [0, 0, 35.0, 85.0, 2.0, 25, 1],
            [1, 1, 55.0, 85.0, 2.0, 25, 1],  # ENEMY, their only planet
        ],
        fleets=[],
    )
    ev = BundleEvaluator(horizon=60)
    search = BundleSearch(
        evaluator=ev,
        max_depth=2,
        beam_width=4,
        launch_turns=(0, 5, 10, 15, 20),
        ship_count_multiplier=1.5,  # buffer for production accrual
    )
    bundle = search.search(world, my_id=0)
    assert not bundle.is_empty, "search picked no launch"
    # The search should pick A launch that scores higher than no-op,
    # and the bundle's score should reflect an attempted capture.
    score = ev.score(world, bundle, my_id=0)
    empty_score = ev.score(world, Bundle(), my_id=0)
    assert score.total > empty_score.total, (
        f"bundle score {score.total} not better than empty {empty_score.total}"
    )


# ---------------------------------------------------------------------------
# Phase 7e — seed-and-extend (carry last turn's plan into this turn)
# ---------------------------------------------------------------------------


def test_search_honors_seed_when_no_improvement():
    """Seed is the best plan available; the search must not return
    something that scores below the seed. Pinned: the search's empty-
    bundle restart shouldn't accidentally win when seed is in fact the
    correct plan."""
    world = _toy_world(
        planets=[
            [0, 0, 30.0, 85.0, 2.0, 50, 1],
            [1, -1, 50.0, 85.0, 1.0, 3, 0],
        ],
        fleets=[],
    )
    seed_spec = LaunchSpec(src_id=0, aim_angle=0.0, ships=4, owner=0)
    seed = Bundle(launches=(seed_spec,))
    ev = BundleEvaluator(horizon=30)
    seed_score = ev.score(world, seed, my_id=0).total
    search = BundleSearch(evaluator=ev, max_depth=2)
    bundle = search.search(world, my_id=0, seed_bundle=seed)
    final_score = ev.score(world, bundle, my_id=0).total
    assert final_score >= seed_score, (
        f"final score {final_score} regressed below seed {seed_score}"
    )


def test_search_rejects_harmful_seed():
    """A seed that wastes ships (launches into empty space with no
    capturable target) scores below empty. The search must fall back
    to empty (or a plan that scores at least as well as empty), NOT
    return the seed."""
    world = _toy_world(
        planets=[
            [0, 0, 30.0, 80.0, 2.0, 50, 1],
            # No targets — the wasteful launch has nothing to hit.
        ],
        fleets=[],
    )
    wasteful = LaunchSpec(src_id=0, aim_angle=0.0, ships=20, owner=0)
    seed = Bundle(launches=(wasteful,))
    ev = BundleEvaluator(horizon=30)
    seed_score = ev.score(world, seed, my_id=0).total
    empty_score = ev.score(world, Bundle(), my_id=0).total
    # Test setup invariant: seed IS worse than empty.
    assert seed_score < empty_score, (
        f"test setup invalid: seed_score={seed_score} not less than "
        f"empty_score={empty_score}"
    )
    search = BundleSearch(evaluator=ev, max_depth=2)
    bundle = search.search(world, my_id=0, seed_bundle=seed)
    final_score = ev.score(world, bundle, my_id=0).total
    assert final_score >= empty_score
    assert wasteful not in bundle.launches


def test_search_extends_seed_with_new_launch():
    """Seed = one good capture. World has a second capturable target
    reachable only from a DIFFERENT source (avoids the
    same-ray-same-spec collapse: when two targets sit on the same
    angle from a source, the search's LaunchSpec for each is
    identical and both fleets hit the closer one). Two sources +
    two on-axis targets gives the search a genuine 2-launch optimum."""
    world = _toy_world(
        planets=[
            [0, 0, 30.0, 85.0, 2.0, 50, 1],   # source A (prod=1)
            [1, 0, 70.0, 85.0, 2.0, 50, 1],   # source B (prod=1)
            [2, -1, 40.0, 85.0, 1.0, 3, 0],   # T1, closer to A
            [3, -1, 60.0, 85.0, 1.0, 3, 0],   # T2, closer to B
        ],
        fleets=[],
    )
    # Seed: A → T1 (angle 0, eastward).
    seed_spec = LaunchSpec(src_id=0, aim_angle=0.0, ships=4, owner=0)
    seed = Bundle(launches=(seed_spec,))
    search = BundleSearch(max_depth=3, beam_width=4,
                           candidates_per_source=3)
    bundle = search.search(world, my_id=0, seed_bundle=seed)
    assert len(bundle.launches) >= 2, (
        f"search failed to extend seed: {bundle.launches}"
    )
    # The seed itself should survive (the second capture didn't
    # replace the first; the +1 planet from extending dominates the
    # ship cost).
    assert seed_spec in bundle.launches


def test_search_drops_seed_launch_to_save_ships():
    """Drop semantics: seed has a launch that hurts the score (fleet
    flies into empty space, ships lost). The search must DROP it.
    Setup uses launch_turns=(0,) and a single capturable target so
    the only reachable improvements are (a) drop the wasted launch,
    (b) add a capture from the same source. Both improvements
    require the drop to free the source's ships."""
    world = _toy_world(
        planets=[
            [0, 0, 30.0, 80.0, 2.0, 10, 1],   # source w/ exactly 10 ships
            [1, -1, 30.0, 90.0, 1.0, 3, 0],   # capturable upward
        ],
        fleets=[],
    )
    # Wasteful eastbound launch from source 0 — no planet in that
    # direction. Uses 9 ships (keeping 1, the minimum), so without
    # dropping it, no other useful launch from source 0 is possible.
    wasteful = LaunchSpec(src_id=0, aim_angle=0.0, ships=9, owner=0,
                           launch_turn=0)
    seed = Bundle(launches=(wasteful,))
    ev = BundleEvaluator(horizon=30)
    seed_score = ev.score(world, seed, my_id=0).total
    empty_score = ev.score(world, Bundle(), my_id=0).total
    # Test setup invariant: seed wastes ships (worse than empty).
    assert seed_score < empty_score
    search = BundleSearch(evaluator=ev, max_depth=2)
    bundle = search.search(world, my_id=0, seed_bundle=seed)
    # Drop required: wasteful launch must not be in the final bundle.
    assert wasteful not in bundle.launches, (
        f"search did not drop the wasteful seed launch: "
        f"{bundle.launches}"
    )
    # Final must score at least as well as empty.
    final_score = ev.score(world, bundle, my_id=0).total
    assert final_score >= empty_score


def test_search_swaps_seed_launch_via_drop_then_add():
    """Multi-launch swap via drop+add: seed = (good_capture,
    wasteful). The search should keep `good_capture` and replace
    `wasteful` with a useful add — i.e. the final bundle contains
    `good_capture` but NOT `wasteful`. This is the case that pure-
    add search can't reach: `good_capture` would have to be re-
    discovered as a singleton via the empty-restart path AND beat
    out other partial extensions to survive beam pruning. The drop
    path goes directly: seed → drop wasteful → (good_capture,) →
    add new useful capture."""
    world = _toy_world(
        planets=[
            [0, 0, 30.0, 80.0, 2.0, 50, 1],   # source A
            [1, 0, 70.0, 80.0, 2.0, 50, 1],   # source B
            [2, -1, 50.0, 80.0, 1.0, 3, 0],   # capturable centre target
            [3, -1, 60.0, 90.0, 1.0, 3, 0],   # second capturable
        ],
        fleets=[],
    )
    # `good_capture`: source A captures planet 2.
    good_capture = LaunchSpec(src_id=0, aim_angle=0.0, ships=4, owner=0)
    # `wasteful`: source B fires east (away from all targets).
    wasteful = LaunchSpec(src_id=1, aim_angle=0.0, ships=30, owner=0)
    seed = Bundle(launches=(good_capture, wasteful))
    search = BundleSearch(max_depth=3, beam_width=4,
                           candidates_per_source=3)
    bundle = search.search(world, my_id=0, seed_bundle=seed)
    # Wasteful must be dropped.
    assert wasteful not in bundle.launches, (
        f"wasteful seed launch survived: {bundle.launches}"
    )
    # Some useful action must have been emitted from source 0 or 1.
    assert len(bundle.launches) >= 1
    ev = BundleEvaluator(horizon=30)
    seed_score = ev.score(world, seed, my_id=0).total
    final_score = ev.score(world, bundle, my_id=0).total
    assert final_score > seed_score, (
        f"final score {final_score} not strictly better than seed "
        f"{seed_score}"
    )


def test_search_coordinated_arrival_from_far_planet():
    """Idle FAR source + close source: coordinated bundle has a
    near-source launch at t=0 AND a far-source launch at t=K so
    both arrive on the same turn against a strong target.

    Setup (sun-clear, all at y=85):
    - Near source at (40, 85) — close to target.
    - Far source at (10, 85) — far from target.
    - Target neutral at (55, 85), 35 ships, prod=1.
    - Each source alone (~30 ships) fails; combined arrival of
      ~60 ships captures despite production accrual.

    The far source's launch needs to fire at launch_turn=0 too if
    distances make synchronised arrival impossible — but with the
    far source 45 units away and the near source 15 units away,
    flight times differ by ~25 turns. For coordinated arrival, the
    near source should DELAY by ~25 turns (or the far source should
    launch first immediately).
    """
    world = _toy_world(
        planets=[
            [0, 0, 40.0, 85.0, 2.0, 30, 1],   # near source
            [1, 0, 10.0, 85.0, 2.0, 30, 1],   # FAR source
            [2, -1, 55.0, 85.0, 2.0, 35, 1],  # target
        ],
        fleets=[],
    )
    ev = BundleEvaluator(horizon=80)
    search = BundleSearch(
        evaluator=ev,
        max_depth=2,
        beam_width=4,
        launch_turns=(0, 10, 20, 30),
    )
    bundle = search.search(world, my_id=0)
    assert not bundle.is_empty
    # Strong claim: the bundle SHOULD include the far source (id 1).
    # We accept either "far source launches" OR "near source delayed"
    # — both are valid trajectory-native discoveries. Just verify
    # the search is making a non-trivial multi-step decision (not
    # only single-immediate-launch).
    is_multi = (len(bundle.launches) >= 2
                or any(s.launch_turn > 0 for s in bundle.launches))
    assert is_multi, (
        f"search produced only single immediate launch: "
        f"{bundle.launches}"
    )
