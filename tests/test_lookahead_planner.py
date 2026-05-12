"""Unit tests for lib/lookahead_planner — value fn, adaptive K, comet truncation."""

from __future__ import annotations

import pytest

from lib.lookahead_planner import (
    COMET_SPAWN_STEPS,
    K_MAX,
    K_MIN,
    adaptive_K,
    evaluate_value,
    truncate_K_to_comet_boundary,
)


# ---------------------------------------------------------------------------
# evaluate_value
# Planet row schema: [id, owner, x, y, radius, ships, production]
# Fleet row schema:  [id, owner, x, y, angle, from_planet_id, ships]
# ---------------------------------------------------------------------------


def _obs(planets, fleets=None):
    return {"planets": planets, "fleets": fleets or []}


def test_evaluate_value_empty_world_is_zero():
    assert evaluate_value(_obs([]), my_id=0) == 0.0


def test_evaluate_value_we_own_everything_maxes_share_terms():
    """All planets ours, no opp, no neutrals. Survivor bonus dominates."""
    planets = [
        [0, 0, 50.0, 50.0, 3.0, 10, 2.0],
        [1, 0, 60.0, 60.0, 3.0, 5, 3.0],
    ]
    v = evaluate_value(_obs(planets), my_id=0)
    # prod_share=1, denial=1, ships_share=1, survivor=1
    # 1.0 + 0.4 + 0.05 + 5.0 = 6.45
    assert v == pytest.approx(1.0 + 0.4 + 0.05 + 5.0)


def test_evaluate_value_opp_owns_everything_is_lowest():
    """All planets opp, no us. Denial=0, share=0, survivor=0."""
    planets = [
        [0, 1, 50.0, 50.0, 3.0, 10, 2.0],
        [1, 1, 60.0, 60.0, 3.0, 5, 3.0],
    ]
    v = evaluate_value(_obs(planets), my_id=0)
    assert v == pytest.approx(0.0)


def test_evaluate_value_monotone_in_production_share_when_ships_equal():
    """Hold ships and opp constant; increasing our production share lifts V."""
    # Baseline: we own 1/3 of total production, opp owns 1/3, 1/3 neutral.
    base = [
        [0, 0, 50.0, 50.0, 3.0, 10, 1.0],  # us, prod 1
        [1, 1, 60.0, 60.0, 3.0, 10, 1.0],  # opp, prod 1
        [2, -1, 70.0, 70.0, 3.0, 10, 1.0],  # neutral, prod 1
    ]
    # After: we own 2/3 of production by flipping the neutral. Ships at
    # each planet kept identical so ships_share is unchanged.
    after = [
        [0, 0, 50.0, 50.0, 3.0, 10, 1.0],
        [1, 1, 60.0, 60.0, 3.0, 10, 1.0],
        [2, 0, 70.0, 70.0, 3.0, 10, 1.0],  # neutral → us
    ]
    v_base = evaluate_value(_obs(base), my_id=0)
    v_after = evaluate_value(_obs(after), my_id=0)
    assert v_after > v_base


def test_evaluate_value_denial_rewards_blocking_opp_without_capture():
    """Comparing two states where prod_share is identical but opp owns
    different amounts. Lower opp_share → higher denial term → higher V."""
    # State A: we own 1, opp owns 2, no neutrals. prod_share=1/3, denial=1/3
    state_a = [
        [0, 0, 50.0, 50.0, 3.0, 5, 1.0],
        [1, 1, 60.0, 60.0, 3.0, 5, 1.0],
        [2, 1, 70.0, 70.0, 3.0, 5, 1.0],
    ]
    # State B: we own 1, opp owns 1, 1 neutral. prod_share=1/3, denial=2/3
    state_b = [
        [0, 0, 50.0, 50.0, 3.0, 5, 1.0],
        [1, 1, 60.0, 60.0, 3.0, 5, 1.0],
        [2, -1, 70.0, 70.0, 3.0, 5, 1.0],
    ]
    v_a = evaluate_value(_obs(state_a), my_id=0)
    v_b = evaluate_value(_obs(state_b), my_id=0)
    # Both have prod_share=1/3 (≈0.333) but denial differs (1/3 vs 2/3),
    # and ships_share also differs (1/3 vs 1/2 since one fewer opp pile).
    # We assert V_b > V_a — denial + ships_share both push up.
    assert v_b > v_a


def test_evaluate_value_fleets_contribute_to_ships_share():
    """In-flight fleets are owned ships and should count for ships_share."""
    planets = [
        [0, 0, 50.0, 50.0, 3.0, 1, 1.0],
        [1, 1, 60.0, 60.0, 3.0, 1, 1.0],
    ]
    # No fleets baseline; ours: 50% ships share
    v_no_fleet = evaluate_value(_obs(planets), my_id=0)
    # With a big fleet of ours in flight
    fleet = [[0, 0, 55.0, 55.0, 0.0, 0, 100]]
    v_with_fleet = evaluate_value(_obs(planets, fleet), my_id=0)
    assert v_with_fleet > v_no_fleet


def test_evaluate_value_survivor_bonus_only_when_alone():
    """Lone survivor: opp has no planets owned (even if neutral exists)."""
    # We own one planet, the other is neutral. We're the sole owner.
    planets_alone = [
        [0, 0, 50.0, 50.0, 3.0, 5, 1.0],
        [1, -1, 60.0, 60.0, 3.0, 5, 1.0],
    ]
    v = evaluate_value(_obs(planets_alone), my_id=0)
    # prod_share=1/2, denial=1, ships_share=1, survivor=1
    # = 0.5 + 0.4 + 0.05 + 5.0 = 5.95
    assert v == pytest.approx(0.5 + 0.4 + 0.05 + 5.0)


def test_evaluate_value_zero_denial_weight_drops_term():
    planets = [
        [0, 0, 50.0, 50.0, 3.0, 5, 1.0],
        [1, 1, 60.0, 60.0, 3.0, 5, 1.0],
    ]
    v_default = evaluate_value(_obs(planets), my_id=0)
    v_no_denial = evaluate_value(_obs(planets), my_id=0, denial_weight=0.0)
    assert v_no_denial < v_default


# ---------------------------------------------------------------------------
# adaptive_K
# ---------------------------------------------------------------------------


class _StubPlanet:
    def __init__(self, owner):
        self.owner = owner


class _StubWorld:
    """Minimal World-like stub for adaptive_K — only the fields it reads."""

    def __init__(self, fleets=None, planet_owners=()):
        self.obs_raw = {"fleets": fleets or []}
        self.planets_by_id = {i: _StubPlanet(o) for i, o in enumerate(planet_owners)}


def test_adaptive_K_empty_board_floors_at_K_MIN():
    world = _StubWorld(fleets=[], planet_owners=[])
    assert adaptive_K(world) == K_MIN


def test_adaptive_K_quiet_board_returns_floor():
    # No fleets, only owned planets (no neutrals).
    world = _StubWorld(fleets=[], planet_owners=[0, 1])
    assert adaptive_K(world) == K_MIN


def test_adaptive_K_grows_with_fleets():
    # 4 fleets, no neutrals → entropy=4, K = round(K_MIN + 0.5*4) = K_MIN + 2.
    fleets = [[i, 0, 0, 0, 0, 0, 1] for i in range(4)]
    world = _StubWorld(fleets=fleets, planet_owners=[0, 1])
    assert adaptive_K(world) == K_MIN + 2


def test_adaptive_K_grows_with_contested():
    # 0 fleets, 4 neutrals → entropy=2, K = round(K_MIN + 0.5*2) = K_MIN + 1.
    world = _StubWorld(fleets=[], planet_owners=[-1, -1, -1, -1, 0, 1])
    assert adaptive_K(world) == K_MIN + 1


def test_adaptive_K_caps_at_K_MAX():
    # Lots of fleets → entropy huge; cap at K_MAX.
    fleets = [[i, 0, 0, 0, 0, 0, 1] for i in range(30)]
    world = _StubWorld(fleets=fleets, planet_owners=[-1] * 10)
    assert adaptive_K(world) == K_MAX


# ---------------------------------------------------------------------------
# truncate_K_to_comet_boundary
# ---------------------------------------------------------------------------


def test_truncate_K_at_step_47_with_K20_yields_2():
    """Gate-specified case: step 47, K=20 → K=2 (don't cross step 50)."""
    assert truncate_K_to_comet_boundary(20, 47) == 2


def test_truncate_K_at_step_60_with_K30_unchanged():
    """Step 60 is mid-segment (50,150); K=30 doesn't reach 150."""
    assert truncate_K_to_comet_boundary(30, 60) == 30


def test_truncate_K_at_boundary_minus_one_floors_at_1():
    """One step before boundary: allowed = 0, but floor at 1."""
    assert truncate_K_to_comet_boundary(20, 49) == 1


def test_truncate_K_past_all_boundaries_unchanged():
    """After step 450 there are no more boundaries; K returned as-is."""
    assert truncate_K_to_comet_boundary(30, 460) == 30


def test_truncate_K_at_step_just_after_boundary_uses_next_one():
    """Step 50 is itself a boundary; next strictly-greater is 150."""
    # 150 - 50 - 1 = 99, K=30 unchanged.
    assert truncate_K_to_comet_boundary(30, 50) == 30


@pytest.mark.parametrize("boundary", COMET_SPAWN_STEPS)
def test_truncate_K_never_crosses_any_boundary(boundary):
    """For step = boundary - 5 and K big, result must keep step + K < boundary."""
    step = boundary - 5
    K_in = 100
    K_out = truncate_K_to_comet_boundary(K_in, step)
    assert step + K_out < boundary
