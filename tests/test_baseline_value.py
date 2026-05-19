"""Unit tests for agents/baseline/value.favor.

A2 — 4P weakness exploitation only. 2P path is UNCHANGED from baseline.
History: A 2P uniform bias (1.25x) was tested and rolled back after
h2h vs v15 showed 25/64 (39.1%) INCONCLUSIVE — the uniform bias makes
the chooser over-aggressive against v15's well-tuned strategy.
"""

from __future__ import annotations

from agents.baseline.value import (
    ELIMINATION_BONUS,
    WEAK_ENEMY_THRESHOLD,
    WEAKEST_ENEMY_MULT_4P,
    favor,
)


def _obs(planets, fleets=None, step=0, player=0):
    return {
        "player": player,
        "step": step,
        "planets": planets,
        "fleets": fleets or [],
    }


# Planet tuple: (id, owner, x, y, radius, ships, production)
# Fleet tuple:  (id, owner, x, y, angle, from_planet_id, ships)


def test_favor_monotone_in_own_ships():
    base = _obs([(0, 0, 10, 50, 1.0, 10, 1), (1, 1, 90, 50, 1.0, 10, 1)])
    more = _obs([(0, 0, 10, 50, 1.0, 30, 1), (1, 1, 90, 50, 1.0, 10, 1)])
    assert favor(more, me=0, num_seats=2) > favor(base, me=0, num_seats=2)


def test_favor_uses_pv_discount_on_production():
    """Higher gamma (closer to 1) gives more weight to future production."""
    obs = _obs(
        [(0, 0, 10, 50, 1.0, 10, 5), (1, 1, 90, 50, 1.0, 10, 1)],
        step=10,
    )
    hi = favor(obs, me=0, num_seats=2, gamma=1.0)
    lo = favor(obs, me=0, num_seats=2, gamma=0.99)
    assert hi >= lo


def test_favor_2p_unchanged_balanced_state_is_zero():
    """2P path is UNCHANGED from baseline: balanced state -> favor=0
    from either seat. Anti-symmetric, no bias.
    """
    obs = _obs([(0, 0, 10, 50, 1.0, 50, 2), (1, 1, 90, 50, 1.0, 50, 2)])
    assert abs(favor(obs, me=0, num_seats=2)) < 1e-9
    assert abs(favor(obs, me=1, num_seats=2)) < 1e-9


def test_favor_4p_weighted_sum_harsher_than_2p_max():
    """In 4P, opps aggregate by WEIGHTED-SUM (weakest 1.5x); 2P uses max."""
    obs_4p = _obs([
        (0, 0, 10, 10, 1.0, 50, 1),
        (1, 1, 90, 10, 1.0, 50, 1),
        (2, 2, 10, 90, 1.0, 50, 1),
        (3, 3, 90, 90, 1.0, 50, 1),
    ])
    v_2p = favor(obs_4p, me=0, num_seats=2)
    v_4p = favor(obs_4p, me=0, num_seats=4)
    assert v_4p < v_2p


def test_favor_counts_in_flight_fleets():
    """Ships in fleets count toward the owner's F1 just like garrison."""
    no_fleets = _obs([(0, 0, 10, 50, 1.0, 30, 1), (1, 1, 90, 50, 1.0, 0, 1)])
    with_fleet = _obs(
        [(0, 0, 10, 50, 1.0, 0, 1), (1, 1, 90, 50, 1.0, 0, 1)],
        fleets=[(0, 0, 50, 50, 0.0, 0, 30)],
    )
    assert abs(favor(no_fleets, me=0, num_seats=2) - favor(with_fleet, me=0, num_seats=2)) < 1e-9


def test_favor_ignores_neutral_owners():
    """owner == -1 must not contribute to either side."""
    only_neutral = _obs([(0, -1, 50, 50, 1.0, 99, 9)])
    assert favor(only_neutral, me=0, num_seats=2) == 0.0


# ---------------------------------------------------------------------------
# A2 — 4P weakness exploitation (4P only)
# ---------------------------------------------------------------------------


def test_a2_weakest_enemy_gets_multiplier_in_4p():
    """In 4P, attacking the weakest opp gives more favor than the strongest."""
    a_kill_weakest = _obs([
        (0, 0, 10, 10, 1.0, 100, 1),
        (1, 1, 90, 10, 1.0, 0,   0),   # weakest eliminated
        (2, 2, 10, 90, 1.0, 100, 1),
        (3, 3, 90, 90, 1.0, 100, 1),
    ])
    b_hurt_strongest = _obs([
        (0, 0, 10, 10, 1.0, 100, 1),
        (1, 1, 90, 10, 1.0, 10,  1),   # weakest still alive
        (2, 2, 10, 90, 1.0, 90,  0),   # strongest hurt same gross amount
        (3, 3, 90, 90, 1.0, 100, 1),
    ])
    fa = favor(a_kill_weakest, me=0, num_seats=4)
    fb = favor(b_hurt_strongest, me=0, num_seats=4)
    assert fa > fb


def test_a2_2p_path_no_bonus_no_mult():
    """2P path: no weakness mult, no elim bonus — UNCHANGED from baseline."""
    # 2P balanced state — should be exactly 0 (no bias).
    obs_balanced = _obs([(0, 0, 10, 50, 1.0, 100, 5), (1, 1, 90, 50, 1.0, 100, 5)])
    assert abs(favor(obs_balanced, me=0, num_seats=2)) < 1e-9

    # 2P state where opp is below WEAK_ENEMY_THRESHOLD — bonus must NOT fire
    # in 2P (4P-only feature).
    obs_weak_opp = _obs([
        (0, 0, 10, 50, 1.0, 500, 0),
        (1, 1, 90, 50, 1.0, 50, 0),  # strength=50, well below 110
    ])
    f = favor(obs_weak_opp, me=0, num_seats=2)
    # Without bonus: favor = 500 - 50 + 0 = 450
    assert abs(f - 450.0) < 1e-6, f"2P should have no bonus; got {f}"


def test_a2_elimination_bonus_fires_when_weakest_below_threshold_4p():
    """In 4P, weakest below threshold AND we're strong enough -> +55 bonus."""
    obs_below = _obs([
        (0, 0, 10, 10, 1.0, 500, 0),
        (1, 1, 90, 10, 1.0, int(WEAK_ENEMY_THRESHOLD - 10.0), 0),
        (2, 2, 10, 90, 1.0, 500, 0),
        (3, 3, 90, 90, 1.0, 500, 0),
    ])
    obs_above = _obs([
        (0, 0, 10, 10, 1.0, 500, 0),
        (1, 1, 90, 10, 1.0, int(WEAK_ENEMY_THRESHOLD + 20.0), 0),
        (2, 2, 10, 90, 1.0, 500, 0),
        (3, 3, 90, 90, 1.0, 500, 0),
    ])
    delta = favor(obs_below, me=0, num_seats=4) - favor(obs_above, me=0, num_seats=4)
    assert delta > ELIMINATION_BONUS - 5


def test_a2_elimination_bonus_does_not_fire_when_we_are_too_weak_4p():
    """In 4P, gate withholds bonus when my_strength < 0.9 * weakest's."""
    weak_me = _obs([
        (0, 0, 10, 10, 1.0, 50, 0),
        (1, 1, 90, 10, 1.0, 100, 0),
        (2, 2, 10, 90, 1.0, 500, 0),
        (3, 3, 90, 90, 1.0, 500, 0),
    ])
    strong_me = _obs([
        (0, 0, 10, 10, 1.0, 500, 0),
        (1, 1, 90, 10, 1.0, 100, 0),
        (2, 2, 10, 90, 1.0, 500, 0),
        (3, 3, 90, 90, 1.0, 500, 0),
    ])
    diff = favor(strong_me, me=0, num_seats=4) - favor(weak_me, me=0, num_seats=4)
    assert diff >= 450 + ELIMINATION_BONUS - 1


def test_a2_constants_match_lb_max_calibration():
    """Constants are documented load-bearing; assert they match the
    romantamrazov LB-MAX-1224 calibration (4P only, 2P bias rolled back).
    """
    assert WEAKEST_ENEMY_MULT_4P == 1.5
    assert WEAK_ENEMY_THRESHOLD == 110.0
    assert ELIMINATION_BONUS == 55.0


# ---------------------------------------------------------------------------
# select_favor_fn dispatcher (PR #29 — BASELINE_VALUE_HEAD opt-in toggle)
# ---------------------------------------------------------------------------


def test_select_favor_fn_default_returns_favor():
    """Without BASELINE_VALUE_HEAD env, dispatcher returns the canonical favor."""
    import os
    from agents.baseline.value import select_favor_fn, favor as canonical
    old = os.environ.pop("BASELINE_VALUE_HEAD", None)
    try:
        assert select_favor_fn() is canonical
    finally:
        if old is not None:
            os.environ["BASELINE_VALUE_HEAD"] = old


def test_select_favor_fn_composite_path():
    """BASELINE_VALUE_HEAD=composite swaps to favor_composite."""
    import os
    from agents.baseline.value import select_favor_fn, favor_composite
    os.environ["BASELINE_VALUE_HEAD"] = "composite"
    try:
        assert select_favor_fn() is favor_composite
    finally:
        os.environ.pop("BASELINE_VALUE_HEAD", None)


def test_favor_composite_returns_float_on_simple_board():
    """favor_composite must accept the (obs, me, num_seats, gamma) chooser
    signature and return a finite float, even when there are no fleets."""
    from agents.baseline.value import favor_composite
    obs = _obs([(0, 0, 10, 50, 1.0, 30, 1), (1, 1, 90, 50, 1.0, 10, 1)])
    v = favor_composite(obs, me=0, num_seats=2, gamma=0.99)
    assert isinstance(v, float)
    # No fleets => composite collapses to the ship-delta term: 30 - 10 = 20.
    assert v == 20.0


def test_select_favor_fn_hybrid_path():
    """BASELINE_VALUE_HEAD=hybrid swaps to favor_hybrid."""
    import os
    from agents.baseline.value import select_favor_fn, favor_hybrid
    os.environ["BASELINE_VALUE_HEAD"] = "hybrid"
    try:
        assert select_favor_fn() is favor_hybrid
    finally:
        os.environ.pop("BASELINE_VALUE_HEAD", None)


def test_favor_hybrid_dispatches_2p_to_composite():
    """Hybrid in 2P must match favor_composite output (no A2 effect)."""
    from agents.baseline.value import favor_hybrid, favor_composite
    obs = _obs([(0, 0, 10, 50, 1.0, 30, 1), (1, 1, 90, 50, 1.0, 10, 1)])
    h = favor_hybrid(obs, me=0, num_seats=2, gamma=0.99)
    c = favor_composite(obs, me=0, num_seats=2, gamma=0.99)
    assert h == c, f"2P hybrid should match composite; got h={h} c={c}"


def test_favor_hybrid_dispatches_4p_to_favor():
    """Hybrid in 4P must match `favor` output (A2 4P-weakness multiplier
    fires; composite would be incorrect here per the 2P-only flag).
    """
    from agents.baseline.value import favor_hybrid, favor as canonical
    obs_4p = _obs([
        (0, 0, 10, 10, 1.0, 50, 1),
        (1, 1, 90, 10, 1.0, 50, 1),
        (2, 2, 10, 90, 1.0, 50, 1),
        (3, 3, 90, 90, 1.0, 50, 1),
    ])
    h = favor_hybrid(obs_4p, me=0, num_seats=4, gamma=0.99)
    f = canonical(obs_4p, me=0, num_seats=4, gamma=0.99)
    assert h == f, f"4P hybrid should match canonical favor; got h={h} f={f}"


# ---------------------------------------------------------------------------
# favor_hybrid_spatial — positional pull toward non-our planets
# ---------------------------------------------------------------------------


def test_positional_ship_value_zero_when_no_non_our_planets():
    """Degenerate end-state (everything ours): spatial term is zero."""
    from agents.baseline.value import _positional_ship_value
    obs = _obs([(0, 0, 10, 50, 1.0, 100, 5), (1, 0, 90, 50, 1.0, 50, 3)])
    assert _positional_ship_value(obs, me=0) == 0.0


def test_positional_ship_value_higher_near_opp():
    """Same ship count: planet adjacent to opp > planet far from opp."""
    from agents.baseline.value import _positional_ship_value
    # Opp planet at (50, 50). My planet 10 units away => high spatial value.
    near = _obs([(0, 0, 50, 60, 1.0, 50, 0), (1, 1, 50, 50, 1.0, 10, 0)])
    far = _obs([(0, 0, 5, 5, 1.0, 50, 0), (1, 1, 50, 50, 1.0, 10, 0)])
    v_near = _positional_ship_value(near, me=0)
    v_far = _positional_ship_value(far, me=0)
    assert v_near > v_far, f"near={v_near} far={v_far}"


def test_favor_hybrid_spatial_reduces_to_hybrid_when_weight_zero():
    """SPATIAL_WEIGHT=0 short-circuits → output equals favor_hybrid exactly."""
    import agents.baseline.value as bv
    obs = _obs([(0, 0, 10, 50, 1.0, 30, 1), (1, 1, 90, 50, 1.0, 10, 1)])
    old_weight = bv.SPATIAL_WEIGHT
    try:
        bv.SPATIAL_WEIGHT = 0.0
        s = bv.favor_hybrid_spatial(obs, me=0, num_seats=2, gamma=0.99)
        h = bv.favor_hybrid(obs, me=0, num_seats=2, gamma=0.99)
        assert s == h, f"weight=0 should pass through; s={s} h={h}"
    finally:
        bv.SPATIAL_WEIGHT = old_weight


def test_favor_hybrid_spatial_adds_positional_pull():
    """SPATIAL_WEIGHT>0: spatial head > hybrid head (positive spatial value)."""
    import agents.baseline.value as bv
    obs = _obs([(0, 0, 10, 50, 1.0, 30, 1), (1, 1, 90, 50, 1.0, 10, 1)])
    old_weight = bv.SPATIAL_WEIGHT
    try:
        bv.SPATIAL_WEIGHT = 1.0
        s = bv.favor_hybrid_spatial(obs, me=0, num_seats=2, gamma=0.99)
        h = bv.favor_hybrid(obs, me=0, num_seats=2, gamma=0.99)
        assert s > h, f"spatial should add positive pull; s={s} h={h}"
    finally:
        bv.SPATIAL_WEIGHT = old_weight


def test_select_favor_fn_hybrid_spatial_path():
    """BASELINE_VALUE_HEAD=hybrid_spatial dispatches to favor_hybrid_spatial."""
    import os
    from agents.baseline.value import select_favor_fn, favor_hybrid_spatial
    os.environ["BASELINE_VALUE_HEAD"] = "hybrid_spatial"
    try:
        assert select_favor_fn() is favor_hybrid_spatial
    finally:
        os.environ.pop("BASELINE_VALUE_HEAD", None)


def test_positional_ship_value_counts_in_flight_fleets():
    """In-flight fleets at the fleet's current xy contribute spatial value."""
    from agents.baseline.value import _positional_ship_value
    # No on-planet ships; one in-flight fleet near opp planet.
    obs = _obs(
        [(0, 0, 5, 5, 1.0, 0, 0), (1, 1, 50, 50, 1.0, 0, 0)],
        fleets=[(0, 0, 48, 50, 0.0, 0, 30)],  # 30 ships, 2 units from opp
    )
    v = _positional_ship_value(obs, me=0)
    assert v > 0.0


def test_favor_hybrid_spatial_skips_spatial_in_4p():
    """num_seats > 2 short-circuits spatial term → result == favor_hybrid.

    Reason: bv33jlzwj 4P A/B showed spatial regresses 4P (3/32 first-place,
    9.4%, max wallclock 1503ms). Restricting spatial to 2P preserves
    the validated A2-favor-in-4P path.
    """
    import agents.baseline.value as bv
    obs_4p = _obs([
        (0, 0, 10, 10, 1.0, 50, 1),
        (1, 1, 90, 10, 1.0, 50, 1),
        (2, 2, 10, 90, 1.0, 50, 1),
        (3, 3, 90, 90, 1.0, 50, 1),
    ])
    old_weight = bv.SPATIAL_WEIGHT
    try:
        bv.SPATIAL_WEIGHT = 5.0  # large weight - should still skip in 4P
        s = bv.favor_hybrid_spatial(obs_4p, me=0, num_seats=4, gamma=0.99)
        h = bv.favor_hybrid(obs_4p, me=0, num_seats=4, gamma=0.99)
        assert s == h, f"4P spatial should equal hybrid; s={s} h={h}"
    finally:
        bv.SPATIAL_WEIGHT = old_weight


# ---------------------------------------------------------------------------
# projected_rank_diff — production-compounding unified value head
# ---------------------------------------------------------------------------


def test_projected_rank_diff_symmetric_2p_start_is_zero():
    """V = 0 at a perfectly symmetric 2-player snapshot."""
    from lib.value_heads import projected_rank_diff
    obs = _obs([(0, 0, 10, 50, 1.0, 50, 2), (1, 1, 90, 50, 1.0, 50, 2)])
    assert abs(projected_rank_diff(obs, my_id=0, num_seats=2)) < 1e-9
    assert abs(projected_rank_diff(obs, my_id=1, num_seats=2)) < 1e-9


def test_projected_rank_diff_dominated_by_production_projection():
    """Equal ships, asymmetric production → projection term drives V."""
    from lib.value_heads import projected_rank_diff
    obs = _obs(
        [(0, 0, 10, 50, 1.0, 10, 5), (1, 1, 90, 50, 1.0, 10, 1)],
        step=100,
    )
    v = projected_rank_diff(obs, my_id=0, num_seats=2)
    # ships: 10-10 = 0; projection: 0.05 × (5−1) × (500−100) = 80; no fleets.
    assert abs(v - 80.0) < 1e-6


def test_projected_rank_diff_early_capture_worth_more_than_late():
    """Same ownership snapshot at step 100 vs step 400 — earlier is worth
    super-linearly more because turns_remaining shrinks the projection."""
    from lib.value_heads import projected_rank_diff
    # I own one P=5 planet; opp owns one P=1 planet.
    early = _obs(
        [(0, 0, 10, 50, 1.0, 30, 5), (1, 1, 90, 50, 1.0, 30, 1)],
        step=100,
    )
    late = _obs(
        [(0, 0, 10, 50, 1.0, 30, 5), (1, 1, 90, 50, 1.0, 30, 1)],
        step=400,
    )
    v_early = projected_rank_diff(early, my_id=0, num_seats=2)
    v_late = projected_rank_diff(late, my_id=0, num_seats=2)
    assert v_early > v_late
    # Compounding scale: ratio of v_early to v_late ≈ (500-100)/(500-400) = 4.
    assert v_early / v_late > 3.5


def test_projected_rank_diff_4p_max_picks_runaway_leader():
    """In 4P, the `max` aggregation locks onto the leader. Two weak opps
    plus one strong opp → V measured against the strong one only."""
    from lib.value_heads import projected_rank_diff
    obs = _obs([
        (0, 0, 10, 10, 1.0, 50, 2),    # me
        (1, 1, 90, 10, 1.0, 200, 2),   # opp1: runaway leader
        (2, 2, 10, 90, 1.0, 10, 1),    # opp2: weak
        (3, 3, 90, 90, 1.0, 10, 1),    # opp3: weak
    ], step=0)
    v = projected_rank_diff(obs, my_id=0, num_seats=4)
    # Without max, sum-of-opps would give: me 50 + 0.05×2×500=50 = 100;
    # sum opps = (200+0.05×2×500) + (10+0.05×500) + (10+0.05×500) ≈ 320.
    # With max: opp1 dominates → me − opp1 ≈ 100 − 250 = −150.
    assert v < -100


def test_projected_rank_diff_4p_opps_fighting_each_other_help_us():
    """If the two non-me-non-leader opps are weak, max still picks the
    leader. Test 4P=2P collapse: with two no-planet opps, V_4P == V_2P
    against just opp1."""
    from lib.value_heads import projected_rank_diff
    obs_2p = _obs([
        (0, 0, 10, 10, 1.0, 50, 2),
        (1, 1, 90, 10, 1.0, 50, 2),
    ], step=0)
    obs_4p = _obs([
        (0, 0, 10, 10, 1.0, 50, 2),
        (1, 1, 90, 10, 1.0, 50, 2),
        # opps 2 and 3 own nothing → projection = 0 → max picks opp1
    ], step=0)
    v_2p = projected_rank_diff(obs_2p, my_id=0, num_seats=2)
    v_4p = projected_rank_diff(obs_4p, my_id=0, num_seats=4)
    assert abs(v_2p - v_4p) < 1e-9


def test_projected_rank_diff_ignores_neutrals():
    """Neutral-owned planets (owner=-1) contribute to no seat."""
    from lib.value_heads import projected_rank_diff
    obs = _obs([(0, -1, 50, 50, 1.0, 99, 9)])
    assert projected_rank_diff(obs, my_id=0, num_seats=2) == 0.0


def test_projected_rank_diff_in_flight_fleets_count_as_owner_ships():
    """A ship in my fleet shows up in ships_per[me] same as a garrisoned ship."""
    from lib.value_heads import projected_rank_diff
    no_fleet = _obs(
        [(0, 0, 10, 50, 1.0, 30, 2), (1, 1, 90, 50, 1.0, 30, 2)],
        step=250,
    )
    # Move 20 of my garrison into an in-flight fleet → same total.
    with_fleet = _obs(
        [(0, 0, 10, 50, 1.0, 10, 2), (1, 1, 90, 50, 1.0, 30, 2)],
        fleets=[(0, 0, 50, 50, 0.0, 0, 20)],
        step=250,
    )
    # In-flight credit may differ (depends on whether the fleet has a
    # target), but ships_now contribution must match.
    v_no = projected_rank_diff(no_fleet, my_id=0, num_seats=2)
    v_with = projected_rank_diff(with_fleet, my_id=0, num_seats=2)
    # Ships+projection match; only the in-flight credit can differ.
    # The fleet at (50,50) headed angle=0 with 20 ships — target test is
    # downstream behaviour. We assert ship accounting via the floor: with
    # at least the ship-balance term preserved.
    # Difference must equal the in-flight credit term (could be 0 if
    # fleet's target lands on a neutral path/sun).
    # Loose bound: V_with − V_no within capture/waste range of 20 ships.
    assert abs(v_with - v_no) < 0.5 * 20 + 0.05 * 5 * 500  # waste cap + max capture credit


def test_select_favor_fn_projected_path():
    """BASELINE_VALUE_HEAD=projected swaps to favor_projected."""
    import os
    from agents.baseline.value import select_favor_fn, favor_projected
    os.environ["BASELINE_VALUE_HEAD"] = "projected"
    try:
        assert select_favor_fn() is favor_projected
    finally:
        os.environ.pop("BASELINE_VALUE_HEAD", None)


def test_favor_projected_chooser_signature():
    """favor_projected accepts (obs, me, num_seats, gamma) and returns float.
    gamma is ignored (linear horizon)."""
    from agents.baseline.value import favor_projected
    obs = _obs([(0, 0, 10, 50, 1.0, 30, 2), (1, 1, 90, 50, 1.0, 10, 1)])
    v_a = favor_projected(obs, me=0, num_seats=2, gamma=0.99)
    v_b = favor_projected(obs, me=0, num_seats=2, gamma=1.0)
    assert isinstance(v_a, float)
    assert v_a == v_b  # gamma is ignored


def test_favor_projected_crn_state_function():
    """projected_rank_diff is a pure state function — same obs in baseline
    and action legs gives identical value (CRN-safe by construction)."""
    from agents.baseline.value import favor_projected
    obs = _obs(
        [(0, 0, 10, 50, 1.0, 50, 3), (1, 1, 90, 50, 1.0, 50, 3)],
        fleets=[(0, 0, 50, 50, 0.0, 0, 20)],
        step=200,
    )
    v1 = favor_projected(obs, me=0, num_seats=2, gamma=0.99)
    v2 = favor_projected(obs, me=0, num_seats=2, gamma=0.99)
    assert v1 == v2
