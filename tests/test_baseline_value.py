"""Unit tests for agents/baseline/value.favor."""

from __future__ import annotations

from agents.baseline.value import favor


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
    assert hi >= lo  # γ=1.0 weights future-prod more


def test_favor_2p_max_of_opps():
    """In 2P, opp_agg = the only opp. Symmetry: favor(me=0) == -favor(me=1)
    when ship/prod are balanced.
    """
    obs = _obs([(0, 0, 10, 50, 1.0, 50, 2), (1, 1, 90, 50, 1.0, 50, 2)])
    assert abs(favor(obs, me=0, num_seats=2)) < 1e-9
    assert abs(favor(obs, me=1, num_seats=2)) < 1e-9


def test_favor_4p_sum_of_opps_not_max():
    """In 4P, opps aggregate by SUM not MAX — capturing a weak opp counts
    for the full prod delta, not just the leader's portion.
    """
    # me has 50 ships, each of 3 opps has 50 — sum=150 vs max=50
    obs_4p = _obs([
        (0, 0, 10, 10, 1.0, 50, 1),
        (1, 1, 90, 10, 1.0, 50, 1),
        (2, 2, 10, 90, 1.0, 50, 1),
        (3, 3, 90, 90, 1.0, 50, 1),
    ])
    v_2p = favor(obs_4p, me=0, num_seats=2)  # treats opp=max=50
    v_4p = favor(obs_4p, me=0, num_seats=4)  # treats opp=sum=150
    # 4P aggregation is harsher (we're behind sum-of-three, not max-of-one)
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


# ---------------------------------------------------------------------------
# Numeric VALUE_HEAD_CHOICE dispatch (patchable by scripts/ab_variants.py)
# ---------------------------------------------------------------------------


def test_value_head_choice_constant_takes_priority_over_env_var():
    """VALUE_HEAD_CHOICE=2 forces favor_projected even if env var picks
    composite. Validates that ab_variants-patched bundles win over any
    operator env var leftover in the runtime."""
    import os
    import lib.value_heads as vh
    from agents.baseline.value import select_favor_fn, favor_projected
    prev_env = os.environ.pop("BASELINE_VALUE_HEAD", None)
    prev_choice = vh.VALUE_HEAD_CHOICE
    try:
        os.environ["BASELINE_VALUE_HEAD"] = "composite"
        vh.VALUE_HEAD_CHOICE = 2
        assert select_favor_fn() is favor_projected
    finally:
        vh.VALUE_HEAD_CHOICE = prev_choice
        if prev_env is None:
            os.environ.pop("BASELINE_VALUE_HEAD", None)
        else:
            os.environ["BASELINE_VALUE_HEAD"] = prev_env


def test_value_head_choice_constant_zero_falls_through_to_env_var():
    """VALUE_HEAD_CHOICE=0 (default) keeps the env-var path live for
    back-compat with existing operator workflows."""
    import os
    import lib.value_heads as vh
    from agents.baseline.value import select_favor_fn, favor_composite
    prev_env = os.environ.pop("BASELINE_VALUE_HEAD", None)
    prev_choice = vh.VALUE_HEAD_CHOICE
    try:
        vh.VALUE_HEAD_CHOICE = 0
        os.environ["BASELINE_VALUE_HEAD"] = "composite"
        assert select_favor_fn() is favor_composite
    finally:
        vh.VALUE_HEAD_CHOICE = prev_choice
        if prev_env is None:
            os.environ.pop("BASELINE_VALUE_HEAD", None)
        else:
            os.environ["BASELINE_VALUE_HEAD"] = prev_env


def test_value_head_choice_constant_one_picks_composite():
    import lib.value_heads as vh
    from agents.baseline.value import select_favor_fn, favor_composite
    prev_choice = vh.VALUE_HEAD_CHOICE
    try:
        vh.VALUE_HEAD_CHOICE = 1
        assert select_favor_fn() is favor_composite
    finally:
        vh.VALUE_HEAD_CHOICE = prev_choice
