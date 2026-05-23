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
# Attack-pull positional term (2026-05-23 passivity fix)
# ---------------------------------------------------------------------------


def test_attack_pull_returns_zero_with_no_enemy():
    """No enemy planets → term is 0 (degenerate end-state)."""
    from agents.baseline.value import _attack_pull_ship_value
    obs = _obs([
        (0, 0, 10, 50, 1.0, 100, 1),   # mine
        (1, -1, 50, 50, 1.0, 30, 1),   # neutral
    ])
    assert _attack_pull_ship_value(obs, me=0) == 0.0


def test_attack_pull_higher_when_ships_forward():
    """100 ships at distance 20 from enemy > 100 ships at distance 80."""
    from agents.baseline.value import _attack_pull_ship_value
    # State A: my planet at (10, 50), enemy at (90, 50). dist=80.
    state_back = _obs([
        (0, 0, 10.0, 50.0, 1.0, 100, 1),
        (1, 1, 90.0, 50.0, 1.0, 10, 1),
    ])
    # State B: my planet moved to (70, 50). dist=20.
    state_forward = _obs([
        (0, 0, 70.0, 50.0, 1.0, 100, 1),
        (1, 1, 90.0, 50.0, 1.0, 10, 1),
    ])
    v_back = _attack_pull_ship_value(state_back, me=0)
    v_forward = _attack_pull_ship_value(state_forward, me=0)
    assert v_forward > v_back, (
        f"forward={v_forward} should exceed back={v_back}; "
        f"positional pull is broken"
    )


def test_attack_pull_credits_in_flight_fleets():
    """A fleet in flight toward enemy contributes to the term — without
    this the chooser can't credit "ships en route to attack" as positional
    value, which is half the incentive correction.
    """
    from agents.baseline.value import _attack_pull_ship_value
    # All ships at home (distance 80 from enemy).
    home = _obs([
        (0, 0, 10.0, 50.0, 1.0, 100, 1),
        (1, 1, 90.0, 50.0, 1.0, 10, 1),
    ])
    # Same total ship count but 50 in-flight at (70, 50) toward enemy.
    inflight = _obs(
        [
            (0, 0, 10.0, 50.0, 1.0, 50, 1),
            (1, 1, 90.0, 50.0, 1.0, 10, 1),
        ],
        fleets=[(0, 0, 70.0, 50.0, 0.0, 0, 50)],
    )
    v_home = _attack_pull_ship_value(home, me=0)
    v_inflight = _attack_pull_ship_value(inflight, me=0)
    assert v_inflight > v_home, (
        f"in-flight fleet at d=20 should score higher than all-home at d=80; "
        f"home={v_home} inflight={v_inflight}"
    )


def test_attack_pull_ignores_neutral_planets():
    """Distance to neutral planets does NOT factor in — only enemy counts.
    Otherwise the term would give weak signal early-game when neutrals are
    everywhere (the bug in the existing _positional_ship_value)."""
    from agents.baseline.value import _attack_pull_ship_value
    # My planet at (10, 50) with 100 ships.
    # Two scenarios: nearby neutral vs nearby enemy. Score for the nearby
    # enemy should be MUCH higher; nearby neutral should not pull at all.
    near_neutral = _obs([
        (0, 0, 10.0, 50.0, 1.0, 100, 1),
        (1, -1, 30.0, 50.0, 1.0, 30, 1),   # neutral at d=20 from me
        (2, 1, 90.0, 50.0, 1.0, 10, 1),    # enemy at d=80
    ])
    near_enemy = _obs([
        (0, 0, 10.0, 50.0, 1.0, 100, 1),
        (1, -1, 90.0, 50.0, 1.0, 30, 1),   # neutral at d=80
        (2, 1, 30.0, 50.0, 1.0, 10, 1),    # enemy at d=20
    ])
    v_near_neutral = _attack_pull_ship_value(near_neutral, me=0)
    v_near_enemy = _attack_pull_ship_value(near_enemy, me=0)
    assert v_near_enemy > v_near_neutral, (
        f"near-enemy={v_near_enemy} should exceed near-neutral={v_near_neutral}; "
        f"the term must ignore neutrals (only enemy matters)"
    )


def test_select_favor_fn_attack_pull_path():
    """BASELINE_VALUE_HEAD=hybrid_attack_pull swaps to the new head."""
    import os
    from agents.baseline.value import select_favor_fn, favor_hybrid_attack_pull
    os.environ["BASELINE_VALUE_HEAD"] = "hybrid_attack_pull"
    try:
        assert select_favor_fn() is favor_hybrid_attack_pull
    finally:
        os.environ.pop("BASELINE_VALUE_HEAD", None)
