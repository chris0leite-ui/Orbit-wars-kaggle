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
