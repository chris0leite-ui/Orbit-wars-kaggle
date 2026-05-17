"""Unit tests for agents/baseline/value.favor."""

from __future__ import annotations

from agents.baseline.value import (
    ELIMINATION_BONUS,
    WEAK_ENEMY_THRESHOLD,
    WEAKEST_ENEMY_MULT_2P,
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
    assert hi >= lo  # γ=1.0 weights future-prod more


def test_favor_2p_consistent_from_either_seat():
    """In 2P with 1.25x enemy bias, balanced state is non-zero but BOTH
    seats compute the SAME magnitude (their "perceived disadvantage").
    Strategic anti-symmetry holds at the *action-Δ* level — both biased
    agents play more aggressively, a symmetric strategic change.
    """
    obs = _obs([(0, 0, 10, 50, 1.0, 50, 2), (1, 1, 90, 50, 1.0, 50, 2)])
    f0 = favor(obs, me=0, num_seats=2)
    f1 = favor(obs, me=1, num_seats=2)
    # Same magnitude from either perspective on a balanced state.
    assert abs(f0 - f1) < 1e-9, f"f0={f0} f1={f1} should be equal"


def test_favor_4p_weighted_sum_harsher_than_2p_max():
    """In 4P, opps aggregate by WEIGHTED-SUM (weakest 1.5x); 2P uses
    single-opp 1.25x. 4P aggregation is still more punishing.
    """
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
    """owner == -1 must not contribute to either side. No bonus when no
    surviving enemies.
    """
    only_neutral = _obs([(0, -1, 50, 50, 1.0, 99, 9)])
    assert favor(only_neutral, me=0, num_seats=2) == 0.0


# ---------------------------------------------------------------------------
# A2 — 4P weakness exploitation + 2P enemy bias
# ---------------------------------------------------------------------------


def test_a2_weakest_enemy_gets_multiplier_in_4p():
    """In 4P, attacking the weakest opp gives more favor than the strongest.

    Setup: me has 100 ships, opp_1 (weakest) is hurt vs opp_2 (strongest)
    hurt by the same gross amount. Killing weakest gets the 1.5x mult AND
    elim bonus when threshold/gate clear.
    """
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
    assert fa > fb, (
        f"killing weakest ({fa}) should score higher than hurting strongest ({fb})"
    )


def test_a2_2p_uniform_enemy_bias():
    """In 2P, the single opp's contribution is scaled by WEAKEST_ENEMY_MULT_2P.
    Equivalent to a uniform enemy bias: same state as no-mult would be
    1.25x more "punishing" — but our chooser uses Δ, so this makes captures
    positive-EV where they'd be 0-EV without the bias.
    """
    obs = _obs([(0, 0, 10, 50, 1.0, 100, 5), (1, 1, 90, 50, 1.0, 100, 5)])
    f = favor(obs, me=0, num_seats=2)
    # Balanced state, but with bias: F1 = 100 - 1.25*100 = -25 (perceived disadvantage)
    # F2 = (5 - 1.25*5) * pv = -1.25 * pv ≈ -125 (at step=0)
    # No elim (both at strength 175 > 110).
    assert f < 0, f"with 1.25x enemy bias, balanced state should perceive disadvantage; got {f}"


def test_a2_elimination_bonus_fires_when_weakest_below_threshold():
    """When weakest's strength <= WEAK_ENEMY_THRESHOLD AND we're strong
    enough (>=0.9x theirs), ELIMINATION_BONUS is added to favor.
    """
    obs_below = _obs([
        (0, 0, 10, 10, 1.0, 500, 0),
        (1, 1, 90, 10, 1.0, int(WEAK_ENEMY_THRESHOLD - 10.0), 0),  # 100; below threshold
        (2, 2, 10, 90, 1.0, 500, 0),
        (3, 3, 90, 90, 1.0, 500, 0),
    ])
    obs_above = _obs([
        (0, 0, 10, 10, 1.0, 500, 0),
        (1, 1, 90, 10, 1.0, int(WEAK_ENEMY_THRESHOLD + 20.0), 0),  # 130; above
        (2, 2, 10, 90, 1.0, 500, 0),
        (3, 3, 90, 90, 1.0, 500, 0),
    ])
    f_below = favor(obs_below, me=0, num_seats=4)
    f_above = favor(obs_above, me=0, num_seats=4)
    delta = f_below - f_above
    # f_below: opp1=100*1.5=150 (less than f_above's 130*1.5=195); + ELIM=55
    # delta = (-195+150) + 55 = -45 + 55 = no wait:
    # f_below opp_ships = 100*1.5 + 500 + 500 = 1150; favor = (500 - 1150) + 0 + 55 = -595
    # f_above opp_ships = 130*1.5 + 500 + 500 = 1195; favor = (500 - 1195) + 0 + 0  = -695
    # delta = -595 - (-695) = +100 (mix of mult diff + elim bonus)
    assert delta > ELIMINATION_BONUS - 5, f"expected elim bonus, got delta={delta}"


def test_a2_elimination_bonus_does_not_fire_when_we_are_too_weak():
    """Gate: my_strength >= 0.9 * weakest_strength. Otherwise no bonus."""
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
    f_weak = favor(weak_me, me=0, num_seats=4)
    f_strong = favor(strong_me, me=0, num_seats=4)
    # f_strong gains 450 F1 + ELIM=55. f_weak does NOT have elim bonus.
    diff = f_strong - f_weak
    assert diff >= 450 + ELIMINATION_BONUS - 1, (
        f"gate should withhold bonus from weak side: diff={diff}"
    )


def test_a2_constants_match_lb_max_calibration():
    """Constants are documented load-bearing; assert they match the
    romantamrazov LB-MAX-1224 calibration so a future tweak surfaces.
    """
    assert WEAKEST_ENEMY_MULT_4P == 1.5
    assert WEAKEST_ENEMY_MULT_2P == 1.25
    assert WEAK_ENEMY_THRESHOLD == 110.0
    assert ELIMINATION_BONUS == 55.0
