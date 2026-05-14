"""tests/test_favor.py — favor() unit smoke.

Verifies:
  - 2-player symmetric state → favor(s, 0) + favor(s, 1) ≈ 0
  - Lopsided garrison → favor_diff sign matches expectation
  - Comet near end-of-life contributes less than fresh comet of same prod
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from favor import favor, favor_breakdown, FavorConfig


def _obs(planets=(), fleets=(), comets=(), comet_planet_ids=(), step=0):
    return {
        "planets": [list(p) for p in planets],
        "fleets": [list(f) for f in fleets],
        "comets": [dict(c) for c in comets],
        "comet_planet_ids": list(comet_planet_ids),
        "step": step,
    }


def test_symmetric_zero_sum():
    """Two players, mirrored planets and fleets → favor(0) ≈ −favor(1)."""
    # planets: [id, owner, x, y, radius, ships, production]
    obs = _obs(
        planets=[
            (0, 0, 20.0, 20.0, 2.0, 10, 3),  # P0
            (1, 1, 80.0, 80.0, 2.0, 10, 3),  # P1 mirror
            (2, -1, 50.0, 50.0, 1.5, 5, 1),  # neutral, central
        ],
        fleets=[
            (10, 0, 30.0, 30.0, 0.0, 0, 5),
            (11, 1, 70.0, 70.0, math.pi, 1, 5),
        ],
        step=100,
    )
    f0 = favor(obs, 0)
    f1 = favor(obs, 1)
    assert abs(f0 + f1) < 1e-6, f"symmetric should be zero-sum; got f0={f0}, f1={f1}"


def test_ship_lead_dominant():
    """Lopsided garrison favors the player with more ships."""
    obs = _obs(
        planets=[
            (0, 0, 20.0, 20.0, 2.0, 100, 3),   # me, lots of ships
            (1, 1, 80.0, 80.0, 2.0, 5,   3),   # opp, few
        ],
        step=200,
    )
    f0 = favor(obs, 0)
    f1 = favor(obs, 1)
    # 100 - 5 = 95 ship lead; production tied; horizon = 300; F2 = 0.
    assert f0 - f1 > 0, f"player 0 should be favored; got f0={f0}, f1={f1}"
    bd = favor_breakdown(obs, 0)
    assert bd["F1_ship_lead"] == 95
    assert bd["F2_prod_lead_x_horizon"] == 0  # production tied


def test_production_lead_scales_with_horizon():
    """Same production lead at turn 50 should be worth more than at turn 450."""
    base_planets = [
        (0, 0, 20.0, 20.0, 2.0, 10, 5),  # me, prod 5
        (1, 1, 80.0, 80.0, 2.0, 10, 1),  # opp, prod 1
    ]
    early = _obs(planets=base_planets, step=50)
    late  = _obs(planets=base_planets, step=450)
    f_early = favor(early, 0) - favor(early, 1)
    f_late  = favor(late,  0) - favor(late,  1)
    assert f_early > f_late, (
        f"production lead should matter more early; early={f_early}, late={f_late}"
    )


def test_comet_decay():
    """A comet near end-of-life contributes less production than a fresh comet."""
    # comet_planet_ids tells favor() which planet IDs are comets.
    # comets group exposes paths + path_index; with path_index near len(path),
    # remaining_lifetime is small → contribution scales down.
    path = [[float(x), 0.0] for x in range(100)]  # 100-step path
    fresh_comet_group = {
        "planet_ids": [5, 6, 7, 8],
        "paths": [path, path, path, path],
        "path_index": 0,    # full lifetime ahead
    }
    expiring_comet_group = {
        "planet_ids": [5, 6, 7, 8],
        "paths": [path, path, path, path],
        "path_index": 95,   # only 5 steps left
    }

    planets = [
        (0, 0, 20.0, 20.0, 2.0, 10, 1),   # me, stable planet prod 1
        (1, 1, 80.0, 80.0, 2.0, 10, 1),   # opp, stable planet prod 1
        (5, 0, 50.0, 50.0, 1.0, 0, 5),    # comet owned by me, prod 5
    ]
    fresh = _obs(planets=planets, comets=[fresh_comet_group], comet_planet_ids=[5,6,7,8], step=100)
    expiring = _obs(planets=planets, comets=[expiring_comet_group], comet_planet_ids=[5,6,7,8], step=100)

    f_fresh = favor(fresh, 0) - favor(fresh, 1)
    f_exp = favor(expiring, 0) - favor(expiring, 1)
    assert f_fresh > f_exp, (
        f"fresh comet should be worth more than expiring; "
        f"fresh_diff={f_fresh}, expiring_diff={f_exp}"
    )


def test_fleets_in_transit_count_as_ships():
    """An in-transit fleet should still count in F1 (ships exist; just not on a planet)."""
    obs = _obs(
        planets=[
            (0, 0, 20.0, 20.0, 2.0, 10, 3),
            (1, 1, 80.0, 80.0, 2.0, 60, 3),  # opp has 60 garrison
        ],
        fleets=[
            (10, 0, 50.0, 50.0, 0.0, 0, 50),  # but I have 50 ships in transit
        ],
        step=200,
    )
    bd0 = favor_breakdown(obs, 0)
    # my ships = 10 garrison + 50 fleet = 60; opp = 60; F1 = 0
    assert bd0["my_ships"] == 60
    assert bd0["F1_ship_lead"] == 0
