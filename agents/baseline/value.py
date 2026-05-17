"""Leaf value function: F1 + F2 favor with PV-discounted production.

F1 = my_ships - opp_ships_agg          (in-flight + on-planet)
F2 = (my_prod - opp_prod_agg) * pv     (pv = pv_horizon discount)

PV-discount keeps F2 on a comparable scale to F1; without it the future-
production term over-weights captures by ~100x in late game and the
chooser stops valuing ship preservation. opp aggregation is max-of-opps
in 2P and sum-of-opps in 4P (weak-opp captures get full credit).

Opt-in alternative: `BASELINE_VALUE_HEAD=composite` switches the chooser to
`lib.value_heads.composite_capture_value` (waste + capture-aware per-fleet
credit). 2P-only — composite does not distinguish opp identity in 4P.
Default remains `favor` (proven on v15 line at live μ~1108).
"""

from __future__ import annotations

import os

from lib.scoring import pv_horizon

EPISODE_STEPS = 500
DEFAULT_GAMMA = 0.99


def _read(obs, attr, default):
    if hasattr(obs, attr):
        return getattr(obs, attr)
    return obs.get(attr, default) if isinstance(obs, dict) else default


def favor(obs, me: int, num_seats: int = 2, gamma: float = DEFAULT_GAMMA) -> float:
    planets = _read(obs, "planets", []) or []
    fleets = _read(obs, "fleets", []) or []
    step = int(_read(obs, "step", 0))

    ships_by_owner: dict[int, float] = {}
    prod_by_owner: dict[int, float] = {}
    for p in planets:
        owner = int(p[1])
        if owner < 0:
            continue
        ships_by_owner[owner] = ships_by_owner.get(owner, 0.0) + float(p[5])
        prod_by_owner[owner] = prod_by_owner.get(owner, 0.0) + float(p[6])
    for f in fleets:
        owner = int(f[1])
        if owner < 0:
            continue
        ships_by_owner[owner] = ships_by_owner.get(owner, 0.0) + float(f[6])

    my_ships = ships_by_owner.get(me, 0.0)
    my_prod = prod_by_owner.get(me, 0.0)

    if num_seats <= 2:
        opp_ships = max(
            (v for k, v in ships_by_owner.items() if k != me), default=0.0,
        )
        opp_prod = max(
            (v for k, v in prod_by_owner.items() if k != me), default=0.0,
        )
    else:
        opp_ships = sum(v for k, v in ships_by_owner.items() if k != me)
        opp_prod = sum(v for k, v in prod_by_owner.items() if k != me)

    pv = pv_horizon(step, 0, gamma=gamma, t_total=EPISODE_STEPS)
    return (my_ships - opp_ships) + (my_prod - opp_prod) * pv


def favor_composite(obs, me: int, num_seats: int = 2,
                    gamma: float = DEFAULT_GAMMA) -> float:
    """4P-aware composite: `composite_capture_value` in 2P, fall back
    to the proven `favor` (sum-of-opps) in 4P.

    `composite_capture_value` collapses all non-me planets into one
    "enemy" bucket — it has no opp-seat differentiation. `favor` uses
    `max-of-opps` in 2P and `sum-of-opps` in 4P, where the latter
    gives weak-opp captures full credit. ~36% of ladder games are 4P;
    routing those to `favor` avoids regression.

    Validated 2P A/B (2026-05-17):
    - vs v9_scavenge (μ=1119.9 team peak): 30/32 = 93.8% (Wlo=0.799)
    - vs v15 (μ=1108.4 champion):         43/64 = 67.2% (Wlo=0.550)
    - vs v7_0 / v4_planner / v3.5.1:      ≥87% all PASS

    See knowledge-base/thoughts/2026-05-17-composite-head-beats-team-peaks.md.
    """
    if num_seats >= 4:
        return favor(obs, me, num_seats, gamma=gamma)
    from lib.value_heads import composite_capture_value
    return composite_capture_value(obs, me)


def select_favor_fn():
    """Pick the leaf value function. Default = `favor` (the v15 baseline).

    Switch via env var `BASELINE_VALUE_HEAD=composite`. Anything else
    falls through to the default. The chooser uses the SAME function for
    both `build_idle_baseline` and `score_action` so the Δ stays well-
    defined.
    """
    choice = os.environ.get("BASELINE_VALUE_HEAD", "").strip().lower()
    if choice == "composite":
        return favor_composite
    return favor
