"""Leaf value function: F1 + F2 favor with PV-discounted production.

F1 = my_ships - opp_ships_agg          (in-flight + on-planet)
F2 = (my_prod - opp_prod_agg) * pv     (pv = pv_horizon discount)

PV-discount keeps F2 on a comparable scale to F1; without it the future-
production term over-weights captures by ~100x in late game and the
chooser stops valuing ship preservation. opp aggregation is max-of-opps
in 2P and weighted-sum-of-opps in 4P (weakest 1.5x).

A2 (4P weakness exploitation + 2P enemy bias) derives from
romantamrazov/orbit-star-wars-lb-max-1224 (peak LB μ=1224, +109 above
our v15 ceiling). Two multipliers + an elimination bonus:

  - 2P: uniform 1.25x bias on the single opponent's ships + production.
    Makes capture trades positive-EV (was 0-EV at exact parity).
  - 4P: 1.5x bias on the WEAKEST opponent's contribution; other opps
    unweighted. Biases leaf valuation toward states that further
    weaken (or eliminate) them.
  - Elimination bonus (both formats): +55 when weakest's strength
    (ships + 15*prod) <= 110 AND my_strength >= 0.9 * weakest's
    (only fire when WE can finish — no elim-then-die bias).
"""

from __future__ import annotations

from lib.scoring import pv_horizon

EPISODE_STEPS = 500
DEFAULT_GAMMA = 0.99

ELIMINATION_BONUS = 55.0
WEAK_ENEMY_THRESHOLD = 110.0
WEAKEST_ENEMY_MULT_4P = 1.5
WEAKEST_ENEMY_MULT_2P = 1.25
ELIMINATION_GATE_RATIO = 0.9
STRENGTH_PROD_WEIGHT = 15.0


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

    opps = sorted(
        o for o in (set(ships_by_owner) | set(prod_by_owner))
        if o != me and o >= 0
    )

    if not opps:
        # No surviving enemies — pure F1 + F2 from my side, no bonus.
        opp_ships = opp_prod = 0.0
        weakest_str = float("inf")
    elif num_seats <= 2 or len(opps) < 2:
        # 2P (or degenerate ≤1 opp survives): uniform 1.25x on the only opp.
        only_opp = max(opps, key=lambda o: ships_by_owner.get(o, 0.0)
                                            + prod_by_owner.get(o, 0.0)
                                            * STRENGTH_PROD_WEIGHT)
        opp_ships = ships_by_owner.get(only_opp, 0.0) * WEAKEST_ENEMY_MULT_2P
        opp_prod = prod_by_owner.get(only_opp, 0.0) * WEAKEST_ENEMY_MULT_2P
        weakest_str = (ships_by_owner.get(only_opp, 0.0)
                       + prod_by_owner.get(only_opp, 0.0) * STRENGTH_PROD_WEIGHT)
    else:
        # 4P: weighted sum (weakest 1.5x).
        opp_strengths = {
            o: ships_by_owner.get(o, 0.0)
               + prod_by_owner.get(o, 0.0) * STRENGTH_PROD_WEIGHT
            for o in opps
        }
        weakest = min(opps, key=lambda o: opp_strengths[o])
        weakest_str = opp_strengths[weakest]
        opp_ships = sum(
            ships_by_owner.get(o, 0.0)
            * (WEAKEST_ENEMY_MULT_4P if o == weakest else 1.0)
            for o in opps
        )
        opp_prod = sum(
            prod_by_owner.get(o, 0.0)
            * (WEAKEST_ENEMY_MULT_4P if o == weakest else 1.0)
            for o in opps
        )

    # Elimination bonus (both formats): weakest at-or-below threshold AND
    # we're strong enough to finish them (gates against elim-then-die).
    my_strength = my_ships + my_prod * STRENGTH_PROD_WEIGHT
    if (opps and weakest_str <= WEAK_ENEMY_THRESHOLD
            and my_strength >= ELIMINATION_GATE_RATIO * weakest_str):
        elim_bonus = ELIMINATION_BONUS
    else:
        elim_bonus = 0.0

    pv = pv_horizon(step, 0, gamma=gamma, t_total=EPISODE_STEPS)
    return (my_ships - opp_ships) + (my_prod - opp_prod) * pv + elim_bonus
