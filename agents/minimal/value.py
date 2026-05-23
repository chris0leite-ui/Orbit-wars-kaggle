"""Leaf value: (my_ships - opp_ships) + (my_prod - opp_prod) * pv.

2P aggregates opps with max(); 4P with sum(). PV-discount keeps the
production term on the same scale as the ship term so the chooser
doesn't over-weight captures in late game.
"""

from __future__ import annotations

from lib.scoring import pv_horizon

EPISODE_STEPS = 500
GAMMA = 0.99


def _read(obs, attr, default):
    if hasattr(obs, attr):
        return getattr(obs, attr)
    return obs.get(attr, default) if isinstance(obs, dict) else default


def favor(obs, me: int, num_seats: int = 2) -> float:
    planets = _read(obs, "planets", []) or []
    fleets = _read(obs, "fleets", []) or []
    step = int(_read(obs, "step", 0))

    ships: dict[int, float] = {}
    prod: dict[int, float] = {}
    for p in planets:
        o = int(p[1])
        if o < 0:
            continue
        ships[o] = ships.get(o, 0.0) + float(p[5])
        prod[o] = prod.get(o, 0.0) + float(p[6])
    for f in fleets:
        o = int(f[1])
        if o < 0:
            continue
        ships[o] = ships.get(o, 0.0) + float(f[6])

    me_s = ships.get(me, 0.0)
    me_p = prod.get(me, 0.0)
    if num_seats <= 2:
        opp_s = max((v for k, v in ships.items() if k != me), default=0.0)
        opp_p = max((v for k, v in prod.items() if k != me), default=0.0)
    else:
        opp_s = sum(v for k, v in ships.items() if k != me)
        opp_p = sum(v for k, v in prod.items() if k != me)

    pv = pv_horizon(step, 0, gamma=GAMMA, t_total=EPISODE_STEPS)
    return (me_s - opp_s) + (me_p - opp_p) * pv
