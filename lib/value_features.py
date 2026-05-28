"""Feature extractor for the learned value head.

Pure function `obs -> np.ndarray(shape=(40,), dtype=float32)` consumed by
`agents/baseline/value_learned.py`. Lives in `lib/` (not `agents/`) so
the bundler inlines it once into the submission and so train-time
self-play data generation can import it without dragging in agent
modules.

Feature layout (40 = 16 × 2 per-seat + 8 global):

  Per-seat block for seat S (indexed `[16*S : 16*(S+1)]`):
    [ 0] ship_total              planet + in-flight ships (raw)
    [ 1] prod_total              total production per turn
    [ 2] planet_count            number of planets owned
    [ 3] in_flight_fleet_count   non-zero fleets owned
    [ 4] in_flight_ship_total    ships currently in flight
    [ 5] mean_garrison           mean ships per planet (0 if no planets)
    [ 6] max_garrison            max ships on any owned planet
    [ 7] planet_ships_total      planet garrisons only (not in-flight)
    [ 8] mean_prod               mean production per planet
    [ 9] max_prod                max production on any owned planet
    [10] mean_dist_to_other      mean dist from owned planets to nearest
                                 non-mine planet
    [11] min_dist_to_other       min  dist from owned planets to nearest
                                 non-mine planet
    [12] high_value_planet_count count of owned planets with prod > 3.0
    [13] incoming_threat_ships   sum of ENEMY fleets within DANGER_RADIUS
                                 of any of our planets
    [14] centroid_x_norm         mean planet x / 100
    [15] centroid_y_norm         mean planet y / 100

  Block 0 (`[ 0:16]`) is for `me`. Block 1 (`[16:32]`) is opp aggregated:
    2P: the single opponent's block, computed identically.
    4P: sum/max/mean over all non-me seats (sums for cumulative features
        like ship_total / prod_total / planet_count; mean/max for
        per-planet features computed on the union of all opp planets).

  Global block `[32:40]`:
    [32] step / 500
    [33] num_seats / 4
    [34] total_planets / 16
    [35] neutral_planets / 16
    [36] total_ships_global / 200
    [37] total_prod_global / 50
    [38] pv_horizon(step, 0)        time-discount factor (lib.scoring)
    [39] in_flight_count_global / 20

Normalisation choices keep all outputs roughly in `[0, 5]` range; the
learned MLP handles the rest. Tuple indices match
`lib/game/interpreter.py:143-148`:
  Planet = (id=0, owner=1, x=2, y=3, radius=4, ships=5, production=6)
  Fleet  = (id=0, owner=1, x=2, y=3, angle=4, from_planet_id=5, ships=6)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from lib.scoring import pv_horizon

FEATURE_DIM = 40
PER_SEAT_FEATURES = 16
GLOBAL_FEATURES = 8
DANGER_RADIUS = 25.0
HIGH_VALUE_PROD = 3.0
EPISODE_STEPS = 500


def _read(obs: Any, attr: str, default: Any) -> Any:
    if hasattr(obs, attr):
        return getattr(obs, attr)
    if isinstance(obs, dict):
        return obs.get(attr, default)
    return default


def _seat_block(
    me: int,
    planets: list,
    fleets: list,
    other_planet_xy: list[tuple[float, float]],
) -> np.ndarray:
    """16-feature block for seat `me`.

    `other_planet_xy` is the list of (x,y) for planets NOT owned by `me`
    (passed in to avoid recomputation across the two seat-blocks).
    """
    out = np.zeros(PER_SEAT_FEATURES, dtype=np.float32)

    my_planets = [p for p in planets if int(p[1]) == me]
    my_fleets = [f for f in fleets if int(f[1]) == me]

    planet_ships = sum(float(p[5]) for p in my_planets)
    planet_prods = [float(p[6]) for p in my_planets]
    garrisons = [float(p[5]) for p in my_planets]
    in_flight_ships = sum(float(f[6]) for f in my_fleets)
    n_planets = len(my_planets)

    out[0] = planet_ships + in_flight_ships
    out[1] = sum(planet_prods)
    out[2] = float(n_planets)
    out[3] = float(len(my_fleets))
    out[4] = in_flight_ships
    out[5] = (planet_ships / n_planets) if n_planets else 0.0
    out[6] = max(garrisons) if garrisons else 0.0
    out[7] = planet_ships
    out[8] = (sum(planet_prods) / n_planets) if n_planets else 0.0
    out[9] = max(planet_prods) if planet_prods else 0.0

    # Distance features: mean / min over my planets of dist-to-nearest-
    # non-mine-planet. If no other planets exist or we have no planets,
    # leave at 0.
    if my_planets and other_planet_xy:
        d_per_planet: list[float] = []
        for p in my_planets:
            px, py = float(p[2]), float(p[3])
            d = min(math.hypot(px - ox, py - oy) for ox, oy in other_planet_xy)
            d_per_planet.append(d)
        out[10] = float(np.mean(d_per_planet))
        out[11] = float(min(d_per_planet))

    out[12] = float(sum(1 for prod in planet_prods if prod > HIGH_VALUE_PROD))

    # Incoming threat: enemy fleets within DANGER_RADIUS of any of MY planets.
    if my_planets:
        my_planet_xy = [(float(p[2]), float(p[3])) for p in my_planets]
        threat = 0.0
        for f in fleets:
            if int(f[1]) == me or int(f[1]) < 0:
                continue
            fx, fy = float(f[2]), float(f[3])
            for px, py in my_planet_xy:
                if math.hypot(fx - px, fy - py) <= DANGER_RADIUS:
                    threat += float(f[6])
                    break
        out[13] = threat

        out[14] = float(np.mean([xy[0] for xy in my_planet_xy])) / 100.0
        out[15] = float(np.mean([xy[1] for xy in my_planet_xy])) / 100.0

    return out


def _opp_aggregated_block(
    me: int,
    num_seats: int,
    planets: list,
    fleets: list,
) -> np.ndarray:
    """16-feature block aggregated over all non-`me` seats.

    For 2P this equals the opp's seat_block. For 4P it's a roll-up:
    cumulative features (ship_total, prod_total, planet_count, in-flight)
    are summed; per-planet derived features (mean/max/dist) are computed
    on the union of all opp planets.
    """
    opp_planets = [p for p in planets if int(p[1]) != me and int(p[1]) >= 0]
    opp_fleets = [f for f in fleets if int(f[1]) != me and int(f[1]) >= 0]
    # "other planets" for distance features = MY planets (for symmetry with
    # how the me-block computes dist-to-non-mine).
    my_planet_xy = [
        (float(p[2]), float(p[3])) for p in planets if int(p[1]) == me
    ]

    out = np.zeros(PER_SEAT_FEATURES, dtype=np.float32)

    planet_ships = sum(float(p[5]) for p in opp_planets)
    planet_prods = [float(p[6]) for p in opp_planets]
    garrisons = [float(p[5]) for p in opp_planets]
    in_flight_ships = sum(float(f[6]) for f in opp_fleets)
    n_planets = len(opp_planets)

    out[0] = planet_ships + in_flight_ships
    out[1] = sum(planet_prods)
    out[2] = float(n_planets)
    out[3] = float(len(opp_fleets))
    out[4] = in_flight_ships
    out[5] = (planet_ships / n_planets) if n_planets else 0.0
    out[6] = max(garrisons) if garrisons else 0.0
    out[7] = planet_ships
    out[8] = (sum(planet_prods) / n_planets) if n_planets else 0.0
    out[9] = max(planet_prods) if planet_prods else 0.0

    if opp_planets and my_planet_xy:
        d_per_planet: list[float] = []
        for p in opp_planets:
            px, py = float(p[2]), float(p[3])
            d = min(math.hypot(px - ox, py - oy) for ox, oy in my_planet_xy)
            d_per_planet.append(d)
        out[10] = float(np.mean(d_per_planet))
        out[11] = float(min(d_per_planet))

    out[12] = float(sum(1 for prod in planet_prods if prod > HIGH_VALUE_PROD))

    # Incoming threat for opp aggregate = MY fleets within DANGER_RADIUS of
    # any opp planet.
    if opp_planets:
        opp_planet_xy = [(float(p[2]), float(p[3])) for p in opp_planets]
        threat = 0.0
        for f in fleets:
            if int(f[1]) != me:
                continue
            fx, fy = float(f[2]), float(f[3])
            for px, py in opp_planet_xy:
                if math.hypot(fx - px, fy - py) <= DANGER_RADIUS:
                    threat += float(f[6])
                    break
        out[13] = threat

        out[14] = float(np.mean([xy[0] for xy in opp_planet_xy])) / 100.0
        out[15] = float(np.mean([xy[1] for xy in opp_planet_xy])) / 100.0

    return out


def extract_features(
    obs: Any, me: int, num_seats: int = 2
) -> np.ndarray:
    """Build the 40-dim feature vector for value-head inference / training.

    `obs` is whatever the env hands the agent: a dict with `planets`,
    `fleets`, `step` keys, OR a kaggle_environments observation object
    with same attributes. Schema mirrors `agents/baseline/value.py:_read`.

    Returns a `float32` ndarray of shape (40,). Order is fixed and
    documented in this module's header — DO NOT permute without retraining
    the value head.
    """
    planets = list(_read(obs, "planets", []) or [])
    fleets = list(_read(obs, "fleets", []) or [])
    step = int(_read(obs, "step", 0))

    other_planet_xy_for_me = [
        (float(p[2]), float(p[3])) for p in planets if int(p[1]) != me
    ]

    out = np.zeros(FEATURE_DIM, dtype=np.float32)
    out[0:PER_SEAT_FEATURES] = _seat_block(
        me, planets, fleets, other_planet_xy_for_me
    )
    out[PER_SEAT_FEATURES:2 * PER_SEAT_FEATURES] = _opp_aggregated_block(
        me, num_seats, planets, fleets
    )

    # Global block.
    total_ships = sum(float(p[5]) for p in planets) + sum(
        float(f[6]) for f in fleets
    )
    total_prod = sum(float(p[6]) for p in planets if int(p[1]) >= 0)
    neutral = sum(1 for p in planets if int(p[1]) < 0)
    total_planets = len(planets)

    g = 2 * PER_SEAT_FEATURES
    out[g + 0] = step / float(EPISODE_STEPS)
    out[g + 1] = num_seats / 4.0
    out[g + 2] = total_planets / 16.0
    out[g + 3] = neutral / 16.0
    out[g + 4] = total_ships / 200.0
    out[g + 5] = total_prod / 50.0
    out[g + 6] = float(pv_horizon(step, 0))
    out[g + 7] = len(fleets) / 20.0

    return out
