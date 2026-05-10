"""Fleet primitives for Orbit Wars.

Implements the fleet-speed formula from `data/README.md::Fleet Speed`:

    speed = 1.0 + (max_speed - 1.0) * (log(ships) / log(1000)) ^ 1.5

with `max_speed` defaulting to 6.0 (Configuration table).
"""

from __future__ import annotations

import math

from .geometry import Point, dist

DEFAULT_MAX_SPEED: float = 6.0
LOG_1000: float = math.log(1000.0)


def speed(ships: int | float, max_speed: float = DEFAULT_MAX_SPEED) -> float:
    """Speed (board units per turn) for a fleet of size `ships`.

    Spec corner cases:
    - 1 ship  → exactly 1.0 (log(1)=0).
    - 1000 ships → exactly `max_speed` (log(1000)/log(1000)=1).
    - ships <= 0  → 1.0 (avoid log of non-positive; treat as floor speed).
    - ships > 1000 → clamped at `max_speed` (the formula would over-shoot
      otherwise; the env caps fleet speed at maxSpeed).
    """
    if ships <= 1:
        return 1.0
    if ships >= 1000:
        return float(max_speed)
    ratio = math.log(ships) / LOG_1000
    return 1.0 + (max_speed - 1.0) * (ratio ** 1.5)


def travel_time(src: Point, dst: Point, ships: int | float,
                max_speed: float = DEFAULT_MAX_SPEED) -> float:
    """Float turns for a `ships`-ship fleet to traverse the straight-line path.

    Does NOT account for sun collisions or board boundaries; callers should
    pre-filter via `geometry.path_clears_sun`. Returns inf for zero-distance
    plus zero ships (degenerate launch).
    """
    d = dist(src, dst)
    if d == 0.0:
        return 0.0
    return d / speed(ships, max_speed)


def eta_turns(src: Point, dst: Point, ships: int | float,
              max_speed: float = DEFAULT_MAX_SPEED) -> int:
    """Integer-turn ETA (ceil of travel_time). The arrival turn relative to
    the obs from which this is called.
    """
    t = travel_time(src, dst, ships, max_speed)
    return int(math.ceil(t)) if t > 0 else 0
