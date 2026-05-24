"""Polar-angle helpers for counterclockwise target ordering.

Used by `agents/momentum_strike` to order expansion candidates: from each
owned planet, the "next neutral counterclockwise" around the sun comes
first. This gives a globally consistent sweep direction across all
sources — instead of each source spinning independently.

All angles are normalized to `[0, 2pi)`. The reference centre defaults
to the board centre (50, 50), the sun's position.
"""

from __future__ import annotations

import math

from lib.geometry import CENTER

_TWO_PI = 2.0 * math.pi


def polar_angle_about(point, centre=(CENTER, CENTER)) -> float:
    """Polar angle of `point` around `centre`, in `[0, 2pi)`."""
    ang = math.atan2(point[1] - centre[1], point[0] - centre[0])
    return ang % _TWO_PI


def ccw_delta(src_theta: float, target_theta: float) -> float:
    """Counterclockwise angular distance from `src_theta` to `target_theta`,
    in `[0, 2pi)`. Useful as a sort key.
    """
    return (target_theta - src_theta) % _TWO_PI


def sort_ccw_from_source(source, candidates, centre=(CENTER, CENTER)) -> list:
    """Sort `candidates` (Planet-likes with `.x`, `.y`) by counterclockwise
    polar-angle distance from `source`'s own angle around `centre`.

    Index 0 is "the next candidate counterclockwise from source around
    the centre." Candidates at the exact same angle as source come last
    (distance 0 means the same direction, but we want the NEXT one, so
    we map exact-equal to 2pi via the `% _TWO_PI` semantics — except 0
    stays 0; in practice a source and target at the exact same polar
    angle is degenerate and we let the natural sort fall through).
    """
    src_theta = polar_angle_about((source.x, source.y), centre)
    return sorted(
        candidates,
        key=lambda c: ccw_delta(src_theta,
                                polar_angle_about((c.x, c.y), centre)),
    )
