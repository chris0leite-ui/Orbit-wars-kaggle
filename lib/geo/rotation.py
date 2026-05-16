"""Rotation-alignment primitives for geometry-aware mission scoring.

Operationalises PI's directive: prefer captures of planets whose orbital
motion carries them TOWARD our cluster (longer expected hold time once
captured) and de-prefer planets drifting AWAY toward enemy territory.

Static planets (`is_orbiting` False) return 0.0 from `rotation_alignment`
since they don't drift either way — sustainability for static planets is
a function of garrison-and-distance, not rotation.
"""

from __future__ import annotations

import math

from lib.geometry import CENTER
from lib.orbit import is_orbiting, predict_relative


def rotation_alignment(target, omega: float, anchor_xy, horizon: int = 30) -> float:
    """How much the target's orbit carries it toward `anchor_xy` over `horizon` turns.

    Returns a scalar in roughly [-1, +1]:
      +1  : planet ends `horizon` turns ~one orbital radius CLOSER to anchor
       0  : net motion is orthogonal to anchor direction, OR planet is static
      -1  : planet ends ~one orbital radius FARTHER from anchor

    `anchor_xy` is typically our cluster centroid (mean of my planet positions)
    so a positive value means "this orbit brings the planet within easier
    defensive reach of our home base over the next 30 turns."

    Score is normalised by the planet's orbital radius so the magnitude is
    comparable across inner/outer orbits. Caller decides how to weight it
    into the candidate score.

    `target` is the env tuple [id, owner, x, y, radius, ships, production].
    """
    if not is_orbiting(list(target)):
        return 0.0
    px, py = float(target[2]), float(target[3])
    orb_r = math.hypot(px - CENTER, py - CENTER)
    if orb_r <= 1e-6:
        return 0.0
    ax, ay = float(anchor_xy[0]), float(anchor_xy[1])
    dist_now = math.hypot(px - ax, py - ay)
    fx, fy = predict_relative(list(target), omega, horizon)
    dist_future = math.hypot(fx - ax, fy - ay)
    return (dist_now - dist_future) / orb_r


def drift_window(target, omega: float, anchor_xy, max_horizon: int = 100) -> int:
    """Turns until the target is at its CLOSEST approach to `anchor_xy`.

    Returns 0 if the planet is currently at or past the closest approach
    (drifting away). Returns -1 if static or omega == 0. Otherwise scans
    1..max_horizon and returns the turn index that minimises distance.

    Used to decide whether it's worth WAITING for the planet to come to
    us instead of chasing it now. A small positive return value (e.g. 5)
    suggests "wait 5 turns, then fire" may be more efficient than a
    fire-now candidate.
    """
    if not is_orbiting(list(target)) or omega == 0.0:
        return -1
    ax, ay = float(anchor_xy[0]), float(anchor_xy[1])
    px, py = float(target[2]), float(target[3])
    best_dist = math.hypot(px - ax, py - ay)
    best_t = 0
    for t in range(1, max_horizon + 1):
        fx, fy = predict_relative(list(target), omega, t)
        d = math.hypot(fx - ax, fy - ay)
        if d < best_dist:
            best_dist = d
            best_t = t
    return best_t


def my_cluster_centroid(my_planets) -> tuple[float, float]:
    """Production-weighted centroid of my planets.

    Production weighting (not uniform mean) emphasises the planets we'd
    actually USE as launch sources for defense — a prod-5 home pulls the
    centroid more than a prod-1 scavenge planet. Returns CENTER if we
    have no planets.

    `my_planets` is any iterable yielding objects with `.x`, `.y`,
    `.production` attributes (e.g. our `lib.intent.Planet` view).
    """
    total_w = 0.0
    cx = 0.0
    cy = 0.0
    for p in my_planets:
        w = max(1.0, float(p.production))
        cx += float(p.x) * w
        cy += float(p.y) * w
        total_w += w
    if total_w <= 0.0:
        return (CENTER, CENTER)
    return (cx / total_w, cy / total_w)
