"""Unit tests for lib/geo/rotation.py."""

from __future__ import annotations

import math

from lib.geo.rotation import (
    drift_window,
    my_cluster_centroid,
    rotation_alignment,
)
from lib.geometry import CENTER


def _planet(pid, x, y, radius=2.0, production=2):
    """Build a planet tuple [id, owner, x, y, radius, ships, production]."""
    return [pid, -1, float(x), float(y), float(radius), 10, production]


def test_static_planet_returns_zero():
    """Planets at orbital_radius + planet_radius >= ROTATION_RADIUS_LIMIT
    (50.0) don't rotate; alignment = 0. Use (95, 50): orb_r=45, +r=2.5,
    sum=47.5 → orbiting. Use (98, 50): orb_r=48, +r=2.5, sum=50.5 → static."""
    static = _planet(0, 98.0, 50.0, radius=2.5)
    score = rotation_alignment(static, omega=0.05, anchor_xy=(50.0, 50.0))
    assert score == 0.0


def test_rotation_toward_anchor_positive():
    """Planet at (75, 50) orbiting CCW (omega>0) starts moving DOWN
    (toward y=70) — which moves AWAY from anchor (75, 30) at first.
    Use anchor (50, 75): a CCW-rotating planet at (75, 50) moves toward
    (50, 75) over ~30 turns."""
    # CCW rotation: omega > 0 turns counter-clockwise (math convention).
    # At (75, 50), CCW carries the planet through (~70, 65) at small t.
    # Anchor at (50, 75) (above-left) gets closer as planet moves.
    target = _planet(0, 75.0, 50.0, radius=2.0)
    anchor = (50.0, 75.0)
    score = rotation_alignment(target, omega=0.05, anchor_xy=anchor, horizon=30)
    assert score > 0.0, f"expected positive alignment, got {score}"


def test_rotation_away_from_anchor_negative():
    """Same planet, opposite-side anchor: motion takes it away → negative."""
    target = _planet(0, 75.0, 50.0, radius=2.0)
    anchor = (75.0, 25.0)  # below the planet
    # CCW rotation from (75, 50) heads UP first, away from (75, 25).
    score = rotation_alignment(target, omega=0.05, anchor_xy=anchor, horizon=30)
    assert score < 0.0, f"expected negative alignment, got {score}"


def test_alignment_magnitude_scales_with_omega():
    """Double omega → larger angular displacement → larger |alignment|."""
    target = _planet(0, 75.0, 50.0, radius=2.0)
    anchor = (50.0, 75.0)
    slow = rotation_alignment(target, omega=0.025, anchor_xy=anchor, horizon=30)
    fast = rotation_alignment(target, omega=0.05, anchor_xy=anchor, horizon=30)
    assert abs(fast) > abs(slow), \
        f"fast |{fast:.3f}| not > slow |{slow:.3f}|"


def test_drift_window_for_approaching_planet():
    """A planet currently moving toward anchor returns a positive
    drift_window (turns until closest approach)."""
    target = _planet(0, 75.0, 50.0, radius=2.0)
    # Place anchor where the planet is heading on CCW rotation.
    # CCW from (75, 50): θ_now = 0, after t turns θ = omega*t.
    # Planet position: (50 + 25*cos(θ), 50 + 25*sin(θ)).
    # At t = π/(2*omega), the planet is at (50, 75) — closest to (50, 75).
    omega = 0.05
    expected_t = round(math.pi / (2 * omega))
    anchor = (50.0, 75.0)
    t = drift_window(target, omega, anchor, max_horizon=100)
    assert abs(t - expected_t) <= 2, \
        f"expected ~{expected_t} turns, got {t}"


def test_drift_window_static_returns_minus_one():
    static = _planet(0, 98.0, 50.0, radius=2.5)
    assert drift_window(static, omega=0.05, anchor_xy=(50.0, 50.0)) == -1


class _MockPlanet:
    def __init__(self, x, y, production):
        self.x = x
        self.y = y
        self.production = production


def test_cluster_centroid_production_weighted():
    """Centroid is pulled toward higher-production planets."""
    planets = [_MockPlanet(10, 50, production=5),
               _MockPlanet(50, 50, production=1)]
    cx, cy = my_cluster_centroid(planets)
    # Unweighted mean would be (30, 50); weighted should be closer to (10, 50).
    assert cx < 30.0, f"weighted centroid should pull left of unweighted mean; got cx={cx}"
    assert cy == 50.0


def test_cluster_centroid_empty_returns_center():
    cx, cy = my_cluster_centroid([])
    assert (cx, cy) == (CENTER, CENTER)
