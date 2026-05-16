"""Unit tests for lib/compound.fleet_path_safe.

The PI's explicit symptom: recent agents fly fleets into the sun. Test
that fleet_path_safe drops candidates whose straight-line trajectory
crosses the sun's safety zone.
"""

from __future__ import annotations

import math

from lib.compound import fleet_path_safe
from lib.geometry import CENTER, SUN_RADIUS


class _MockSrc:
    def __init__(self, x, y, radius=2.0):
        self.x = x
        self.y = y
        self.radius = radius


def test_sun_crossing_path_dropped():
    """Fleet aimed from (10, 50) toward (90, 50) — straight through the
    sun at (50, 50). Should be dropped."""
    src = _MockSrc(x=10.0, y=50.0)
    # Aim directly through the center.
    angle = math.atan2(50.0 - 50.0, 90.0 - 10.0)  # = 0.0 rad (east)
    assert fleet_path_safe(src, angle, ships=20, eta=20) is False


def test_safe_perimeter_path_passes():
    """Fleet aimed AROUND the sun — should pass."""
    src = _MockSrc(x=10.0, y=10.0)
    # Aim to (90, 10) — straight along the top, well clear of the sun.
    angle = math.atan2(10.0 - 10.0, 90.0 - 10.0)
    assert fleet_path_safe(src, angle, ships=20, eta=30) is True


def test_oob_arrival_dropped():
    """Fleet aimed off the board should be dropped."""
    src = _MockSrc(x=10.0, y=10.0)
    # Aim northwest, off the board.
    angle = math.atan2(-50.0, -50.0)  # toward (-44, -44) at eta=20
    assert fleet_path_safe(src, angle, ships=20, eta=20) is False


def test_short_eta_safe_path_passes():
    """A short fleet trajectory that doesn't reach the sun zone passes,
    even if the angle would eventually point through the sun. The
    path_clears_sun check uses the actual segment endpoints, so a fleet
    that only travels a short distance (and never gets close to the sun)
    is fine."""
    src = _MockSrc(x=10.0, y=20.0)  # well below the sun row
    angle = 0.0  # due east
    # 1 ship → speed 1, eta=5 → fleet at (~17, 20). Sun at (50, 50).
    # Min distance from (50, 50) to segment (12.1, 20) → (17.1, 20) is
    # > 30. Safe.
    assert fleet_path_safe(src, angle, ships=1, eta=5) is True


def test_tangent_path_outside_safety():
    """A fleet tangent to the sun at distance > SUN_RADIUS + 0.5 passes."""
    src = _MockSrc(x=0.0, y=20.0)
    # Aim east — y stays at 20. Sun center at (50, 50), radius 10.
    # Distance from (50, 50) to segment y=20: 30 units. Well clear.
    angle = 0.0
    assert fleet_path_safe(src, angle, ships=50, eta=20) is True


def test_fleet_speed_scales_path_length():
    """Big fleet (faster) covers more ground at same eta. Path must stay
    on the board and clear the sun."""
    src = _MockSrc(x=5.0, y=20.0)
    angle = 0.0  # east
    # 1000-ship fleet → speed=6, eta=14 → endpoint at (5 + 6*14, 20) =
    # (89, 20). On board, clear of sun. Should pass.
    assert fleet_path_safe(src, angle, ships=1000, eta=14) is True
    # Same but eta=20 → endpoint at (125, 20) → OOB. Should drop.
    assert fleet_path_safe(src, angle, ships=1000, eta=20) is False
