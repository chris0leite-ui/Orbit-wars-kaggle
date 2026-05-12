"""Tests for the fleet-speed formula in agent.py.

Spec from data/README.md::Fleet Speed:

    speed = 1.0 + (max_speed - 1.0) * (log(ships) / log(1000)) ^ 1.5

with max_speed defaulting to 6.0 (Configuration table).
"""

from __future__ import annotations

import pytest

from agent import fleet_speed


def test_speed_one_ship_is_exactly_one_unit_per_turn():
    assert fleet_speed(1) == pytest.approx(1.0)


def test_speed_thousand_ships_hits_max_speed():
    assert fleet_speed(1000) == pytest.approx(6.0)


def test_speed_zero_or_negative_is_floor_one_unit():
    assert fleet_speed(0) == 1.0
    assert fleet_speed(-5) == 1.0


def test_speed_above_thousand_clamps_to_max():
    assert fleet_speed(5000) == 6.0
    assert fleet_speed(10**6) == 6.0


def test_speed_monotonic_in_ships_within_range():
    samples = [fleet_speed(s) for s in (1, 2, 10, 100, 500, 999)]
    assert samples == sorted(samples)


def test_speed_500_ships_close_to_five_per_readme():
    assert 4.7 < fleet_speed(500) < 5.3


def test_speed_respects_custom_max_speed():
    assert fleet_speed(1000, max_speed=10.0) == pytest.approx(10.0)
    assert fleet_speed(1, max_speed=10.0) == 1.0
