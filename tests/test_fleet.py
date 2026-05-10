"""Tests for lib/fleet.py — speed formula + ETA.

Specs taken from `data/README.md::Fleet Speed`:

    speed = 1.0 + (max_speed - 1.0) * (log(ships) / log(1000)) ^ 1.5

with `max_speed` defaulting to 6.0 (Configuration table).
"""

from __future__ import annotations

import math

import pytest

from lib import fleet as F


def test_speed_one_ship_is_exactly_one_unit_per_turn():
    assert F.speed(1) == pytest.approx(1.0)


def test_speed_thousand_ships_hits_max_speed():
    assert F.speed(1000) == pytest.approx(6.0)


def test_speed_zero_or_negative_is_floor_one_unit():
    assert F.speed(0) == 1.0
    assert F.speed(-5) == 1.0


def test_speed_above_thousand_clamps_to_max():
    assert F.speed(5000) == 6.0
    assert F.speed(10**6) == 6.0


def test_speed_monotonic_in_ships_within_range():
    samples = [F.speed(s) for s in (1, 2, 10, 100, 500, 999)]
    assert samples == sorted(samples)


def test_speed_500_ships_close_to_five_per_readme():
    # README: "A fleet of ~500 ships moves at ~5"
    assert 4.7 < F.speed(500) < 5.3


def test_speed_respects_custom_max_speed():
    assert F.speed(1000, max_speed=10.0) == pytest.approx(10.0)
    assert F.speed(1, max_speed=10.0) == 1.0


def test_travel_time_distance_over_speed():
    # 10 board units, 1-ship fleet → 1 unit/turn → 10 turns.
    assert F.travel_time((0.0, 0.0), (10.0, 0.0), ships=1) == pytest.approx(10.0)


def test_travel_time_zero_distance_returns_zero():
    assert F.travel_time((5.0, 5.0), (5.0, 5.0), ships=10) == 0.0


def test_travel_time_uses_fleet_speed():
    # 60 units, 1000 ships → 6 units/turn → 10 turns.
    assert F.travel_time((0.0, 0.0), (60.0, 0.0), ships=1000) == pytest.approx(10.0)


def test_eta_turns_ceils_partial_turns():
    # 11 units at 1 unit/turn (1 ship) → 11 turns.
    assert F.eta_turns((0.0, 0.0), (11.0, 0.0), ships=1) == 11
    # 10.5 units at 1 unit/turn → 11 turns (ceiling).
    assert F.eta_turns((0.0, 0.0), (10.5, 0.0), ships=1) == 11


def test_eta_turns_zero_distance_is_zero():
    assert F.eta_turns((5.0, 5.0), (5.0, 5.0), ships=10) == 0
