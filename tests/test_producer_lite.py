"""Unit tests for lib.producer_lite (the Producer attack-policy port).

Covers the output contract, the safe_drain hold-reserve recurrence, the
max-waves cap, and a clear aggression case. These are NECESSARY (the policy
behaves sanely on synthetic input) but NOT SUFFICIENT for fidelity — the
acceptance gates (vs lite_greedy / baseline / full Producer) decide that.
"""
from __future__ import annotations

import math

from lib.producer_lite import (
    _build_projection,
    _safe_drain,
    producer_lite_policy,
)

# Planet tuple schema: (id, owner, x, y, radius, ships, production).


def _board(planets, fleets=None, omega=0.0, player=0):
    return {
        "planets": planets,
        "fleets": fleets or [],
        "comet_planet_ids": [],
        "angular_velocity": omega,
        "step": 0,
        "player": player,
        "initial_planets": [list(p) for p in planets],
    }


def test_empty_board_returns_no_moves():
    assert producer_lite_policy(_board([])) == []
    assert producer_lite_policy({"planets": None, "player": 0}) == []


def test_output_contract_and_no_over_drain():
    planets = [
        (0, 0, 10.0, 10.0, 1.0, 50, 2),   # mine, strong
        (1, -1, 20.0, 10.0, 1.0, 3, 1),   # weak neutral nearby
        (2, 1, 80.0, 80.0, 1.0, 10, 2),   # enemy, far
    ]
    moves = producer_lite_policy(_board(planets))
    assert moves, "expected at least one launch from the strong planet"
    ships_by_src: dict[int, int] = {}
    my_ships = {int(p[0]): float(p[5]) for p in planets if int(p[1]) == 0}
    for m in moves:
        assert isinstance(m, list) and len(m) == 3
        src_id, angle, ships = m
        assert src_id in my_ships, "source must be one of my planets"
        assert isinstance(angle, float) and math.isfinite(angle)
        assert isinstance(ships, int) and ships >= 1
        ships_by_src[src_id] = ships_by_src.get(src_id, 0) + ships
    # No source launches more than it currently holds (budget invariant).
    for src_id, total in ships_by_src.items():
        assert total <= my_ships[src_id] + 1e-6


def test_max_waves_cap():
    # Many strong sources + many weak targets → more than 6 viable waves,
    # but the policy must fire at most max_waves_per_turn (6, 2P).
    planets = []
    pid = 0
    for k in range(8):  # 8 of my strong planets
        planets.append((pid, 0, 5.0 + k * 3.0, 5.0, 1.0, 40, 2))
        pid += 1
    for k in range(8):  # 8 weak neutral targets
        planets.append((pid, -1, 5.0 + k * 3.0, 50.0, 1.0, 2, 1))
        pid += 1
    moves = producer_lite_policy(_board(planets))
    assert len(moves) <= 6


def test_safe_drain_held_vs_doomed():
    # Held: owner stays me, ships dip to 20 → drain = min(20, source_ships).
    H = 5
    owner = {0: [0] * (H + 1)}
    ships = {0: [30.0, 28.0, 20.0, 22.0, 25.0, 30.0]}
    assert _safe_drain(0, owner, ships, H, me=0, source_ships=30.0) == 20.0
    # Doomed: never held within horizon → send everything (cap = source_ships).
    owner_d = {0: [0, 1, 1, 1, 1, 1]}  # flips immediately
    ships_d = {0: [30.0, 5.0, 8.0, 11.0, 14.0, 17.0]}
    assert _safe_drain(0, owner_d, ships_d, H, me=0, source_ships=30.0) == 30.0


def test_projection_resolves_production_and_neutral_constant():
    planets = [
        (0, 0, 10.0, 10.0, 1.0, 10, 2),    # mine, prod 2
        (1, -1, 90.0, 90.0, 1.0, 5, 1),    # neutral (no production)
    ]
    owner, ships, flip = _build_projection(planets, [], omega=0.0, H=4, me=0)
    # My planet accrues production each turn.
    assert ships[0][1] == 12.0 and ships[0][4] == 18.0
    assert owner[0][4] == 0 and flip[0] is None
    # Neutral stays constant (env rule: neutrals don't produce).
    assert ships[1][4] == 5.0 and owner[1][4] == -1
