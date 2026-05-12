"""Tests for lib/combat.resolve_arrivals — README §combat rules 1-4.

Hand-built scenarios covering each rule branch:
1. Same-step arrivals grouped by owner; ships summed.
2. Largest attacker fights second-largest; difference survives.
3a. Survivor's owner == garrison's owner: reinforce.
3b. Survivor's owner != garrison's owner: fights garrison; if survivor
    > garrison ships, ownership flips.
4. Two-way tie among attackers: all destroyed.
"""

from __future__ import annotations

from agent import resolve_arrivals


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_no_arrivals_returns_garrison_unchanged():
    assert resolve_arrivals(0, 10.0, []) == (0, 10.0)


def test_neutral_unchanged_when_no_arrivals():
    assert resolve_arrivals(-1, 5.0, []) == (-1, 5.0)


def test_zero_ship_arrivals_ignored():
    """An arrival with ships=0 is skipped (rule: ships <= 0 filtered)."""
    assert resolve_arrivals(0, 10.0, [(1, 0)]) == (0, 10.0)


# ---------------------------------------------------------------------------
# Rule 1: same-owner arrivals sum
# ---------------------------------------------------------------------------


def test_two_same_owner_arrivals_sum_first():
    """P1 sends 5 + 7 ships — combat treats this as a single 12-ship attack."""
    # Garrison owner=0 (us) with 10 ships; both P1 arrivals sum to 12 > 10 → flip.
    new_owner, new_ships = resolve_arrivals(0, 10.0, [(1, 5), (1, 7)])
    assert new_owner == 1
    assert new_ships == 2.0


# ---------------------------------------------------------------------------
# Rule 2: largest vs second-largest survives the difference
# ---------------------------------------------------------------------------


def test_two_attackers_difference_survives():
    """P1=10, P2=4 → P1 survives with 6 ships, then fights garrison=0 (neutral)."""
    new_owner, new_ships = resolve_arrivals(-1, 0.0, [(1, 10), (2, 4)])
    assert new_owner == 1
    assert new_ships == 6.0


# ---------------------------------------------------------------------------
# Rule 3a: same-owner survivor reinforces garrison
# ---------------------------------------------------------------------------


def test_friendly_arrival_reinforces_garrison():
    """Our own fleet arriving at our owned planet adds ships."""
    new_owner, new_ships = resolve_arrivals(0, 20.0, [(0, 7)])
    assert new_owner == 0
    assert new_ships == 27.0


# ---------------------------------------------------------------------------
# Rule 3b: enemy survivor vs garrison; flip if survivor wins
# ---------------------------------------------------------------------------


def test_enemy_survives_takes_planet_when_outnumbers_garrison():
    new_owner, new_ships = resolve_arrivals(0, 5.0, [(1, 12)])
    assert new_owner == 1
    assert new_ships == 7.0


def test_enemy_survives_loses_to_strong_garrison():
    new_owner, new_ships = resolve_arrivals(0, 20.0, [(1, 5)])
    assert new_owner == 0
    assert new_ships == 15.0


def test_enemy_exactly_equals_garrison_garrison_holds():
    """survivor == garrison → garrison_ships goes to 0 but owner stays."""
    new_owner, new_ships = resolve_arrivals(0, 5.0, [(1, 5)])
    assert new_owner == 0
    assert new_ships == 0.0


# ---------------------------------------------------------------------------
# Rule 4: two-way tie among attackers destroys both
# ---------------------------------------------------------------------------


def test_two_way_attacker_tie_destroys_both():
    """P1=10 vs P2=10 → both destroyed; garrison untouched."""
    new_owner, new_ships = resolve_arrivals(0, 5.0, [(1, 10), (2, 10)])
    assert new_owner == 0
    assert new_ships == 5.0


def test_two_way_attacker_tie_at_neutral_keeps_neutral():
    new_owner, new_ships = resolve_arrivals(-1, 0.0, [(1, 10), (2, 10)])
    assert new_owner == -1
    assert new_ships == 0.0
