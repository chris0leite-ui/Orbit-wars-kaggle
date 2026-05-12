"""Tests for the snipe-priority modifiers added in v3.4
(2026-05-11 games-analysis upgrade).

Two modifiers are stacked multiplicatively on the cost-aware ROI score:
1. NEUTRAL_BONUS / COMET_BONUS — uncontested target premium.
2. LEADER_MULTIPLIER — 4P spoiler: attack the leader when ranked 3rd+.

Audit reference: audit/2026-05-11-v3-snipe-games-analysis.md §P1/P2.
"""

from __future__ import annotations

from types import SimpleNamespace

from lib.intent import World
from lib.missions import snipe as snipe_module
from lib.missions.snipe import (
    AIRTIME_PENALTY_WEIGHT,
    COMET_BONUS,
    ENDGAME_NEUTRAL_BONUS,
    ENDGAME_STEP,
    LEADER_MULTIPLIER,
    NEUTRAL_BONUS,
    _leader_pid,
    _player_totals,
    propose_snipe_missions,
)
from lib.world_model import WorldModel


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    return SimpleNamespace(
        id=pid, owner=owner, x=x, y=y, radius=radius,
        ships=ships, production=production,
    )


def _world(my_id, planets, *, step=10, fleets=None, comet_ids=()):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": list(fleets) if fleets else [],
        "angular_velocity": 0.0,
        "comet_planet_ids": list(comet_ids),
        "step": step,
    }
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# NEUTRAL_BONUS / COMET_BONUS
# ---------------------------------------------------------------------------


def test_neutral_target_gets_bonus_over_identical_enemy_target():
    """Two identical targets (same production, ships, distance) — one
    neutral, one enemy. The neutral should outrank by exactly NEUTRAL_BONUS.

    Live data: 78.6% of comet-steps go unclaimed, only 4.9% to us. Score
    function under-prices unclaimed targets even though they're cheaper
    to take (no garrison growth, no opponent competition)."""
    src = _planet(0, owner=0, x=10.0, y=50.0, ships=100)
    neutral = _planet(1, owner=-1, x=70.0, y=50.0, ships=10, production=2)
    enemy = _planet(2, owner=1, x=70.0, y=50.0, ships=10, production=2)
    world = _world(my_id=0, planets=[src, neutral, enemy])
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    by_target = {m.target_id: m for m in missions}
    assert 1 in by_target and 2 in by_target
    # Neutral score should be NEUTRAL_BONUS × enemy score.
    assert by_target[1].score == NEUTRAL_BONUS * by_target[2].score


def test_comet_target_gets_comet_bonus_not_neutral_bonus():
    """Comets and non-comet neutrals both get a bonus, but the comet
    bonus is its own constant (calibrated separately to reflect
    finite lifetime trade-off)."""
    src = _planet(0, owner=0, x=10.0, y=50.0, ships=100)
    comet = _planet(1, owner=-1, x=70.0, y=50.0, ships=5, production=1)
    obs = {
        "player": 0,
        "planets": [
            (src.id, src.owner, src.x, src.y, src.radius, src.ships, src.production),
            (comet.id, comet.owner, comet.x, comet.y, comet.radius, comet.ships, comet.production),
        ],
        "fleets": [],
        "angular_velocity": 0.0,
        "comet_planet_ids": [1],
        "step": 10,
        "comets": [{
            "planet_ids": [1],
            "paths": [[[70.0, 50.0]] * 80],   # 80 future positions
            "path_index": 0,
        }],
    }
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    assert len(missions) == 1
    # No regular neutral to compare against directly here; instead check
    # the priority is exactly COMET_BONUS by deriving from baseline.
    eta = missions[0].eta
    base_value = comet.production * max(0, 80 - eta)
    base_score = base_value / (
        max(1, comet.ships + 1)
        + (70.0 - 10.0)
        + AIRTIME_PENALTY_WEIGHT * eta
        + 1.0
    )
    assert abs(missions[0].score - COMET_BONUS * base_score) < 1e-6


# ---------------------------------------------------------------------------
# 4P spoiler (LEADER_MULTIPLIER)
# ---------------------------------------------------------------------------


def test_player_totals_sums_planets_and_fleets():
    src = _planet(0, owner=0, x=10.0, y=50.0, ships=30)
    enemy = _planet(1, owner=1, x=70.0, y=50.0, ships=50)
    fleet = [99, 0, 30.0, 50.0, 0.0, 0, 25]  # us, 25 ships in flight
    world = _world(my_id=0, planets=[src, enemy], fleets=[fleet])
    totals = _player_totals(world)
    assert totals[0] == 30 + 25
    assert totals[1] == 50


def test_leader_pid_returns_none_for_2p():
    src = _planet(0, owner=0, x=10.0, y=50.0, ships=10)
    enemy = _planet(1, owner=1, x=70.0, y=50.0, ships=10)
    world = _world(my_id=0, planets=[src, enemy])
    leader, rank = _leader_pid(world)
    assert leader is None and rank is None


def test_leader_pid_identifies_strongest_opponent_in_4p():
    src = _planet(0, owner=0, x=10.0, y=10.0, ships=10)
    weak = _planet(1, owner=1, x=90.0, y=10.0, ships=5)
    strong = _planet(2, owner=2, x=10.0, y=90.0, ships=100)
    mid = _planet(3, owner=3, x=90.0, y=90.0, ships=30)
    world = _world(my_id=0, planets=[src, weak, strong, mid])
    leader, rank = _leader_pid(world)
    assert leader == 2  # player 2 has most ships
    # Ordered: 2(100), 3(30), 0(10), 1(5). We're rank 2.
    assert rank == 2


def test_spoiler_boosts_leader_targets_when_we_rank_3rd():
    """4P, we rank 3rd. Leader's planet at same distance as a peer enemy
    should outscore it by LEADER_MULTIPLIER."""
    # Construct totals: leader=300, p1=200, us=50, p3=10  → we're rank 2 (0-indexed)
    src = _planet(0, owner=0, x=10.0, y=10.0, ships=50)
    p1_planet = _planet(1, owner=1, x=90.0, y=10.0, ships=10, production=2)
    p1_home = _planet(5, owner=1, x=10.0, y=90.0, ships=190, production=2)  # boosts P1 to 200
    leader_target = _planet(2, owner=2, x=90.0, y=20.0, ships=10, production=2)
    leader_home = _planet(3, owner=2, x=20.0, y=90.0, ships=290, production=2)  # boosts P2 to 300
    p3_planet = _planet(4, owner=3, x=30.0, y=90.0, ships=10, production=2)
    world = _world(my_id=0, planets=[src, p1_planet, p1_home, leader_target, leader_home, p3_planet])
    leader, rank = _leader_pid(world)
    assert leader == 2, f"expected leader=2, got {leader} (totals: {_player_totals(world)})"
    assert rank == 2, f"expected our_rank=2, got {rank}"

    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    by_target = {m.target_id: m for m in missions}
    # Compare planet 1 (P1 owned, not leader) vs planet 2 (leader-owned).
    # Same distance from src (both at x=90, y in [10,20]).
    # Planet 2 should be LEADER_MULTIPLIER × baseline; planet 1 plain.
    assert 1 in by_target and 2 in by_target
    # Both have identical ships+production+ETA, near-identical distance.
    # Score ratio leader/non-leader ≈ LEADER_MULTIPLIER.
    ratio = by_target[2].score / by_target[1].score
    assert ratio > 1.3, f"expected leader boost (>~{LEADER_MULTIPLIER}), got ratio={ratio:.3f}"


def test_spoiler_does_not_fire_when_we_are_leader():
    """If we ARE the leader, don't boost — focus on consolidating gains."""
    src = _planet(0, owner=0, x=10.0, y=10.0, ships=500)   # huge garrison
    p1 = _planet(1, owner=1, x=90.0, y=10.0, ships=10, production=2)
    p2 = _planet(2, owner=2, x=90.0, y=20.0, ships=10, production=2)
    p3 = _planet(3, owner=3, x=10.0, y=90.0, ships=10, production=2)
    world = _world(my_id=0, planets=[src, p1, p2, p3])
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    by_target = {m.target_id: m for m in missions}
    # All three enemy targets have equal priority; spoiler must not fire.
    # (Spoiler fires only when our_rank >= 2; we're rank 0 here.)
    # All non-leader baseline scores should be equal-ish (modulo distance).
    s1 = by_target[1].score
    s2 = by_target[2].score
    # No leader bonus → near-identical scores (only distance differs slightly).
    ratio = max(s1, s2) / min(s1, s2)
    assert ratio < 1.1  # within 10% — no big multiplier difference
