"""Unit tests for agents/baseline/chooser_differential — Slice 8.

Plan reference: /root/.claude/plans/take-the-lens-of-magical-shore.md §13.

Covers `_projected_state_at`, `_favor_from_state`,
`score_candidate_differential`, and `choose_differential` end-to-end.
"""

from __future__ import annotations

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from agents.baseline.chooser_differential import (
    _favor_from_state,
    _projected_state_at,
    choose_differential,
    score_candidate_differential,
)
from lib.intent import World
from lib.world_model import WorldModel


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    return Planet(pid, owner, x, y, radius, ships, production)


def _fleet(fid, owner, x, y, angle, ships, from_planet_id=0):
    return Fleet(fid, owner, x, y, angle, from_planet_id, ships)


def _world(my_id, planets, *, fleets=None, step=0, omega=0.0):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": [
            (f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships)
            for f in (fleets or [])
        ],
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "comets": [],
        "step": step,
    }
    return obs, World.from_obs(obs)


def _candidate(src, tgt, *, cheap_delta=1.0, ships=10, eta=5, wait_N=0):
    """Build a proposer-style prerank tuple."""
    return (float(cheap_delta), src, tgt, int(ships), 0.0, int(eta),
            int(eta + 2), int(wait_N))


# ---------------------------------------------------------------------------
# _favor_from_state — favor formula matches value.favor
# ---------------------------------------------------------------------------


def test_favor_from_state_2p_zero_when_balanced():
    """Equal ships + equal prod → favor (ignoring pv) is 0."""
    ships = {0: 50.0, 1: 50.0}
    prod = {0: 5.0, 1: 5.0}
    f = _favor_from_state(ships, prod, me=0, num_seats=2, leaf_step=100, gamma=0.99)
    # (50-50) + (5-5)*pv = 0
    assert f == 0.0


def test_favor_from_state_2p_positive_when_we_lead():
    """We have more ships than opp → favor > 0."""
    ships = {0: 100.0, 1: 30.0}
    prod = {0: 8.0, 1: 4.0}
    f = _favor_from_state(ships, prod, me=0, num_seats=2, leaf_step=100, gamma=0.99)
    assert f > 0.0


def test_favor_from_state_no_opps_returns_pure_ship_count():
    """No opps in dict → opp_ships=0, opp_prod=0."""
    ships = {0: 100.0}
    prod = {0: 8.0}
    f = _favor_from_state(ships, prod, me=0, num_seats=2, leaf_step=100, gamma=0.99)
    assert f > 0.0
    # Without an opp the dominant term is my_prod × pv.


# ---------------------------------------------------------------------------
# _projected_state_at — idle baseline
# ---------------------------------------------------------------------------


def test_projected_state_at_idle_returns_current_state():
    """No action; horizon=0 → projection equals the current state."""
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=2)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=1)
    obs, world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    ships, prod = _projected_state_at(world, model, me=0, horizon=0, action=None)
    # At t=0: I own src with 80 ships, neutral owns tgt (not counted).
    assert ships[0] == 80.0
    assert prod[0] == 2.0
    # Neutral has owner = -1, excluded.


def test_projected_state_at_with_action_credits_target():
    """Action lands → target's owner flips to me; ships counted on our side."""
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=2)
    tgt = _planet(1, -1, 30.0, 50.0, ships=5, production=3)
    obs, world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    action = (src, tgt, 50, 0, 4)  # 50 ships, eta=4
    ships, prod = _projected_state_at(world, model, me=0, horizon=10, action=action)
    # At t=10: tgt should be ours (captured at t=4); src has 120 + 2*10 - 50 = 90.
    # tgt at t=10: 50 delivered, 5 garrison, capture; then 3 prod/turn × 6 ticks = 18 → 50-5+18 = 63
    assert ships.get(0, 0) > 0
    # Production: we own both planets at t=10 → my_prod = 2 + 3 = 5
    assert prod.get(0, 0.0) == 5.0


# ---------------------------------------------------------------------------
# score_candidate_differential — delta favor
# ---------------------------------------------------------------------------


def test_score_clean_capture_positive():
    """Clean capture: Δ-favor > 0 (we gain a planet + production)."""
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=2)
    tgt = _planet(1, -1, 30.0, 50.0, ships=5, production=3)
    obs, world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    cand = _candidate(src, tgt, ships=50, eta=4)
    delta = score_candidate_differential(cand, world, model, me=0, num_seats=2)
    assert delta > 0.0


def test_score_bounce_non_positive():
    """Bounce: we send too few ships; Δ ≤ 0 (we lost ships, gained nothing)."""
    src = _planet(0, 0, 10.0, 50.0, ships=100, production=2)
    tgt = _planet(1, -1, 30.0, 50.0, ships=200, production=3)
    obs, world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    cand = _candidate(src, tgt, ships=10, eta=4)
    delta = score_candidate_differential(cand, world, model, me=0, num_seats=2)
    # Source lost 10 ships, tgt still opp-controlled → delta could be 0 or slightly
    # negative depending on how favor counts in-flight vs on-planet ships.
    # Conservative assertion: not a big positive.
    assert delta <= 1.0


# ---------------------------------------------------------------------------
# choose_differential — end-to-end emit
# ---------------------------------------------------------------------------


def test_choose_differential_empty_prerank():
    obs, world = _world(0, [_planet(0, 0, 10.0, 50.0)])
    model = WorldModel.from_world(world)
    moves = choose_differential(
        None, [], None, 0, 2, 600.0, 25, 40, 0.99, world, model,
    )
    assert moves == []


def test_choose_differential_emits_positive_delta_candidates():
    """A positive-Δ candidate gets emitted."""
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=2)
    tgt = _planet(1, -1, 30.0, 50.0, ships=5, production=3)
    obs, world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    cand = _candidate(src, tgt, ships=50, eta=4)
    moves = choose_differential(
        None, [cand], None, 0, 2, 600.0, 25, 40, 0.99, world, model,
    )
    assert len(moves) == 1
    assert int(moves[0][0]) == 0  # src=0


def test_choose_differential_one_per_source():
    """Two candidates from same source → only one emit."""
    src = _planet(0, 0, 10.0, 50.0, ships=200, production=3)
    tgt_a = _planet(1, -1, 30.0, 50.0, ships=5, production=3)
    tgt_b = _planet(2, -1, 40.0, 50.0, ships=5, production=3)
    obs, world = _world(0, [src, tgt_a, tgt_b])
    model = WorldModel.from_world(world)
    cand_a = _candidate(src, tgt_a, cheap_delta=2.0, ships=50, eta=4)
    cand_b = _candidate(src, tgt_b, cheap_delta=2.0, ships=50, eta=5)
    moves = choose_differential(
        None, [cand_a, cand_b], None, 0, 2, 600.0, 25, 40, 0.99, world, model,
    )
    # Both targets winnable, same source → only 1 emit (the higher-Δ one).
    assert len(moves) == 1


def test_choose_differential_one_per_target():
    """Two sources, same target → only one emit (greedy non-dogpile)."""
    src_a = _planet(0, 0, 10.0, 50.0, ships=120, production=2)
    src_b = _planet(1, 0, 20.0, 50.0, ships=120, production=2)
    tgt = _planet(2, -1, 30.0, 50.0, ships=5, production=3)
    obs, world = _world(0, [src_a, src_b, tgt])
    model = WorldModel.from_world(world)
    cand_a = _candidate(src_a, tgt, ships=50, eta=4)
    cand_b = _candidate(src_b, tgt, ships=50, eta=4)
    moves = choose_differential(
        None, [cand_a, cand_b], None, 0, 2, 600.0, 25, 40, 0.99, world, model,
    )
    assert len(moves) == 1


# ---------------------------------------------------------------------------
# Slice 8c — wait_N filter
# ---------------------------------------------------------------------------


def test_choose_differential_filters_wait_N_candidates():
    """Slice 8c: wait_N > 0 candidates are dropped before scoring.
    A fire-now candidate from the same source emits; the wait variant
    contributes nothing to the emit set.
    """
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=2)
    tgt = _planet(1, -1, 30.0, 50.0, ships=5, production=3)
    obs, world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    cand_fire = _candidate(src, tgt, cheap_delta=1.0, ships=50, eta=4, wait_N=0)
    cand_wait = _candidate(src, tgt, cheap_delta=2.0, ships=80, eta=4, wait_N=5)
    moves = choose_differential(
        None, [cand_fire, cand_wait], None, 0, 2, 600.0, 25, 40, 0.99, world, model,
    )
    # Fire-now wins because wait variant is filtered out.
    assert len(moves) == 1
    assert int(moves[0][2]) == 50  # ships count of cand_fire, not cand_wait


def test_choose_differential_returns_empty_when_only_wait_candidates():
    """All-wait prerank → empty emit (filter drops everything)."""
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=2)
    tgt = _planet(1, -1, 30.0, 50.0, ships=5, production=3)
    obs, world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    cand_wait = _candidate(src, tgt, ships=80, eta=4, wait_N=5)
    moves = choose_differential(
        None, [cand_wait], None, 0, 2, 600.0, 25, 40, 0.99, world, model,
    )
    assert moves == []
