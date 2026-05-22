"""Unit tests for `agents/coord/_endgame.py` + the bundle endgame bonus.

Tests cover:
- Predicate primitives (`prod_advantage`, `remaining_turns`, `opp_pool`,
  `winning_margin`).
- Per-bundle ΔW (`bundle_delta_w_attack`, `bundle_delta_w_defend`).
- Bundle-level injection (`_bundle_endgame_bonus`) with kind dispatch,
  env-var gating, model=None short-circuit, and 4P attribution.

Mirrors world-construction patterns from `test_coord_bundle_enum.py`.
"""
from __future__ import annotations

import os

import pytest

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from agents.coord._endgame import (
    EPISODE_STEPS,
    bundle_delta_w_attack,
    bundle_delta_w_defend,
    is_winning_state,
    opp_pool,
    prod_advantage,
    remaining_turns,
    winning_margin,
)
from agents.coord.main import (
    Bundle,
    BundleKind,
    Leg,
    _bundle_endgame_bonus,
    _largest_threat_owner,
    _strongest_opp,
)
from lib.intent import World
from lib.world_model import WorldModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _planet(pid, owner, x=10.0, y=50.0, *, ships=10, production=2, radius=1.5):
    return Planet(pid, owner, x, y, radius, ships, production)


def _world(my_id, planets, *, step=0, omega=0.0, fleets=None):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": list(fleets or []),
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "step": step,
    }
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# Predicate primitives
# ---------------------------------------------------------------------------

def test_remaining_turns_at_step_zero_and_end():
    world = _world(0, [_planet(0, 0)], step=0)
    assert remaining_turns(world) == EPISODE_STEPS
    world500 = _world(0, [_planet(0, 0)], step=EPISODE_STEPS)
    assert remaining_turns(world500) == 0
    world_past = _world(0, [_planet(0, 0)], step=EPISODE_STEPS + 50)
    assert remaining_turns(world_past) == 0  # clamped non-negative


def test_prod_advantage_sums_my_minus_opp():
    p_me1 = _planet(0, 0, production=3)
    p_me2 = _planet(1, 0, production=2)
    p_opp = _planet(2, 1, production=4)
    p_neutral = _planet(3, -1, production=5)
    world = _world(0, [p_me1, p_me2, p_opp, p_neutral])
    # My prod = 3+2 = 5; opp prod = 4; neutral does not contribute.
    assert prod_advantage(world, my_id=0, opp_id=1) == 5 - 4


def test_opp_pool_includes_planets_and_inflight_and_future():
    p_me = _planet(0, 0, ships=20, production=1)
    p_opp = _planet(1, 1, ships=15, production=3)
    # In-flight enemy fleet (idx 6 = ships per build_arrival_ledger).
    fleet = [0, 1, 0, 0, 0, 0, 7]  # owner=1, ships=7
    world = _world(0, [p_me, p_opp], step=10, fleets=[fleet])
    rem = remaining_turns(world)
    expected = 15 + 7 + 3 * rem
    assert opp_pool(world, opp_id=1) == expected


def test_winning_margin_signed_and_matches_is_winning_state():
    # Configure: me strongly ahead → margin positive.
    p_me1 = _planet(0, 0, ships=50, production=10)
    p_me2 = _planet(1, 0, ships=50, production=10)
    p_opp = _planet(2, 1, ships=1, production=1)
    world = _world(0, [p_me1, p_me2, p_opp], step=0)
    m = winning_margin(world, my_id=0, opp_id=1)
    assert m > 0
    assert is_winning_state(world, my_id=0, opp_id=1) is True
    # Configure: opp strongly ahead → margin negative.
    p_me = _planet(0, 0, ships=5, production=1)
    p_opp1 = _planet(1, 1, ships=50, production=10)
    p_opp2 = _planet(2, 1, ships=50, production=10)
    world_bad = _world(0, [p_me, p_opp1, p_opp2], step=0)
    m_bad = winning_margin(world_bad, my_id=0, opp_id=1)
    assert m_bad < 0
    assert is_winning_state(world_bad, my_id=0, opp_id=1) is False


# ---------------------------------------------------------------------------
# Per-bundle ΔW
# ---------------------------------------------------------------------------

def test_bundle_delta_w_attack_neutral_target():
    # Capture of a neutral: ΔW = prod × rem.
    tgt = _planet(5, -1, ships=3, production=4)
    rem = 100
    dw = bundle_delta_w_attack(tgt, my_id=0, opp_id=1, rem=rem)
    assert dw == 4 * 100


def test_bundle_delta_w_attack_opp_target():
    # Capture of opp's planet: ΔW = 3·prod·rem + ships.
    tgt = _planet(5, 1, ships=12, production=3)
    rem = 200
    dw = bundle_delta_w_attack(tgt, my_id=0, opp_id=1, rem=rem)
    assert dw == 3 * 3 * 200 + 12


def test_bundle_delta_w_attack_own_target_zero():
    # ATTACK on own target should be impossible upstream, but the
    # formula must return 0 defensively.
    tgt = _planet(5, 0, ships=12, production=3)
    rem = 200
    dw = bundle_delta_w_attack(tgt, my_id=0, opp_id=1, rem=rem)
    assert dw == 0


def test_bundle_delta_w_attack_other_opp_in_4p():
    # 4P edge: opp_id attributed to seat 1, but target is owned by seat 2.
    # Formula falls back to neutral-style ΔW = prod·rem (we gain prod;
    # opp seat 1's pool unchanged).
    tgt = _planet(5, 2, ships=12, production=3)
    rem = 200
    dw = bundle_delta_w_attack(tgt, my_id=0, opp_id=1, rem=rem)
    assert dw == 3 * 200


def test_bundle_delta_w_defend_avoided_loss():
    # DEFEND of own planet against opp threat: ΔW = +3·prod·rem.
    tgt = _planet(5, 0, ships=10, production=4)
    rem = 150
    dw = bundle_delta_w_defend(tgt, my_id=0, opp_threat=1, rem=rem)
    assert dw == 3 * 4 * 150


def test_bundle_delta_w_defend_returns_zero_for_non_own_target():
    # DEFEND helper called on a non-own target (caller bug) returns 0.
    tgt = _planet(5, 1, ships=10, production=4)
    rem = 150
    dw = bundle_delta_w_defend(tgt, my_id=0, opp_threat=1, rem=rem)
    assert dw == 0


# ---------------------------------------------------------------------------
# Bundle endgame bonus — kind dispatch, env-var gating, 4P attribution
# ---------------------------------------------------------------------------

def _attack_bundle(target_id, src_id=0, ships=10):
    return Bundle(
        target_id=target_id,
        arrival_step=5,
        legs=(Leg(src_id=src_id, ships=ships, angle=0.0, wait_N=0, eta=5),),
        kind=BundleKind.ATTACK,
    )


def _defend_bundle(target_id, src_id=1, ships=10):
    return Bundle(
        target_id=target_id,
        arrival_step=3,
        legs=(Leg(src_id=src_id, ships=ships, angle=0.0, wait_N=0, eta=3),),
        kind=BundleKind.DEFEND,
    )


def test_bundle_endgame_bonus_returns_zero_when_rem_is_zero():
    src = _planet(0, 0)
    tgt = _planet(1, 1, production=3)
    world = _world(0, [src, tgt], step=EPISODE_STEPS)
    model = WorldModel.from_world(world)
    bundle = _attack_bundle(target_id=1, src_id=0)
    assert _bundle_endgame_bonus(bundle, world, model, me=0, num_seats=2) == 0.0


def test_bundle_endgame_bonus_disabled_via_env(monkeypatch):
    monkeypatch.setenv("COORD_DELTA_W", "0")
    src = _planet(0, 0)
    tgt = _planet(1, 1, production=3)
    world = _world(0, [src, tgt], step=0)
    model = WorldModel.from_world(world)
    bundle = _attack_bundle(target_id=1, src_id=0)
    assert _bundle_endgame_bonus(bundle, world, model, me=0, num_seats=2) == 0.0


def test_bundle_endgame_bonus_model_none_short_circuits(monkeypatch):
    monkeypatch.setenv("COORD_DELTA_W", "1")
    src = _planet(0, 0)
    tgt = _planet(1, 1, production=3)
    world = _world(0, [src, tgt], step=0)
    bundle = _attack_bundle(target_id=1, src_id=0)
    assert _bundle_endgame_bonus(
        bundle, world, model=None, me=0, num_seats=2,
    ) == 0.0


def test_bundle_endgame_bonus_attack_opp_target_2p(monkeypatch):
    monkeypatch.setenv("COORD_DELTA_W", "1")
    monkeypatch.setenv("COORD_LAMBDA_W", "1.0")  # easy arithmetic
    src = _planet(0, 0, ships=50, production=2)
    tgt = _planet(1, 1, ships=8, production=3)
    world = _world(0, [src, tgt], step=0)
    model = WorldModel.from_world(world)
    bundle = _attack_bundle(target_id=1, src_id=0)
    rem = EPISODE_STEPS  # step=0
    expected = 1.0 * (3 * 3 * rem + 8)
    assert _bundle_endgame_bonus(
        bundle, world, model, me=0, num_seats=2,
    ) == pytest.approx(expected)


def test_bundle_endgame_bonus_attack_neutral_target(monkeypatch):
    monkeypatch.setenv("COORD_DELTA_W", "1")
    monkeypatch.setenv("COORD_LAMBDA_W", "1.0")
    src = _planet(0, 0, ships=50, production=2)
    # Neutral target (owner=-1) — formula returns prod·rem regardless
    # of attributed opp, AND _strongest_opp picks an opp (irrelevant
    # for neutral ΔW value).
    other_opp = _planet(2, 1, ships=10, production=4)
    tgt_neutral = _planet(1, -1, ships=3, production=2)
    world = _world(0, [src, other_opp, tgt_neutral], step=0)
    model = WorldModel.from_world(world)
    bundle = _attack_bundle(target_id=1, src_id=0)
    rem = EPISODE_STEPS
    expected = 1.0 * (2 * rem)  # prod=2
    assert _bundle_endgame_bonus(
        bundle, world, model, me=0, num_seats=2,
    ) == pytest.approx(expected)


def test_bundle_endgame_bonus_defend_with_threat(monkeypatch):
    monkeypatch.setenv("COORD_DELTA_W", "1")
    monkeypatch.setenv("COORD_LAMBDA_W", "1.0")
    own = _planet(0, 0, x=10.0, y=50.0, ships=5, production=4)
    peer = _planet(1, 0, x=10.0, y=60.0, ships=50, production=2)
    enemy = _planet(2, 1, x=30.0, y=50.0, ships=40, production=3)
    # Build an inbound enemy fleet targeting `own` so model.ledger has
    # a (eta, owner=1, ships=20) entry.
    # Fleet tuple shape: [id, owner, src_x, src_y, dst_x, dst_y, ships, ...]
    # We just need owner=1 and a target that resolves to planet 0.
    # Use the same parser path as the production env — list of [id, owner,
    # x, y, vx, vy, ships, age, target_id, ...]. To avoid the trajectory
    # parsing details, point the in-flight fleet straight at own (which
    # `build_arrival_ledger` handles by ray-cast); easier: build the
    # ledger directly on the model after construction.
    world = _world(0, [own, peer, enemy], step=0)
    model = WorldModel.from_world(world)
    # Inject a synthetic ledger entry: (eta=5, owner=1, ships=20).
    model.ledger[0] = [(5, 1, 20)]
    bundle = _defend_bundle(target_id=0, src_id=1)
    rem = EPISODE_STEPS
    expected = 1.0 * (3 * 4 * rem)  # prod=4
    assert _bundle_endgame_bonus(
        bundle, world, model, me=0, num_seats=2,
    ) == pytest.approx(expected)


def test_bundle_endgame_bonus_defend_no_threat_returns_zero(monkeypatch):
    monkeypatch.setenv("COORD_DELTA_W", "1")
    monkeypatch.setenv("COORD_LAMBDA_W", "1.0")
    own = _planet(0, 0, ships=5, production=4)
    peer = _planet(1, 0, x=10.0, y=60.0, ships=50, production=2)
    world = _world(0, [own, peer], step=0)
    model = WorldModel.from_world(world)
    # No ledger entries → no threat owner identifiable.
    assert model.ledger.get(0, []) == []
    bundle = _defend_bundle(target_id=0, src_id=1)
    assert _bundle_endgame_bonus(
        bundle, world, model, me=0, num_seats=2,
    ) == 0.0


def test_bundle_endgame_bonus_attack_4p_attributes_to_target_owner(monkeypatch):
    monkeypatch.setenv("COORD_DELTA_W", "1")
    monkeypatch.setenv("COORD_LAMBDA_W", "1.0")
    src = _planet(0, 0, ships=50, production=2)
    # 4P: target owned by seat 2 (not seat 1). Bonus should attribute
    # to seat 2 (target's current owner) per (c1) design.
    tgt_seat2 = _planet(1, 2, ships=8, production=3)
    # Add some other-seat presence so num_seats=4 makes sense.
    seat1_planet = _planet(2, 1, ships=10, production=3)
    seat3_planet = _planet(3, 3, ships=10, production=3)
    world = _world(0, [src, tgt_seat2, seat1_planet, seat3_planet], step=0)
    model = WorldModel.from_world(world)
    bundle = _attack_bundle(target_id=1, src_id=0)
    rem = EPISODE_STEPS
    # Expected: opp_id=2 (target.owner). prod=3, ships=8.
    expected = 1.0 * (3 * 3 * rem + 8)
    assert _bundle_endgame_bonus(
        bundle, world, model, me=0, num_seats=4,
    ) == pytest.approx(expected)


def test_largest_threat_owner_picks_max_ships():
    own = _planet(0, 0)
    world = _world(0, [own], step=0)
    model = WorldModel.from_world(world)
    # Manually inject ledger entries — owner=2 has more ships than owner=1.
    model.ledger[0] = [(5, 1, 10), (7, 2, 25), (9, 1, 15)]
    assert _largest_threat_owner(target_id=0, model=model, me=0) == 2


def test_largest_threat_owner_filters_self_and_neutral():
    own = _planet(0, 0)
    world = _world(0, [own], step=0)
    model = WorldModel.from_world(world)
    model.ledger[0] = [(5, 0, 100), (7, -1, 50)]  # self + neutral
    assert _largest_threat_owner(target_id=0, model=model, me=0) is None


def test_strongest_opp_in_4p_returns_max_pool_owner():
    me_planet = _planet(0, 0, production=1)
    weak_opp = _planet(1, 1, ships=1, production=1)
    strong_opp = _planet(2, 2, ships=100, production=10)
    mid_opp = _planet(3, 3, ships=10, production=5)
    world = _world(0, [me_planet, weak_opp, strong_opp, mid_opp])
    assert _strongest_opp(world, me=0, num_seats=4) == 2
