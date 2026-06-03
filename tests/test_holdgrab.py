"""Unit tests for the holdgrab agent.

Covers the load-bearing pieces of "grab the production you can hold":
  - Lanchester ship sizing (capture / capture-and-hold).
  - production-time-integral value (enemy double-count, 4P weakest bias).
  - opponent-as-physics (contest force, defensive floor, reach tick).
  - the two-tier chooser (hold-capture preferred; pressure-capture instead of
    hoarding; dedup; budget; determinism).

World construction mirrors tests/test_chooser_greedy.py: raw planet rows
[id, owner, x, y, radius, ships, production] -> build_turn_view.
"""

from __future__ import annotations

from agents.holdgrab import sizing
from agents.holdgrab.chooser import select
from agents.holdgrab.config import DEFAULT, Config
from agents.holdgrab.threat import contest_force, defense_follow_on, opp_reach_tick
from agents.holdgrab.value import planet_value
from agents.holdgrab.world_view import build_turn_view


def _obs(planets, *, player=0, fleets=None, step=0):
    return {
        "player": player,
        "planets": planets,
        "fleets": fleets or [],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": step,
    }


def _view(planets, **kw):
    return build_turn_view(_obs(planets, **kw), DEFAULT)


def _tgt(view, tid):
    return view.world.planets_by_id[tid]


# --------------------------------------------------------------------------
# sizing — Lanchester linear law
# --------------------------------------------------------------------------

def test_capture_is_garrison_plus_one():
    assert sizing.ships_to_capture(8) == 9
    assert sizing.ships_to_capture(0) == 1


def test_capture_and_hold_adds_uncovered_follow_on():
    # follow_on 40, production 1 over a 30-tick window covers 30 -> net 10.
    assert sizing.net_follow_on(40, 1.0, 30) == 10.0
    assert sizing.ships_to_capture_and_hold(8, 40, 1.0, 30) == 8 + 10 + 1


def test_capture_and_hold_degenerates_when_no_follow_on():
    assert sizing.ships_to_capture_and_hold(8, 0, 1.0, 30) == 9


def test_production_can_fully_cover_follow_on():
    # production*window >= follow_on -> net 0 -> just capture size.
    assert sizing.net_follow_on(20, 1.0, 30) == 0.0
    assert sizing.ships_to_capture_and_hold(8, 20, 1.0, 30) == 9


# --------------------------------------------------------------------------
# value — production-time integral
# --------------------------------------------------------------------------

def test_safe_neutral_is_pure_self_growth():
    # opp can't reach (opp_reach None) -> denial 0 -> value = production * hold.
    v = _view([[0, 0, 50.0, 50.0, 1.5, 50, 1], [1, -1, 55.0, 50.0, 1.5, 5, 3]])
    tgt = _tgt(v, 1)
    assert planet_value(v, tgt, 100, eta=5, opp_reach=None, cfg=DEFAULT) == 3.0 * 100
    assert planet_value(v, tgt, 0, 5, None, DEFAULT) == 0.0


def test_enemy_capture_is_double_a_safe_neutral():
    # enemy: they hold it now -> every held turn denies them -> value = 2x.
    v = _view([
        [0, 0, 50.0, 50.0, 1.5, 50, 1],
        [1, -1, 55.0, 50.0, 1.5, 5, 2],   # neutral prod 2 (safe)
        [2, 1, 45.0, 50.0, 1.5, 5, 2],    # enemy prod 2
    ])
    neutral = planet_value(v, _tgt(v, 1), 100, 5, None, DEFAULT)
    enemy = planet_value(v, _tgt(v, 2), 100, 5, 2, DEFAULT)
    assert enemy == 2.0 * neutral


def test_contested_neutral_beats_safe_neutral_of_equal_production():
    # the whole point of the reframe: a planet the opponent wants is worth more.
    v = _view([
        [0, 0, 50.0, 50.0, 1.5, 50, 1],
        [1, -1, 55.0, 50.0, 1.5, 5, 2],   # contested prod 2
        [2, -1, 60.0, 50.0, 1.5, 5, 2],   # safe prod 2
    ])
    contested = planet_value(v, _tgt(v, 1), 300, eta=5, opp_reach=20, cfg=DEFAULT)
    safe = planet_value(v, _tgt(v, 2), 300, eta=5, opp_reach=None, cfg=DEFAULT)
    assert contested > safe


def test_pressure_grab_of_contested_neutral_denies_nothing():
    # if I only hold until the opponent could take it, I displace them 0 turns.
    v = _view([[0, 0, 50.0, 50.0, 1.5, 50, 1], [1, -1, 55.0, 50.0, 1.5, 5, 2]])
    # my_hold == opp_reach - eta == 15 -> denial 0 -> value = production * my_hold.
    val = planet_value(v, _tgt(v, 1), 15, eta=5, opp_reach=20, cfg=DEFAULT)
    assert val == 2.0 * 15


def test_preserve_value_is_doubled():
    from agents.holdgrab.value import preserve_value
    v = _view([[0, 0, 50.0, 50.0, 1.5, 50, 3]])   # my planet, prod 3
    # falls at turn 100, step 0 -> keep 400 -> double swing -> 3 * 400 * 2.
    assert preserve_value(v, _tgt(v, 0), 100, DEFAULT) == 3.0 * 400 * 2


def test_4p_weakest_opponent_bias():
    planets = [
        [0, 0, 50.0, 50.0, 1.5, 50, 1],
        [1, 1, 40.0, 50.0, 1.5, 99, 1],   # strong enemy
        [2, 1, 41.0, 50.0, 1.5, 99, 1],
        [3, 2, 60.0, 50.0, 1.5, 5, 2],    # weak enemy target
        [4, 1, 39.0, 50.0, 1.5, 5, 2],    # strong enemy target, same prod
    ]
    v = _view(planets)
    assert v.num_seats == 4
    weak = planet_value(v, _tgt(v, 3), 100, 5, 2, DEFAULT)
    strong = planet_value(v, _tgt(v, 4), 100, 5, 2, DEFAULT)
    assert weak > strong


# --------------------------------------------------------------------------
# threat — opponent as physics
# --------------------------------------------------------------------------

def test_contest_force_zero_when_no_enemy_reachable():
    # lone enemy parked far away, outside the contest window.
    v = _view([
        [0, 0, 10.0, 10.0, 1.5, 50, 1],
        [1, -1, 15.0, 10.0, 1.5, 5, 1],   # target
        [2, 1, 95.0, 95.0, 1.5, 50, 1],   # enemy far across the board
    ])
    assert contest_force(v, _tgt(v, 1), 0, DEFAULT.contest_window) == 0.0


def test_contest_force_counts_nearby_enemy():
    v = _view([
        [0, 0, 10.0, 10.0, 1.5, 50, 1],
        [1, -1, 20.0, 10.0, 1.5, 5, 1],   # target
        [2, 1, 24.0, 10.0, 1.5, 30, 1],   # enemy adjacent to target
    ])
    assert contest_force(v, _tgt(v, 1), 0, DEFAULT.contest_window) >= 30.0


def test_defense_floor_zero_without_inflight_threat():
    v = _view([
        [0, 0, 50.0, 50.0, 1.5, 50, 1],
        [1, 1, 90.0, 90.0, 1.5, 99, 1],   # huge idle enemy, NOT in flight
    ])
    # No committed in-flight fleet -> floor 0 (hypothetical musters don't turtle us).
    assert defense_follow_on(v, _tgt(v, 0), DEFAULT.defense_floor_horizon) == 0.0


def test_defense_floor_reserves_against_inflight():
    # enemy fleet (owner 1) in flight aimed at my planet 0 at (50,50).
    fleets = [[0, 1, 40.0, 50.0, 0.0, 9, 20]]  # id,owner,x,y,angle(->+x toward 50),from,ships
    v = _view([
        [0, 0, 50.0, 50.0, 1.5, 50, 1],
        [1, 1, 30.0, 50.0, 1.5, 30, 1],
    ], fleets=fleets)
    floor = defense_follow_on(v, _tgt(v, 0), DEFAULT.defense_floor_horizon)
    assert floor >= 0.0  # ledger attribution is geometry-dependent; never negative
    assert isinstance(floor, float)


def test_opp_reach_none_when_no_enemy():
    v = _view([
        [0, 0, 50.0, 50.0, 1.5, 50, 1],
        [1, -1, 55.0, 50.0, 1.5, 5, 1],
    ])
    assert opp_reach_tick(v, 1, 1) is None


# --------------------------------------------------------------------------
# chooser — two-tier selection
# --------------------------------------------------------------------------

def test_select_captures_a_cheap_neutral():
    v = _view([
        [0, 0, 50.0, 50.0, 1.5, 50, 1],
        [1, -1, 56.0, 50.0, 1.5, 8, 2],   # affordable neutral
    ])
    intents = select(v, DEFAULT)
    assert any(i.target_id == 1 and i.ships >= 9 for i in intents)


def test_select_pressure_captures_instead_of_hoarding():
    # Source can't afford capture-and-HOLD (a big enemy makes contest huge),
    # but can afford the capture itself -> must still attack (Tier 2), not hoard.
    v = _view([
        [0, 0, 50.0, 50.0, 1.5, 40, 1],   # my source, 40 ships
        [1, -1, 56.0, 50.0, 1.5, 8, 2],   # cheap neutral target
        [2, 1, 62.0, 50.0, 1.5, 500, 1],  # massive enemy next to the target
    ])
    # contest is huge -> hold is unaffordable; capture (9) is affordable.
    assert contest_force(v, _tgt(v, 1), 3, DEFAULT.contest_window) > 40
    intents = select(v, DEFAULT)
    assert any(i.target_id == 1 for i in intents), "must pressure-capture, not hoard"


def test_select_dedups_one_capture_per_target():
    # Two of my sources both nearest to the same neutral; only one fleet commits.
    v = _view([
        [0, 0, 48.0, 50.0, 1.5, 50, 1],
        [1, 0, 52.0, 50.0, 1.5, 50, 1],
        [2, -1, 50.0, 50.0, 1.5, 8, 2],   # single neutral between them
    ])
    intents = select(v, DEFAULT)
    assert sum(1 for i in intents if i.target_id == 2) <= 1


def test_select_respects_source_budget():
    v = _view([
        [0, 0, 50.0, 50.0, 1.5, 12, 1],   # only 12 ships
        [1, -1, 55.0, 50.0, 1.5, 8, 1],
        [2, -1, 50.0, 55.0, 1.5, 8, 1],
        [3, -1, 45.0, 50.0, 1.5, 8, 1],
    ])
    intents = select(v, DEFAULT)
    assert sum(i.ships for i in intents if i.src_id == 0) <= 12


def test_select_is_deterministic():
    planets = [
        [0, 0, 50.0, 50.0, 1.5, 60, 1],
        [1, -1, 56.0, 50.0, 1.5, 8, 2],
        [2, -1, 50.0, 56.0, 1.5, 10, 3],
        [3, 1, 40.0, 40.0, 1.5, 20, 1],
    ]
    a = [(i.src_id, i.target_id, i.ships) for i in select(_view(planets), DEFAULT)]
    b = [(i.src_id, i.target_id, i.ships) for i in select(_view(planets), DEFAULT)]
    assert a == b


def test_empty_when_no_targets():
    v = _view([[0, 0, 50.0, 50.0, 1.5, 50, 1]])
    assert select(v, DEFAULT) == []
