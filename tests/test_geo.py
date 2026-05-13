"""Tests for the geo agent (lib/geo/{sense,posture,allocator} + agents/geo).

Synthetic-obs unit tests for each layer + an end-to-end smoke that the
entry point runs without exception on a kaggle_environments game.
"""

from __future__ import annotations

import pytest

from lib.intent import World
from lib.mission import Mission
from lib.world_model import WorldModel

from lib.geo.allocator import allocate, allocate_greedy_multi, allocate_lp
from lib.geo.posture import Posture, decide_posture
from lib.geo.sense import sense_state, MUTUAL_REACH_TURNS


def _planet(pid, owner, ships, prod=2, x=50.0, y=50.0, radius=1.5):
    return [pid, owner, x, y, radius, ships, prod]


def _world(planets, *, my_id=0, step=0, fleets=None, comets=None):
    obs = {
        "player": my_id,
        "planets": planets,
        "angular_velocity": 0.05,
        "comet_planet_ids": [],
        "step": step,
        "comets": comets or [],
        "fleets": fleets or [],
    }
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# sense.py
# ---------------------------------------------------------------------------


def test_sense_clusters_two_close_planets_together():
    """Two of my planets ~5 turns apart should land in the same cluster."""
    planets = [
        _planet(0, owner=0, ships=10, x=10.0, y=10.0),
        _planet(1, owner=0, ships=10, x=20.0, y=10.0),  # ~10 units = small ETA
        _planet(2, owner=1, ships=10, x=90.0, y=90.0),  # far enemy
    ]
    world = _world(planets)
    model = WorldModel.from_world(world)
    sense = sense_state(world, model)
    assert len(sense.my_clusters) == 1
    assert set(sense.my_clusters[0].planet_ids) == {0, 1}
    assert len(sense.enemy_clusters) == 1
    assert sense.enemy_clusters[0].planet_ids == [2]


def test_sense_splits_distant_planets_into_separate_clusters():
    """Two of my planets far apart land in different clusters."""
    planets = [
        _planet(0, owner=0, ships=10, x=5.0, y=5.0),
        _planet(1, owner=0, ships=10, x=95.0, y=95.0),
    ]
    world = _world(planets)
    model = WorldModel.from_world(world)
    sense = sense_state(world, model)
    assert len(sense.my_clusters) == 2


def test_sense_voronoi_assigns_close_neutral_to_my_cluster():
    """Neutral planet much closer to me than enemy → in my Voronoi cell."""
    planets = [
        _planet(0, owner=0,  ships=10, x=10.0, y=10.0),
        _planet(1, owner=-1, ships=2,  x=20.0, y=20.0),  # near me
        _planet(2, owner=1,  ships=10, x=90.0, y=90.0),  # far enemy
    ]
    world = _world(planets)
    model = WorldModel.from_world(world)
    sense = sense_state(world, model)
    assert sense.voronoi.get(1) == sense.my_clusters[0].idx


def test_sense_voronoi_marks_contested_when_equidistant():
    """Neutral roughly equidistant from my and enemy clusters → contested."""
    planets = [
        _planet(0, owner=0,  ships=10, x=10.0, y=50.0),
        _planet(1, owner=-1, ships=2,  x=50.0, y=50.0),  # midpoint
        _planet(2, owner=1,  ships=10, x=90.0, y=50.0),
    ]
    world = _world(planets)
    model = WorldModel.from_world(world)
    sense = sense_state(world, model)
    # Either contested or absent from voronoi (enemy reaches faster) — both
    # are valid "not in our Voronoi cell" outcomes.
    assert 1 not in sense.voronoi or sense.voronoi[1] == -1


def test_sense_front_detection_flags_planets_near_enemy():
    """My planet within front-radius of an enemy planet is on the front."""
    planets = [
        _planet(0, owner=0, ships=10, x=10.0, y=10.0),   # far from enemy
        _planet(1, owner=0, ships=10, x=55.0, y=55.0),   # adjacent to enemy
        _planet(2, owner=1, ships=10, x=60.0, y=60.0),
    ]
    world = _world(planets)
    model = WorldModel.from_world(world)
    sense = sense_state(world, model)
    assert 1 in sense.front_pids
    assert 0 not in sense.front_pids


def test_sense_empty_world_returns_empty_state():
    world = _world([])
    model = WorldModel.from_world(world)
    sense = sense_state(world, model)
    assert sense.my_clusters == []
    assert sense.enemy_clusters == []
    assert sense.voronoi == {}
    assert sense.front_pids == set()


# ---------------------------------------------------------------------------
# posture.py
# ---------------------------------------------------------------------------


def _build_sense_with_threat(my_pid: int, incoming: int, on_front: bool):
    """Build a SenseState with one threatened planet for posture tests."""
    from lib.geo.sense import SenseState, Cluster
    s = SenseState()
    s.my_clusters = [Cluster(idx=0, owner=0, planet_ids=[my_pid], total_ships=10)]
    s.enemy_clusters = [Cluster(idx=1, owner=1, planet_ids=[99], total_ships=10)]
    s.threat_budget = {my_pid: incoming}
    if on_front:
        s.front_pids = {my_pid}
    return s


def test_posture_opening_in_first_six_steps():
    for step in (0, 1, 5):
        planets = [_planet(0, owner=0, ships=10)]
        world = _world(planets, step=step)
        model = WorldModel.from_world(world)
        sense = sense_state(world, model)
        assert decide_posture(world, sense, model) is Posture.OPENING


def test_posture_defend_when_front_planet_under_heavy_threat():
    planets = [
        _planet(0, owner=0, ships=10, x=50.0, y=50.0),
        _planet(99, owner=1, ships=10, x=55.0, y=55.0),
    ]
    world = _world(planets, step=100)
    model = WorldModel.from_world(world)
    sense = _build_sense_with_threat(0, incoming=8, on_front=True)
    assert decide_posture(world, sense, model) is Posture.DEFEND


def test_posture_expand_when_no_threat_no_concentration():
    planets = [
        _planet(0, owner=0, ships=10, x=10.0, y=10.0),
        _planet(99, owner=1, ships=20, x=90.0, y=90.0),
    ]
    world = _world(planets, step=100)
    model = WorldModel.from_world(world)
    sense = sense_state(world, model)
    # No front-planet threat and we don't dominate → EXPAND.
    assert decide_posture(world, sense, model) is Posture.EXPAND


# ---------------------------------------------------------------------------
# allocator.py
# ---------------------------------------------------------------------------


def _mk_mission(src_id, target_id, ships, score, eta=5, klass="snipe"):
    return Mission(
        mission_class=klass,
        src_id=src_id,
        target_id=target_id,
        ships=ships,
        score=score,
        eta=eta,
    )


def _trivial_world_for_alloc():
    planets = [
        _planet(0, owner=0, ships=50, x=10.0, y=10.0),
        _planet(1, owner=1, ships=5,  x=30.0, y=30.0),
        _planet(2, owner=1, ships=5,  x=70.0, y=70.0),
    ]
    world = _world(planets, step=100)
    model = WorldModel.from_world(world)
    sense = sense_state(world, model)
    return world, model, sense


def test_greedy_picks_highest_scoring_missions_within_source_budget():
    world, model, sense = _trivial_world_for_alloc()
    missions = [
        _mk_mission(0, 1, ships=10, score=10.0),
        _mk_mission(0, 2, ships=10, score=5.0),
    ]
    intents = allocate_greedy_multi(missions, world, sense, Posture.EXPAND, model)
    # Both missions affordable from src 0 (budget = 50 - 4 = 46). Both should fire.
    assert len(intents) == 2
    targets = {i.target_id for i in intents}
    assert targets == {1, 2}


def test_greedy_respects_source_budget_cap():
    world, model, sense = _trivial_world_for_alloc()
    missions = [
        _mk_mission(0, 1, ships=40, score=10.0),
        _mk_mission(0, 2, ships=40, score=9.0),  # together exceed budget
    ]
    intents = allocate_greedy_multi(missions, world, sense, Posture.EXPAND, model)
    # Only the higher-score one fits.
    assert len(intents) == 1
    assert intents[0].target_id == 1


def test_lp_matches_greedy_on_simple_case():
    world, model, sense = _trivial_world_for_alloc()
    missions = [
        _mk_mission(0, 1, ships=10, score=10.0),
        _mk_mission(0, 2, ships=10, score=5.0),
    ]
    lp_intents = allocate_lp(missions, world, sense, Posture.EXPAND, model)
    greedy_intents = allocate_greedy_multi(missions, world, sense, Posture.EXPAND, model)
    assert {i.target_id for i in lp_intents} == {i.target_id for i in greedy_intents}


def test_allocator_empty_missions_returns_empty():
    world, model, sense = _trivial_world_for_alloc()
    assert allocate([], world, sense, Posture.EXPAND, model) == []


def test_allocator_opening_posture_uses_zero_reserve():
    """In OPENING, source budget should be the full garrison (no reserve)."""
    world, model, sense = _trivial_world_for_alloc()
    # Mission requesting ALL 50 ships from src 0. EXPAND keeps reserve=4, so
    # 50 > 50-4=46 → drops. OPENING reserve=0, so 50 ≤ 50 → fires.
    missions = [_mk_mission(0, 1, ships=50, score=1.0)]
    out_expand = allocate_greedy_multi(missions, world, sense, Posture.EXPAND, model)
    out_opening = allocate_greedy_multi(missions, world, sense, Posture.OPENING, model)
    assert len(out_expand) == 0
    assert len(out_opening) == 1


# ---------------------------------------------------------------------------
# End-to-end agent smoke (with kaggle_environments)
# ---------------------------------------------------------------------------


def test_agent_runs_full_game_without_error():
    """The agent should play a full 500-step game vs random without crashing."""
    pytest.importorskip("kaggle_environments")
    from kaggle_environments import make
    from agents.geo.main import agent as geo_agent

    env = make("orbit_wars", configuration={"seed": 0}, debug=False)
    env.run([geo_agent, "random"])
    # Both players returned a final reward (None means crash).
    final = env.steps[-1]
    assert final[0].reward is not None
    assert final[1].reward is not None


def test_agent_returns_action_list_at_step_0():
    """Calling agent on a step-0 obs returns a list (possibly empty)."""
    pytest.importorskip("kaggle_environments")
    from kaggle_environments import make
    from agents.geo.main import agent as geo_agent

    env = make("orbit_wars", configuration={"seed": 0}, debug=False)
    obs = env.steps[0][0].observation
    action = geo_agent(obs, env.configuration)
    assert isinstance(action, list)
