"""Tests for the simple-strategy panel (agents/simple/*).

The five strategies share v1's tie-break RNG, the DEFAULT_MECHANISMS
stack, and the "one launch per owned planet per turn" structure. The only
thing that differs is the target-selection score function.

Coverage per strategy:
- imports cleanly
- emits one Intent per owned planet on a hand-built panel obs
- picks the documented expected target on that obs
- agent(obs) returns env-format actions (list[[src_id, angle, ships]])
- returns [] when there are no targets
- 1-game smoke vs `random` builtin completes without crashing
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from kaggle_environments import make


REPO = Path(__file__).resolve().parents[1]
STRATEGIES = ["nearest", "production", "roi", "weakest", "enemy_first"]


# Hand-built obs: one mine + five targets chosen so each strategy picks a
# DIFFERENT target. All five paths from the mine are sun-clear (sun is at
# (50, 50) with radius 10) so the score-function tests verify what they
# intend even after sun_avoid joins DEFAULT_MECHANISMS. The previous layout
# had p2=(60,60) and p3=(50,50) crossing or hitting the sun from mine=(10,10).
#
# Distances from mine (10, 10):
#   p1=30.0  p2=85.0  p3=50.0  p4=7.07  p5=20.0
# ROI = production / (distance + 1):
#   p1=0.032 p2=0.058 p3=0.020 p4=0.124 p5=0.190
PANEL_OBS = {
    "player": 0,
    "step": 0,
    "planets": [
        [0, 0,  10.0, 10.0, 2.0, 200, 2],   # MINE (id=0)
        [1, 1,  40.0, 10.0, 2.0,  99, 1],   # enemy_first picks: only enemy
        [2, -1, 10.0, 95.0, 2.0,  99, 5],   # production picks: highest prod
        [3, -1, 60.0, 10.0, 1.0,   1, 1],   # weakest picks: 1 ship
        [4, -1, 15.0, 15.0, 1.0,  99, 1],   # nearest picks: closest
        [5, -1, 10.0, 30.0, 2.0,  20, 4],   # roi picks: best prod/dist
    ],
    "angular_velocity": 0.0,
    "comet_planet_ids": [],
    "comets": [],
}

EXPECTED_TARGET = {
    "nearest": 4,
    "production": 2,
    "roi": 5,
    "weakest": 3,
    "enemy_first": 1,
}


@pytest.fixture(scope="module", params=STRATEGIES)
def strategy(request):
    """Import each strategy module by name."""
    mod = importlib.import_module(f"agents.simple.{request.param}")
    return request.param, mod


def test_emits_one_intent_per_owned_planet(strategy):
    name, mod = strategy
    intents = mod.propose_intents(PANEL_OBS)
    assert len(intents) == 1, f"{name}: expected 1 intent (1 owned planet), got {len(intents)}"
    assert intents[0].src_id == 0


def test_picks_expected_target_on_panel_obs(strategy):
    name, mod = strategy
    intents = mod.propose_intents(PANEL_OBS)
    assert intents[0].target_id == EXPECTED_TARGET[name], (
        f"{name}: expected target {EXPECTED_TARGET[name]}, got {intents[0].target_id}. "
        f"PANEL_OBS distances/ROIs are documented in this test file."
    )


def test_agent_returns_env_format(strategy):
    """agent(obs) -> list of [src_id, aim_angle (float), ships (int)]."""
    name, mod = strategy
    actions = mod.agent(PANEL_OBS)
    assert isinstance(actions, list), name
    assert len(actions) == 1, name
    src_id, aim_angle, ships = actions[0]
    assert src_id == 0, name
    assert isinstance(aim_angle, float), name
    assert isinstance(ships, int) and ships > 0, name


def test_no_targets_returns_empty(strategy):
    """When all planets are owned by the player, no intents should be emitted."""
    name, mod = strategy
    obs = {
        "player": 0,
        "step": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 2.0, 100, 2],
            [1, 0, 50.0, 50.0, 2.0,  50, 3],
        ],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }
    assert mod.propose_intents(obs) == [], name
    assert mod.agent(obs) == [], name


@pytest.mark.parametrize("strategy_name", STRATEGIES)
def test_smoke_game_vs_random_completes(strategy_name):
    """1-game self-vs-random — would catch crashes in lib/mechanism wiring or bad obs handling."""
    mod = importlib.import_module(f"agents.simple.{strategy_name}")
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.run([mod.agent, "random"])
    final = env.steps[-1]
    assert all(s.status == "DONE" for s in final), (
        f"{strategy_name}: game did not reach DONE — statuses={[s.status for s in final]}"
    )


