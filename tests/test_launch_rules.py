"""Tests for the post-emit launch-discipline validator (Rules A & B).

Rule A — neutral discipline: a fleet sent to a NEUTRAL planet is kept
  only if it (alone, or with same-tick coalition partners) captures it.
Rule B — opponent predictability ceiling: an opponent capture is kept
  only if the fleet arrives within K turns; later arrivals are dropped.

Two layers:
  * Logic tests fake `predict_fleet_fate` to assign each move a precise
    destination + arrival tick, but use the REAL `predict_garrison_at`
    combat so capture/bounce math is the production path.
  * Integration tests follow Rule 38: with the gate OFF, real mid-game
    boards emit rule-violating launches (reproduce the failure); with it
    ON, the agent's emitted moves contain zero violations (failure gone).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from kaggle_environments import make

import agents.baseline.launch_rules as LR
from agents.baseline.launch_rules import enforce_launch_rules
from lib.intent import World
from lib.world_model import WorldModel


# --------------------------------------------------------------------------
# Logic layer — fake fate, real combat.
# --------------------------------------------------------------------------
#
# A move's `angle` field encodes its destination for the fake ray-cast:
#     angle = hit_pid * 1000 + arrival_step
# (hit_pid == 999 means the fleet dies in the sun — out of scope).

def _fake_fate(src, target, angle, ships, world, max_steps=200, wait_N=0):
    code = int(round(float(angle)))
    pid, step = divmod(code, 1000)
    if pid == 999:
        return SimpleNamespace(outcome="sun", hit_planet_id=None, step=step)
    return SimpleNamespace(outcome="planet", hit_planet_id=pid, step=step)


def _aim(pid, step):
    return float(pid * 1000 + step)


def _planet(pid, owner, ships=0, production=0):
    # predict_garrison_at reads .owner/.ships/.production; resolve reads .id.
    return SimpleNamespace(
        id=pid, owner=owner, ships=float(ships), production=production,
    )


class _World:
    def __init__(self, planets):
        self.planets_by_id = {int(p.id): p for p in planets}


class _Model:
    def __init__(self, ledger=None):
        self.ledger = ledger or {}


@pytest.fixture(autouse=True)
def _enable_rules(monkeypatch):
    monkeypatch.setenv("BASELINE_LAUNCH_RULES", "1")
    monkeypatch.setenv("BASELINE_CAPTURE_HORIZON_K", "10")
    monkeypatch.setattr(LR, "predict_fleet_fate", _fake_fate)


# Source planet shared by the logic tests (id 1, ours).
def _setup(*targets, me=0, ledger=None):
    planets = [_planet(1, me, ships=200, production=5), *targets]
    return planets, _World(planets), _Model(ledger)


def test_neutral_solo_bounce_dropped():
    _p, world, model = _setup(_planet(5, -1, ships=20))
    moves = [[1, _aim(5, 3), 10]]  # 10 ships vs 20 garrison → bounce
    assert enforce_launch_rules(moves, _p, 0, world, model) == []


def test_neutral_solo_capture_kept():
    _p, world, model = _setup(_planet(5, -1, ships=8))
    moves = [[1, _aim(5, 3), 12]]  # 12 vs 8 → capture
    assert enforce_launch_rules(moves, _p, 0, world, model) == moves


def test_neutral_same_tick_coalition_kept():
    # 12 + 12 = 24 arriving the SAME tick beats a 20-ship garrison.
    planets = [
        _planet(1, 0, ships=200, production=5),
        _planet(2, 0, ships=200, production=5),
        _planet(5, -1, ships=20),
    ]
    world, model = _World(planets), _Model()
    moves = [[1, _aim(5, 3), 12], [2, _aim(5, 3), 12]]
    assert enforce_launch_rules(moves, planets, 0, world, model) == moves


def test_neutral_staggered_coalition_dropped():
    # Same two fleets but arriving on DIFFERENT ticks → neither group
    # captures → both dropped (Rule A forbids staggered pokes).
    planets = [
        _planet(1, 0, ships=200, production=5),
        _planet(2, 0, ships=200, production=5),
        _planet(5, -1, ships=20),
    ]
    world, model = _World(planets), _Model()
    moves = [[1, _aim(5, 3), 12], [2, _aim(5, 4), 12]]
    assert enforce_launch_rules(moves, planets, 0, world, model) == []


def test_neutral_same_tick_partial_coalition_atomic_drop():
    # 12 + 5 = 17 same tick still < 20 → the WHOLE group drops, not one leg.
    planets = [
        _planet(1, 0, ships=200, production=5),
        _planet(2, 0, ships=200, production=5),
        _planet(5, -1, ships=20),
    ]
    world, model = _World(planets), _Model()
    moves = [[1, _aim(5, 3), 12], [2, _aim(5, 3), 5]]
    assert enforce_launch_rules(moves, planets, 0, world, model) == []


def test_neutral_inbound_enemy_same_tick_dropped():
    # Neutral has 5 ships; an enemy fleet (owner 1) also arrives the same
    # tick with 10. My 12 beat the enemy 10 (survivor 2) but 2 < 5 garrison
    # → I do NOT capture → dropped. Exercises the model.ledger path.
    _p, world, model = _setup(
        _planet(5, -1, ships=5), ledger={5: [(3, 1, 10)]},
    )
    moves = [[1, _aim(5, 3), 12]]
    assert enforce_launch_rules(moves, _p, 0, world, model) == []


def test_opponent_within_horizon_kept():
    _p, world, model = _setup(_planet(7, 1, ships=5, production=2))
    moves = [[1, _aim(7, 8), 50]]  # arrival 8 <= K=10
    assert enforce_launch_rules(moves, _p, 0, world, model) == moves


def test_opponent_beyond_horizon_dropped():
    _p, world, model = _setup(_planet(7, 1, ships=5, production=2))
    moves = [[1, _aim(7, 12), 50]]  # arrival 12 > K=10
    assert enforce_launch_rules(moves, _p, 0, world, model) == []


def test_opponent_horizon_is_configurable(monkeypatch):
    monkeypatch.setenv("BASELINE_CAPTURE_HORIZON_K", "5")
    _p, world, model = _setup(_planet(7, 1, ships=5, production=2))
    moves = [[1, _aim(7, 8), 50]]  # arrival 8 > K=5 now
    assert enforce_launch_rules(moves, _p, 0, world, model) == []


def test_near_reinforcement_kept():
    # Reinforcement of our own planet WITHIN the K horizon is kept.
    _p, world, model = _setup(_planet(3, 0, ships=1, production=2))
    moves = [[1, _aim(3, 8), 30]]  # arrival 8 <= K=10
    assert enforce_launch_rules(moves, _p, 0, world, model) == moves


def test_far_reinforcement_dropped():
    # Universal K ceiling (PI 2026-05-30): even reinforcement of our own
    # planet is dropped when the fleet arrives beyond K (the slow fleet
    # bets on an unpredictable board and often lands at a contested planet).
    _p, world, model = _setup(_planet(3, 0, ships=1, production=2))
    moves = [[1, _aim(3, 40), 30]]  # arrival 40 > K=10
    assert enforce_launch_rules(moves, _p, 0, world, model) == []


def test_far_neutral_capture_dropped_by_ceiling():
    # A neutral the model says we'd capture, but arriving beyond K → the
    # universal ceiling drops it before the Rule A capture check.
    _p, world, model = _setup(_planet(5, -1, ships=8))
    moves = [[1, _aim(5, 25), 12]]  # would capture (12 > 8) but arrival 25 > K
    assert enforce_launch_rules(moves, _p, 0, world, model) == []


def test_sun_death_out_of_scope_kept():
    _p, world, model = _setup(_planet(5, -1, ships=8))
    moves = [[1, _aim(999, 5), 10]]  # dies in the sun → not our concern here
    assert enforce_launch_rules(moves, _p, 0, world, model) == moves


def test_gate_off_is_noop(monkeypatch):
    monkeypatch.setenv("BASELINE_LAUNCH_RULES", "0")
    _p, world, model = _setup(_planet(5, -1, ships=20))
    moves = [[1, _aim(5, 3), 10]]  # would bounce, but gate is off
    assert enforce_launch_rules(moves, _p, 0, world, model) == moves


def test_mixed_full_scope_filters_only_violators():
    # One capturing neutral within K (kept), one bouncing neutral (dropped),
    # one in-horizon opponent (kept), one out-of-horizon opponent (dropped),
    # one near reinforcement (kept), one FAR reinforcement (dropped by the
    # universal ceiling). Verifies order preservation + selectivity.
    planets = [
        _planet(1, 0, ships=400, production=5),
        _planet(5, -1, ships=8),    # capture (within K)
        _planet(6, -1, ships=30),   # bounce
        _planet(7, 1, ships=5),     # opp in-horizon
        _planet(8, 1, ships=5),     # opp out-of-horizon
        _planet(9, 0, ships=1),     # reinforce near
        _planet(10, 0, ships=1),    # reinforce far
    ]
    world, model = _World(planets), _Model()
    keep1 = [1, _aim(5, 4), 12]
    drop1 = [1, _aim(6, 4), 10]
    keep2 = [1, _aim(7, 9), 40]
    drop2 = [1, _aim(8, 11), 40]
    keep3 = [1, _aim(9, 8), 5]
    drop3 = [1, _aim(10, 20), 5]
    moves = [keep1, drop1, keep2, drop2, keep3, drop3]
    assert enforce_launch_rules(moves, planets, 0, world, model) == [
        keep1, keep2, keep3,
    ]


# --------------------------------------------------------------------------
# Integration layer — Rule 38 reproduce / confirm on real boards.
# --------------------------------------------------------------------------

def _obs_of(state, seat):
    s = state[seat]
    return s["observation"] if isinstance(s, dict) else s.observation


def _player_of(obs):
    return int(obs["player"] if isinstance(obs, dict) else obs.player)


def _play_collect(seed, rules_on, max_turns):
    """Play a 2P champion-vs-champion game (trajectory chooser); collect
    seat-0's `(obs, moves)` for each turn. The launch-rules gate is set
    per `rules_on` for the whole playout."""
    os.environ["BASELINE_CHOOSER"] = "trajectory"
    if rules_on:
        os.environ["BASELINE_LAUNCH_RULES"] = "1"
        os.environ["BASELINE_CAPTURE_HORIZON_K"] = "10"
    else:
        # Force gate off explicitly — our main.py's setdefault would
        # otherwise re-populate "1" on first import; a pop wouldn't
        # survive the module-import cache between test runs.
        os.environ["BASELINE_LAUNCH_RULES"] = "0"
    from agents.baseline.main import agent
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": 500})
    env.reset(num_agents=2)
    state = env.steps[0]
    turns = []
    for _ in range(max_turns):
        obs0 = _obs_of(state, 0)
        obs1 = _obs_of(state, 1)
        a0 = agent(obs0, env.configuration)
        a1 = agent(obs1, env.configuration)
        turns.append((obs0, a0))
        state = env.step([a0, a1])
        if (state[0]["status"] if isinstance(state[0], dict)
                else state[0].status) != "ACTIVE":
            break
    return turns


def _violations(moves, obs, me):
    """Re-resolve `moves` against the obs and count rule violations,
    independent of any gate state. Uses the real fate + combat."""
    # Force the real fate (logic-layer monkeypatch is function-scoped, but
    # be defensive) and the gate ON with K=10 for the verifier.
    prev_gate = os.environ.get("BASELINE_LAUNCH_RULES")
    prev_k = os.environ.get("BASELINE_CAPTURE_HORIZON_K")
    os.environ["BASELINE_LAUNCH_RULES"] = "1"
    os.environ["BASELINE_CAPTURE_HORIZON_K"] = "10"
    try:
        world = World.from_obs(obs)
        model = WorldModel.from_world(world)
        kept = enforce_launch_rules(list(moves), None, me, world, model)
    finally:
        if prev_gate is None:
            os.environ.pop("BASELINE_LAUNCH_RULES", None)
        else:
            os.environ["BASELINE_LAUNCH_RULES"] = prev_gate
        if prev_k is None:
            os.environ.pop("BASELINE_CAPTURE_HORIZON_K", None)
        else:
            os.environ["BASELINE_CAPTURE_HORIZON_K"] = prev_k
    # number of moves the rules would remove
    return len(moves) - len(kept)


SEEDS = [1, 2, 3]
MAX_TURNS = 60


def test_reproduce_failure_rules_off_emits_violations():
    """With the gate OFF the champion emits at least one rule-violating
    launch over the course of a game (reproduces the failure state).
    Scans turn-by-turn and stops at the first violation found."""
    found = 0
    for seed in SEEDS:
        turns = _play_collect(seed, rules_on=False, max_turns=MAX_TURNS)
        for obs, moves in turns:
            found += _violations(moves, obs, _player_of(obs))
        if found:
            break
    assert found > 0, (
        "the champion never emitted a neutral non-capture or an opponent "
        "capture arriving beyond K on any sampled turn — the rules do not "
        "bind in practice (null finding; see audit)."
    )


def test_rules_on_emits_zero_violations():
    """With the gate ON every emitted launch satisfies both rules on every
    turn of a game (the failure is gone end-to-end)."""
    for seed in SEEDS:
        turns = _play_collect(seed, rules_on=True, max_turns=MAX_TURNS)
        for obs, moves in turns:
            assert _violations(moves, obs, _player_of(obs)) == 0, (
                f"seed {seed}: agent emitted a rule-violating launch with "
                f"the gate on: {moves}"
            )
