"""Phase-dispatch state-machine tests for `agents.buildup_planner.main`.

Mocks `buildup.step` and `consolidation.step` so the test runs without
the heavy proposer/chooser stack. Asserts:

  1. New game (step==0) starts in BUILDUP.
  2. BUILDUP emits moves while `buildup.step` returns non-None.
  3. BUILDUP → CONSOLIDATION transition fires on the FIRST None return,
     SAME TURN — consolidation.step is invoked and its moves returned.
  4. Once in CONSOLIDATION, buildup.step is never called again.
  5. Per-seat state isolation: seat 0 and seat 1 evolve independently.
  6. Step==0 resets a per-seat state even mid-process (new game).
  7. Predicate stub returns None → no STRIKE transition in Step 1.
"""
from __future__ import annotations

from unittest.mock import patch

from agents.buildup_planner import main as bp_main


def _obs(step: int, player: int = 0):
    """Minimal obs dict with one planet so the agent doesn't short-circuit."""
    return {
        "player": player,
        "step": step,
        # One owned + one neutral planet keeps the "have my planets +
        # have opp planets" check happy. owner=-1 is neutral.
        "planets": [
            (0, player, 50.0, 50.0, 3.0, 10, 2),
            (1, -1, 30.0, 30.0, 3.0, 8, 1),
        ],
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
        "angular_velocity": 0.01,
    }


def _reset_state():
    bp_main._PHASE_STATE.clear()


def test_new_game_starts_in_buildup():
    _reset_state()
    with patch.object(bp_main.buildup, "step", return_value=[[0, 0.1, 5]]) as m_bu, \
         patch.object(bp_main.consolidation, "step", return_value=[]) as m_co:
        out = bp_main.agent(_obs(step=0))
    assert out == [[0, 0.1, 5]]
    assert m_bu.called
    assert not m_co.called
    assert bp_main._PHASE_STATE[0]["phase"] == bp_main.PHASE_BUILDUP


def test_buildup_emits_while_schedule_alive():
    _reset_state()
    with patch.object(bp_main.buildup, "step", return_value=[[0, 0.2, 3]]), \
         patch.object(bp_main.consolidation, "step", return_value=[]):
        for s in range(0, 10):
            out = bp_main.agent(_obs(step=s if s > 0 else 0))
            # step==0 starts a new game so only the first iteration resets.
            assert out == [[0, 0.2, 3]]
        assert bp_main._PHASE_STATE[0]["phase"] == bp_main.PHASE_BUILDUP


def test_buildup_to_consolidation_transition_same_turn():
    """First None return must transition AND emit consolidation's moves."""
    _reset_state()
    bp_main._PHASE_STATE[0] = {"phase": bp_main.PHASE_BUILDUP,
                               "strike_plan": None}
    with patch.object(bp_main.buildup, "step", return_value=None) as m_bu, \
         patch.object(bp_main.consolidation, "step",
                      return_value=[[0, 1.5, 7]]) as m_co:
        # step != 0 so the new-game reset doesn't fire and overwrite us.
        out = bp_main.agent(_obs(step=5))
    assert m_bu.called
    assert m_co.called
    assert out == [[0, 1.5, 7]]
    assert bp_main._PHASE_STATE[0]["phase"] == bp_main.PHASE_CONSOLIDATION


def test_consolidation_phase_does_not_call_buildup():
    _reset_state()
    bp_main._PHASE_STATE[0] = {"phase": bp_main.PHASE_CONSOLIDATION,
                               "strike_plan": None}
    with patch.object(bp_main.buildup, "step") as m_bu, \
         patch.object(bp_main.consolidation, "step",
                      return_value=[[0, 0.0, 1]]) as m_co:
        out = bp_main.agent(_obs(step=42))
    assert not m_bu.called
    assert m_co.called
    assert out == [[0, 0.0, 1]]


def test_per_seat_state_isolation():
    _reset_state()
    # Initialise seat 0 in BUILDUP, seat 1 in CONSOLIDATION (manually).
    bp_main._PHASE_STATE[1] = {"phase": bp_main.PHASE_CONSOLIDATION,
                               "strike_plan": None}
    with patch.object(bp_main.buildup, "step", return_value=[[0, 0.0, 1]]) as m_bu, \
         patch.object(bp_main.consolidation, "step",
                      return_value=[[1, 0.0, 2]]) as m_co:
        out0 = bp_main.agent(_obs(step=0, player=0))   # resets seat 0 to BUILDUP
        out1 = bp_main.agent(_obs(step=10, player=1))  # stays CONSOLIDATION
    assert out0 == [[0, 0.0, 1]]
    assert out1 == [[1, 0.0, 2]]
    assert bp_main._PHASE_STATE[0]["phase"] == bp_main.PHASE_BUILDUP
    assert bp_main._PHASE_STATE[1]["phase"] == bp_main.PHASE_CONSOLIDATION
    assert m_bu.called and m_co.called


def test_step_zero_resets_mid_process():
    """Two consecutive games for the same seat: a new step==0 must reset."""
    _reset_state()
    bp_main._PHASE_STATE[0] = {"phase": bp_main.PHASE_CONSOLIDATION,
                               "strike_plan": None}
    with patch.object(bp_main.buildup, "step", return_value=[[0, 0.0, 1]]), \
         patch.object(bp_main.consolidation, "step", return_value=[]):
        bp_main.agent(_obs(step=0, player=0))
    assert bp_main._PHASE_STATE[0]["phase"] == bp_main.PHASE_BUILDUP


def test_predicate_stub_keeps_consolidation():
    """Step 1 stub never elects STRIKE."""
    from agents.buildup_planner import predicates
    _reset_state()
    bp_main._PHASE_STATE[0] = {"phase": bp_main.PHASE_CONSOLIDATION,
                               "strike_plan": None}
    plan = predicates.evaluate_inflection(world=None, model=None,
                                          me=0, opp_id=1)
    assert plan is None
    with patch.object(bp_main.consolidation, "step", return_value=[]):
        bp_main.agent(_obs(step=50))
    assert bp_main._PHASE_STATE[0]["phase"] == bp_main.PHASE_CONSOLIDATION


def test_empty_planets_returns_empty():
    """Degenerate obs (no planets) returns empty without crashing."""
    _reset_state()
    obs = _obs(step=0)
    obs["planets"] = []
    out = bp_main.agent(obs)
    assert out == []


def _obs_2p(step: int, player: int = 0):
    """Obs with a real opponent (player 1) so `opp_id_2p` returns 1
    instead of -1 (which would short-circuit before the predicate)."""
    return {
        "player": player,
        "step": step,
        "planets": [
            (0, 0, 50.0, 30.0, 3.0, 10, 2),    # me
            (1, 1, 50.0, 70.0, 3.0, 10, 2),    # opp
        ],
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
        "angular_velocity": 0.01,
    }


def test_strike_election_routes_to_strike_phase(monkeypatch):
    """When predicate elects AND BUILDUP_PLANNER_STRIKE_ENABLED=1, the
    dispatcher transitions BUILDUP/CONSOLIDATION → STRIKE same-turn,
    calls strike.step, and resets to CONSOLIDATION for next turn.

    Mocks the predicate to return a StrikePlan and strike.step to return
    a sentinel move list. Asserts the dispatcher emits the strike moves
    (NOT consolidation's) on the elect turn AND ends in CONSOLIDATION."""
    from agents.buildup_planner import predicates, strike
    _reset_state()
    bp_main._PHASE_STATE[0] = {"phase": bp_main.PHASE_CONSOLIDATION,
                               "strike_plan": None,
                               "game_id": "test"}
    fake_plan = predicates.StrikePlan(
        target_ids=frozenset({1}), arrival_step=42, shots=(),
    )
    sentinel_moves = [[0, 1.23, 7]]
    monkeypatch.setenv("BUILDUP_PLANNER_STRIKE_ENABLED", "1")
    with patch.object(bp_main.predicates, "evaluate_inflection",
                      return_value=fake_plan), \
         patch.object(strike, "step", return_value=sentinel_moves) as m_strike, \
         patch.object(bp_main.consolidation, "step",
                      return_value=[[99, 0.0, 99]]) as m_cons:
        out = bp_main.agent(_obs_2p(step=50))
    # Strike-only emission: strike's moves, NOT consolidation's.
    assert out == sentinel_moves
    assert m_strike.called
    assert not m_cons.called
    # Phase resets to CONSOLIDATION so the NEXT turn routes normally.
    assert bp_main._PHASE_STATE[0]["phase"] == bp_main.PHASE_CONSOLIDATION
    assert bp_main._PHASE_STATE[0]["strike_plan"] is None


def test_strike_disabled_keeps_consolidation_even_with_plan(monkeypatch):
    """`BUILDUP_PLANNER_STRIKE_ENABLED=0` (default): predicate may return
    a plan but the dispatcher MUST stay in CONSOLIDATION and not call
    strike.step. Guards against accidental flag-flip regressions."""
    from agents.buildup_planner import predicates, strike
    _reset_state()
    bp_main._PHASE_STATE[0] = {"phase": bp_main.PHASE_CONSOLIDATION,
                               "strike_plan": None,
                               "game_id": "test"}
    fake_plan = predicates.StrikePlan(
        target_ids=frozenset({1}), arrival_step=42, shots=(),
    )
    monkeypatch.setenv("BUILDUP_PLANNER_STRIKE_ENABLED", "0")
    with patch.object(bp_main.predicates, "evaluate_inflection",
                      return_value=fake_plan), \
         patch.object(strike, "step", return_value=[[0, 0.0, 1]]) as m_strike, \
         patch.object(bp_main.consolidation, "step",
                      return_value=[[7, 0.7, 7]]) as m_cons:
        out = bp_main.agent(_obs_2p(step=50))
    assert out == [[7, 0.7, 7]]
    assert not m_strike.called
    assert m_cons.called
    assert bp_main._PHASE_STATE[0]["phase"] == bp_main.PHASE_CONSOLIDATION
