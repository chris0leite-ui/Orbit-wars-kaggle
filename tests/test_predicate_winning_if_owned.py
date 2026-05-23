"""Unit tests for `lib.joint_solver.predicate.is_winning_state_if_owned`.

This primitive answers: "if I *also* owned these extra planets, would my
production overwhelm what the opponent could still recover?" It is the
central go/no-go gate the Step-2 inflection predicate consults before
electing a STRIKE. Production has used it only via the multi-source
attack codepath on PFhzM and via `is_winning_state` (the empty-S case)
elsewhere — these cases close the gap on the `extra_planet_ids`
re-attribution math itself.

Pinned cases:
  1. empty S reduces to `is_winning_state`.
  2. S containing already-mine pid is idempotent (prod stays attributed
     to me; opp_pool unchanged).
  3. neutral S bumps `prod_advantage` by `+prod` and leaves `opp_pool`
     unchanged (no ships, no opp prod to remove).
  4. 2P True case: capturing the opp's only planet flips a losing
     position into a winning one.
  5. 2P False case: opp's recovery pool still exceeds my advantage even
     with the capture.
  6. multi-element S re-attributes each planet exactly once (no double
     counting; symmetric in the set).
  7. caller-side `opp_id == -1` short-circuit (4P): document the
     contract by asserting `opp_id_2p` returns `-1` when there are
     multiple opponents.
"""
from __future__ import annotations

from lib.intent import World
from lib.joint_solver import predicate as pred


def _obs(planets, step: int = 0, fleets=None, player: int = 0):
    """Build a kaggle-style obs dict from `planets` tuples.

    Each planet tuple: (id, owner, x, y, radius, ships, production).
    """
    return {
        "player": player,
        "step": step,
        "planets": planets,
        "fleets": fleets or [],
        "comets": [],
        "comet_planet_ids": [],
        "angular_velocity": 0.01,
    }


def _world(planets, step: int = 0, fleets=None, player: int = 0) -> World:
    return World.from_obs(_obs(planets, step, fleets, player))


def test_empty_S_reduces_to_is_winning_state():
    """is_winning_state_if_owned(world, …, set()) == is_winning_state(world, …)."""
    planets = [
        (0, 0, 50.0, 30.0, 3.0, 100, 5),   # me
        (1, 1, 50.0, 70.0, 3.0, 50, 3),    # opp
        (2, -1, 30.0, 50.0, 3.0, 5, 1),    # neutral
    ]
    w = _world(planets, step=100)
    assert (pred.is_winning_state_if_owned(w, my_id=0, opp_id=1, extra_planet_ids=set())
            == pred.is_winning_state(w, my_id=0, opp_id=1))


def test_already_mine_pid_is_idempotent():
    """Passing a planet I already own must not change the verdict.

    is_winning_state_if_owned skips planets already owned by `my_id`,
    so the result must equal the empty-S (is_winning_state) baseline.
    """
    planets = [
        (0, 0, 50.0, 30.0, 3.0, 100, 5),   # me
        (1, 0, 50.0, 70.0, 3.0, 80, 4),    # me, too
        (2, 1, 30.0, 50.0, 3.0, 50, 3),    # opp
    ]
    w = _world(planets, step=100)
    baseline = pred.is_winning_state(w, my_id=0, opp_id=1)
    with_own = pred.is_winning_state_if_owned(w, my_id=0, opp_id=1,
                                              extra_planet_ids={0, 1})
    assert with_own == baseline


def test_neutral_S_bumps_adv_leaves_opp_pool():
    """Capturing a neutral: +prod to advantage; opp_pool unchanged.

    The predicate formula is `adv * remaining_turns > opp_pool`. With a
    neutral added: adv grows by neutral's prod; opp_pool unchanged.
    Construct a scenario where the empty-S verdict is False but adding
    a neutral with large enough production flips it to True.
    """
    # Tuned so the empty-S verdict is False but adding the neutral wins.
    # adv=0, opp_pool large → False. Add neutral with prod=5 → adv=5,
    # rem=400 → 2000 > opp_pool (= 50 + 3*400 = 1250). True.
    planets = [
        (0, 0, 50.0, 30.0, 3.0, 100, 3),   # me, prod=3
        (1, 1, 50.0, 70.0, 3.0, 50, 3),    # opp, prod=3 → adv=0
        (2, -1, 30.0, 50.0, 3.0, 5, 5),    # neutral, prod=5
    ]
    w = _world(planets, step=100)
    assert pred.is_winning_state(w, my_id=0, opp_id=1) is False
    assert pred.is_winning_state_if_owned(w, my_id=0, opp_id=1,
                                          extra_planet_ids={2}) is True


def test_capture_opp_flips_losing_to_winning_2p():
    """2P True case: removing the opp's planet flips the verdict."""
    planets = [
        (0, 0, 50.0, 30.0, 3.0, 50, 3),    # me
        (1, 1, 50.0, 70.0, 3.0, 100, 5),   # opp — losing for me
    ]
    w = _world(planets, step=100)
    assert pred.is_winning_state(w, my_id=0, opp_id=1) is False
    # Capturing the opp's only planet: adv += 2*5=10, opp_pool -= 100+5*400=2100
    # New adv=7, opp_pool_after = 0. 7 * 400 = 2800 > 0 → True.
    assert pred.is_winning_state_if_owned(w, my_id=0, opp_id=1,
                                          extra_planet_ids={1}) is True


def test_capture_opp_insufficient_margin_stays_losing():
    """2P False case: opp recovers fast enough that the capture still loses.

    Opp has TWO planets; we capture only one. The other has high
    production and ships such that the remaining opp_pool still
    out-paces my advantage.
    """
    planets = [
        (0, 0, 50.0, 30.0, 3.0, 1, 1),     # me — minimal prod
        (1, 1, 50.0, 70.0, 3.0, 100, 3),   # opp — we'll capture this
        (2, 1, 70.0, 50.0, 3.0, 500, 5),   # opp — leftover, dominant
    ]
    w = _world(planets, step=480)  # remaining_turns = 20
    # Capturing planet 1: adv = 1 - 8 + 2*3 = -1 (still negative!) → False.
    assert pred.is_winning_state_if_owned(w, my_id=0, opp_id=1,
                                          extra_planet_ids={1}) is False


def test_multi_element_S_attributes_each_planet_once():
    """Multi-element S must apply the per-planet update exactly once and
    be symmetric in set membership."""
    planets = [
        (0, 0, 50.0, 30.0, 3.0, 50, 3),    # me
        (1, 1, 50.0, 70.0, 3.0, 30, 2),    # opp
        (2, 1, 70.0, 50.0, 3.0, 40, 2),    # opp
        (3, -1, 30.0, 50.0, 3.0, 5, 1),    # neutral
    ]
    w = _world(planets, step=100)
    # Symmetry: the answer doesn't depend on set ordering.
    a = pred.is_winning_state_if_owned(w, my_id=0, opp_id=1,
                                       extra_planet_ids={1, 2, 3})
    b = pred.is_winning_state_if_owned(w, my_id=0, opp_id=1,
                                       extra_planet_ids={3, 2, 1})
    assert a == b

    # Idempotent under duplicates (sets dedupe by definition, but
    # double-check the function doesn't accidentally iterate twice).
    c = pred.is_winning_state_if_owned(w, my_id=0, opp_id=1,
                                       extra_planet_ids={1, 2, 3, 1, 2})
    assert c == a

    # And: capturing ALL opp planets must remove opp entirely from the
    # picture, so the predicate becomes "adv > 0 after sweep".
    assert pred.is_winning_state_if_owned(w, my_id=0, opp_id=1,
                                          extra_planet_ids={1, 2}) is True


def test_opp_id_2p_returns_minus_one_in_4p():
    """opp_id_2p contract: -1 in 4P, so the predicate never runs.

    is_winning_state_if_owned itself does NOT short-circuit on opp_id=-1
    (it would compute against a non-existent opponent); the gate lives
    in `opp_id_2p`. Document the contract here so a future refactor
    doesn't silently move it.
    """
    from agents.buildup_planner.predicates import opp_id_2p
    planets = [
        (0, 0, 50.0, 30.0, 3.0, 10, 2),
        (1, 1, 50.0, 70.0, 3.0, 10, 2),
        (2, 2, 30.0, 50.0, 3.0, 10, 2),
        (3, 3, 70.0, 50.0, 3.0, 10, 2),
    ]
    w = _world(planets, step=0)
    assert opp_id_2p(w, me=0) == -1
    # 2P (only one other owned planet) → unique opponent id.
    planets_2p = [
        (0, 0, 50.0, 30.0, 3.0, 10, 2),
        (1, 1, 50.0, 70.0, 3.0, 10, 2),
        (2, -1, 30.0, 50.0, 3.0, 5, 1),
    ]
    w2 = _world(planets_2p, step=0)
    assert opp_id_2p(w2, me=0) == 1
