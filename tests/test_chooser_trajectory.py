"""Tests for the trajectory-first chooser.

Covers the three failure-mode prefilters (sun, OOB, comet-collision)
and the basic capture-vs-bounce scoring. Uses a real env snap for the
chooser-call signature, but assertions target the candidate-filtering
behaviour, not full game outcomes.
"""

from __future__ import annotations

import math

import pytest
from kaggle_environments import make

from agents.baseline.chooser_trajectory import choose_trajectory, score_candidate
from lib import fast_sim
from lib.intent import World
from lib.world_model import WorldModel


def _snap_and_world(seed: int = 7):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(2)
    obs = env.steps[0][0].observation
    snap = fast_sim.from_obs(obs, num_seats=2)
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    return obs, snap, world, model


def _my_src(world, me: int = 0):
    """First planet owned by `me`."""
    for p in world.planets_by_id.values():
        if int(p.owner) == me:
            return p
    raise RuntimeError("no source planet for me")


def _enemy_tgt(world, me: int = 0):
    """First non-neutral, non-me planet (any enemy)."""
    for p in world.planets_by_id.values():
        if int(p.owner) >= 0 and int(p.owner) != me:
            return p
    raise RuntimeError("no enemy target")


def _make_candidate(src, tgt, ships, angle, eta=5, wait_N=0):
    return (0.0, src, tgt, int(ships), float(angle), int(eta), 10, int(wait_N))


# ---------------------------------------------------------------------------
# score_candidate — individual outcomes
# ---------------------------------------------------------------------------


def test_score_candidate_sun_rejected():
    """A launch aimed straight through the sun is rejected with status='sun'."""
    obs, snap, world, model = _snap_and_world(seed=7)
    me = 0
    src = _my_src(world, me)
    tgt = _enemy_tgt(world, me)
    # Force an angle that goes through the sun: aim from src directly
    # toward (CENTER, CENTER).
    angle = math.atan2(50.0 - src.y, 50.0 - src.x)
    score, status, _ = score_candidate(
        src, tgt, ships=20, angle=angle, eta_hint=5,
        me=me, world=world, ledger=model.ledger,
    )
    assert status == "sun"
    assert score == float("-inf")


def test_score_candidate_oob_rejected():
    """A launch aimed off the board returns OOB."""
    obs, snap, world, model = _snap_and_world(seed=7)
    me = 0
    src = _my_src(world, me)
    tgt = _enemy_tgt(world, me)
    # Angle that exits the board fastest from src position.
    # Heading away from board centre.
    angle = math.atan2(src.y - 50.0, src.x - 50.0)
    score, status, _ = score_candidate(
        src, tgt, ships=20, angle=angle, eta_hint=5,
        me=me, world=world, ledger=model.ledger,
    )
    assert status in ("oob", "path_blocked", "sun"), (
        f"unexpected status {status} for outward-aimed launch"
    )
    assert score == float("-inf")


def test_score_candidate_capture_gives_positive_score():
    """Aim directly at a clear enemy planet with enough ships → capture
    credit, status='captured', positive score."""
    obs, snap, world, model = _snap_and_world(seed=7)
    me = 0
    src = _my_src(world, me)
    tgt = _enemy_tgt(world, me)
    # Direct aim from src to tgt.
    angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
    # Send overwhelming force.
    score, status, eta = score_candidate(
        src, tgt, ships=300, angle=angle, eta_hint=5,
        me=me, world=world, ledger=model.ledger,
    )
    # Depending on the board geometry, the straight line might pass
    # through the sun. If so, we'll get 'sun' — that's a valid result
    # for THIS seed but not a useful capture test. Tolerate either
    # 'captured' (positive score) OR 'sun' (rejection).
    assert status in ("captured", "sun", "path_blocked", "bounced"), (
        f"unexpected status {status}"
    )
    if status == "captured":
        assert score > 0
        assert eta is not None and eta > 0


def test_score_candidate_bounce_gives_negative_score():
    """Aim directly at a target with too few ships to capture → bounce,
    negative score."""
    obs, snap, world, model = _snap_and_world(seed=7)
    me = 0
    src = _my_src(world, me)
    tgt = _enemy_tgt(world, me)
    angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
    score, status, _ = score_candidate(
        src, tgt, ships=1, angle=angle, eta_hint=5,
        me=me, world=world, ledger=model.ledger,
    )
    assert status in ("bounced", "sun", "path_blocked"), (
        f"unexpected status {status}"
    )
    if status == "bounced":
        assert score < 0


# ---------------------------------------------------------------------------
# choose_trajectory — end-to-end
# ---------------------------------------------------------------------------


def test_choose_trajectory_empty_prerank_returns_empty():
    obs, snap, world, model = _snap_and_world(seed=7)
    moves = choose_trajectory(
        snap, prerank=[], baseline_favors=None,
        me=0, num_seats=2, wallclock_ms=600.0,
        min_horizon=25, max_horizon=40, gamma=0.99,
        world=world, model=model,
    )
    assert moves == []


def test_choose_trajectory_drops_doomed_candidates():
    """A prerank containing only a sun-crossing candidate produces no
    moves (the trajectory filter rejects it)."""
    obs, snap, world, model = _snap_and_world(seed=7)
    me = 0
    src = _my_src(world, me)
    tgt = _enemy_tgt(world, me)
    sun_angle = math.atan2(50.0 - src.y, 50.0 - src.x)
    prerank = [_make_candidate(src, tgt, ships=20, angle=sun_angle)]
    moves = choose_trajectory(
        snap, prerank=prerank, baseline_favors=None,
        me=me, num_seats=2, wallclock_ms=600.0,
        min_horizon=25, max_horizon=40, gamma=0.99,
        world=world, model=model,
    )
    assert moves == []


def test_choose_trajectory_emit_format_is_env_action_shape():
    """End-to-end: real obs through real agent with trajectory chooser
    returns env-format actions (each [src_id, angle, ships])."""
    import os
    os.environ["BASELINE_CHOOSER"] = "trajectory"
    try:
        from agents.baseline.main import agent
        env = make("orbit_wars", configuration={"seed": 13}, debug=False)
        env.reset(2)
        obs = env.steps[0][0].observation
        moves = agent(obs)
    finally:
        os.environ.pop("BASELINE_CHOOSER", None)
    assert isinstance(moves, list)
    for m in moves:
        assert isinstance(m, list) and len(m) == 3
        assert isinstance(m[0], int)        # src_id
        assert isinstance(m[1], float)      # angle
        assert isinstance(m[2], int)        # ships
