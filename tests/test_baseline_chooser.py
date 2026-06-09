"""Unit tests for agents/baseline/chooser."""

from __future__ import annotations

from kaggle_environments import make

import time

from agents.baseline.chooser import (
    HARDCAP_BAIL_SENTINEL,
    affordable_validate_cap,
    build_idle_baseline,
    choose,
    opp_actions_for_snap,
    score_action,
)
from lib.fast_sim import from_obs as fs_from_obs


def _snapshot_from_seed(seed: int = 42):
    """Spin up a real env at step 0 and return the fast_sim snapshot."""
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(2)
    obs = env.steps[0][0].observation
    return obs, fs_from_obs(obs, num_seats=2)


def test_opp_actions_returns_per_seat_list():
    _obs, snap = _snapshot_from_seed(7)
    actions = opp_actions_for_snap(snap, me=0, num_seats=2)
    assert len(actions) == 2
    assert actions[0] == []  # me-slot is always empty
    assert isinstance(actions[1], list)


def test_build_idle_baseline_length_matches_horizon():
    _obs, snap = _snapshot_from_seed(7)
    h = 10
    favs = build_idle_baseline(snap, me=0, num_seats=2, max_horizon=h, gamma=0.99)
    assert len(favs) == h + 1  # one entry per [0..h]
    assert all(isinstance(v, float) for v in favs)


def test_score_action_no_op_when_horizon_zero():
    """A 0-step rollout returns leaf at the current state minus baseline[0]
    which is the same value — Δ must be 0.
    """
    _obs, snap = _snapshot_from_seed(7)
    favs = build_idle_baseline(snap, me=0, num_seats=2, max_horizon=5, gamma=0.99)
    # An impossible launch (src_id that doesn't exist), but wait_N > horizon
    # makes the action never fire, so Δ = leaf(idle, 0 steps) - favs[0] = 0
    delta = score_action(
        snap, me=0, num_seats=2,
        src_id=999, angle=0.0, ships=1,
        horizon=0, baseline_favors=favs, wait_N=0, gamma=0.99,
    )
    assert delta == 0.0


def test_affordable_cap_has_floor_of_eight():
    """Even with an extreme budget the cap is bounded below by 8."""
    _obs, snap = _snapshot_from_seed(7)
    cap = affordable_validate_cap(
        snap, num_seats=2, max_horizon=10, wallclock_ms=50.0, min_horizon=5,
    )
    assert cap >= 8


def test_choose_empty_prerank_returns_empty():
    _obs, snap = _snapshot_from_seed(7)
    favs = build_idle_baseline(snap, me=0, num_seats=2, max_horizon=10, gamma=0.99)
    assert choose(
        snap, prerank=[], baseline_favors=favs,
        me=0, num_seats=2, wallclock_ms=600.0,
        min_horizon=5, max_horizon=10, gamma=0.99,
    ) == []


def test_score_action_hardcap_bail_returns_sentinel_fast():
    """Rule 38 fix-verification: with hard_deadline already in the past,
    score_action must return HARDCAP_BAIL_SENTINEL before completing any
    rollout step. Reproduces the failure state (rollout runs to
    completion past deadline) and verifies the fix engages."""
    _obs, snap = _snapshot_from_seed(7)
    favs = build_idle_baseline(snap, me=0, num_seats=2, max_horizon=10, gamma=0.99)

    t0 = time.perf_counter()
    delta = score_action(
        snap, me=0, num_seats=2,
        src_id=0, angle=0.0, ships=1,
        horizon=10, baseline_favors=favs, wait_N=0, gamma=0.99,
        hard_deadline=time.perf_counter() - 1.0,  # already past
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert delta == HARDCAP_BAIL_SENTINEL
    assert elapsed_ms < 5.0, (
        f"hardcap bail took {elapsed_ms:.1f}ms — should fire on first iter"
    )


def test_choose_emit_format_is_env_action_shape():
    """End-to-end: run the agent on a real obs and check return shape."""
    from agents.baseline.main import agent

    obs, _snap = _snapshot_from_seed(42)
    out = agent(obs)
    assert isinstance(out, list)
    for move in out:
        assert isinstance(move, list)
        assert len(move) == 3
        sid, angle, ships = move
        assert isinstance(sid, int)
        assert isinstance(angle, float)
        assert isinstance(ships, int)
        assert ships >= 1
