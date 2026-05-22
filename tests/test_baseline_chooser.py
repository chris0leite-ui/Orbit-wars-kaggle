"""Unit tests for agents/baseline/chooser."""

from __future__ import annotations

from kaggle_environments import make

from agents.baseline.chooser import (
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


def test_affordable_cap_is_deterministic_across_calls():
    """Phase 1, 2026-05-22: affordable_validate_cap no longer probes
    wallclock — two calls with identical args must return identical
    (cap, per_cand_ms). Pre-Phase-1 this assertion fails when the
    machine has any timing jitter at all (the probe path used
    time.perf_counter() to measure per-step and per-leaf cost).
    """
    _obs, snap = _snapshot_from_seed(7)
    out_a = affordable_validate_cap(
        snap, me=0, num_seats=2,
        max_horizon=40, wallclock_ms=600.0, min_horizon=25, gamma=0.99,
    )
    out_b = affordable_validate_cap(
        snap, me=0, num_seats=2,
        max_horizon=40, wallclock_ms=600.0, min_horizon=25, gamma=0.99,
    )
    assert out_a == out_b


def test_agent_is_deterministic_across_calls():
    """Phase 1, 2026-05-22: end-to-end determinism on a real obs. The
    same observation passed twice must produce the same move list. Pre-
    Phase-1, the wallclock-based candidate cap drifted across calls and
    different candidates were evaluated → different chosen move. The
    diff_v3 trace (2026-05-22) saw 128/210 turns diverge between
    repeat runs of the same seed.
    """
    from agents.baseline.main import agent

    obs, _snap = _snapshot_from_seed(42)
    out_a = agent(obs)
    out_b = agent(obs)
    assert out_a == out_b
