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
    cap, per_cand_ms = affordable_validate_cap(
        snap, me=0, num_seats=2, max_horizon=10, wallclock_ms=50.0,
        min_horizon=5, gamma=0.99,
    )
    assert cap >= 8
    assert per_cand_ms > 0.0


def test_affordable_cap_shrinks_under_composite_head():
    """The cap probe measures per-leaf cost — composite_capture_value is
    heavier per call than favor when fleets are in flight (builds a
    World + ray-casts every fleet). Under composite on a board WITH
    in-flight fleets the cap should be smaller than under favor at the
    same wallclock budget. Documents the 2026-05-17 timing fix.

    NB: on a board with NO in-flight fleets, composite short-circuits
    and is essentially as cheap as favor. Test must construct fleets
    to exercise the heavy path.
    """
    import os
    import statistics
    from lib.fast_sim import step as fs_step
    obs, snap = _snapshot_from_seed(7)

    # Launch a fleet so composite hits its World/WorldModel/ray-cast
    # path. Action shape: [[me_actions], [opp_actions]]. Find first
    # owned planet from obs, launch 30 ships at angle 0.
    my_planets = [p for p in obs["planets"] if p[1] == 0]
    assert my_planets, "test prereq: seed=7 P0 must own a planet at step 0"
    src_id = int(my_planets[0][0])
    snap = fs_step(snap, [[[src_id, 0.0, 30]], []], in_place=True)
    # Step a few more times so the fleet is mid-flight (not just-launched).
    for _ in range(5):
        snap = fs_step(snap, [[], []], in_place=True)

    args = dict(me=0, num_seats=2, max_horizon=40, wallclock_ms=600.0,
                min_horizon=25, gamma=0.99)

    def median_cap(env_value):
        if env_value is None:
            os.environ.pop("BASELINE_VALUE_HEAD", None)
        else:
            os.environ["BASELINE_VALUE_HEAD"] = env_value
        # Warmup so module-import + WorldModel-build aren't on the path.
        affordable_validate_cap(snap, **args)
        samples = [affordable_validate_cap(snap, **args)[0] for _ in range(3)]
        return statistics.median(samples)

    try:
        cap_favor = median_cap(None)
        cap_composite = median_cap("composite")
    finally:
        os.environ.pop("BASELINE_VALUE_HEAD", None)

    assert cap_composite <= cap_favor, (
        f"composite cap ({cap_composite}) should be <= favor cap "
        f"({cap_favor}) on a board with in-flight fleets — composite "
        f"leaf is heavier (builds World + ray-casts every fleet)"
    )


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
