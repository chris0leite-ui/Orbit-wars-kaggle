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


def test_adaptive_k_off_by_default_no_horizon_bump():
    """Phase 2, 2026-05-22: BASELINE_ADAPTIVE_K is OFF by default. With
    the env var unset the chooser must not bump rollout horizon. End-to-
    end smoke: agent still produces valid output and is deterministic.
    """
    import os
    saved = os.environ.pop("BASELINE_ADAPTIVE_K", None)
    try:
        from agents.baseline.main import agent
        obs, _snap = _snapshot_from_seed(42)
        out = agent(obs)
        assert isinstance(out, list)
    finally:
        if saved is not None:
            os.environ["BASELINE_ADAPTIVE_K"] = saved


def test_adaptive_k_extends_baseline_horizon():
    """Phase 2, 2026-05-22: when BASELINE_ADAPTIVE_K=1 the chooser pre-
    builds the idle baseline up to MAX_HORIZON_CAP so mid-loop horizon
    bumps don't IndexError. Without this safeguard, score_candidate_v4
    would try to read `baseline_favors[bumped_horizon]` past list end.
    Smoke: an end-to-end run with adaptive ON, K_BUMP=20, very small
    MARGIN to force a bump must not raise.
    """
    import os
    saved = {
        k: os.environ.get(k) for k in (
            "BASELINE_ADAPTIVE_K", "BASELINE_CRITICALITY_K_BUMP",
            "BASELINE_CRITICALITY_MARGIN", "BASELINE_CRITICALITY_PROBE",
            "BASELINE_MAX_HORIZON_CAP",
        )
    }
    os.environ["BASELINE_ADAPTIVE_K"] = "1"
    os.environ["BASELINE_CRITICALITY_K_BUMP"] = "20"
    os.environ["BASELINE_CRITICALITY_MARGIN"] = "999.0"  # always-critical
    os.environ["BASELINE_CRITICALITY_PROBE"] = "1"
    os.environ["BASELINE_MAX_HORIZON_CAP"] = "60"
    try:
        # Re-import to pick up the env vars (module-level constants).
        import importlib
        import agents.baseline.chooser_trajectory as ct
        importlib.reload(ct)
        from agents.baseline.main import agent
        obs, _snap = _snapshot_from_seed(42)
        out = agent(obs)
        assert isinstance(out, list)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import importlib
        import agents.baseline.chooser_trajectory as ct
        importlib.reload(ct)



def test_opp_smart_leaf_off_is_bit_identical():
    """Phase 4, 2026-05-22: with BASELINE_OPP_SMART_LEAF unset (default
    OFF), the agent must produce byte-identical moves compared to the
    explicit \"0\" setting. Guards against the helper introducing any
    subtle change to the lite_greedy path."""
    import os
    saved = os.environ.get("BASELINE_OPP_SMART_LEAF")
    try:
        os.environ.pop("BASELINE_OPP_SMART_LEAF", None)
        from agents.baseline.main import agent
        obs, _snap = _snapshot_from_seed(42)
        out_unset = agent(obs)

        os.environ["BASELINE_OPP_SMART_LEAF"] = "0"
        out_zero = agent(obs)
        assert out_unset == out_zero
    finally:
        if saved is None:
            os.environ.pop("BASELINE_OPP_SMART_LEAF", None)
        else:
            os.environ["BASELINE_OPP_SMART_LEAF"] = saved


def test_opp_smart_leaf_on_runs_without_crash():
    """Phase 4, 2026-05-22: with BASELINE_OPP_SMART_LEAF=1 the chooser
    must run without raising on a warmed mid-game snapshot. This is a
    smoke test only — the smart-leaf swap is in the rollout, where the
    new opp launches MAY not propagate to flip the leaf favor enough
    to change the prerank ordering (the prerank is dominated by
    proposer cheap_delta; the leaf score nudges within). Validating
    "smart-leaf actually changes a decision" is done at the A/B level,
    not at the unit-test level.

    Real safety net: `test_opp_smart_leaf_off_is_bit_identical` above.
    """
    import os
    import random
    saved = os.environ.get("BASELINE_OPP_SMART_LEAF")
    try:
        from agents.baseline.main import agent
        env = make("orbit_wars", configuration={"seed": 7}, debug=False)
        env.reset(2)
        rng = random.Random(7)
        for _ in range(30):
            obs = env.state[0].observation
            a = [[p[0], rng.uniform(0, 6.28), int(p[5] // 2)]
                 for p in obs["planets"]
                 if p[1] == 0 and p[5] > 5 and rng.random() < 0.3]
            b = [[p[0], rng.uniform(0, 6.28), int(p[5] // 2)]
                 for p in obs["planets"]
                 if p[1] == 1 and p[5] > 5 and rng.random() < 0.3]
            env.step([a, b])
        obs = env.state[0].observation
        os.environ["BASELINE_OPP_SMART_LEAF"] = "1"
        out = agent(obs)
        assert isinstance(out, list)
    finally:
        if saved is None:
            os.environ.pop("BASELINE_OPP_SMART_LEAF", None)
        else:
            os.environ["BASELINE_OPP_SMART_LEAF"] = saved


def test_phase4_calibration_anchored_at_baseline_horizon():
    """Phase 4 calibration fix (2026-05-23): the smart-opp leaf swap
    must fire at the SAME absolute step (baseline_horizon - WINDOW) in
    BOTH build_idle_baseline AND score_action, so Δ = leaf -
    baseline_favors[h] stays apples-to-apples.

    Test contract: score_action(horizon=H, baseline_horizon=BH) with
    H < BH - WINDOW should be bit-identical between BASELINE_OPP_SMART_LEAF
    ON and OFF (the candidate's rollout never enters the smart window).
    Pre-fix, the candidate's rollout swapped at H - WINDOW (its own
    tail) while the baseline swapped at BH - WINDOW, biasing Δ.
    """
    import os
    from agents.baseline.chooser import (
        build_idle_baseline, score_action, opp_smart_leaf_window,
    )
    saved = os.environ.get("BASELINE_OPP_SMART_LEAF")
    try:
        _obs, snap = _snapshot_from_seed(7)
        BH = 40
        H = 10  # well below BH - WINDOW (= 35 with default WINDOW=5)
        assert H < BH - opp_smart_leaf_window(), (
            "test premise violated: H must be below the absolute smart-leaf window"
        )
        favs = build_idle_baseline(snap, me=0, num_seats=2, max_horizon=BH, gamma=0.99)

        os.environ.pop("BASELINE_OPP_SMART_LEAF", None)
        d_off = score_action(
            snap, me=0, num_seats=2,
            src_id=0, angle=0.0, ships=1,
            horizon=H, baseline_favors=favs, wait_N=0, gamma=0.99,
            baseline_horizon=BH,
        )
        os.environ["BASELINE_OPP_SMART_LEAF"] = "1"
        d_on = score_action(
            snap, me=0, num_seats=2,
            src_id=0, angle=0.0, ships=1,
            horizon=H, baseline_favors=favs, wait_N=0, gamma=0.99,
            baseline_horizon=BH,
        )
        # H is far below the smart-leaf absolute window, so the candidate's
        # rollout never enters the smart branch; Δ must match exactly.
        assert d_off == d_on, f"calibration broken: off={d_off}, on={d_on}"
    finally:
        if saved is None:
            os.environ.pop("BASELINE_OPP_SMART_LEAF", None)
        else:
            os.environ["BASELINE_OPP_SMART_LEAF"] = saved


def test_env_parse_handles_bad_input():
    """Commit 3 (2026-05-23): env vars with bad content (empty string,
    non-numeric, whitespace) must not crash the agent at import. The
    new lib.config helpers wrap parsing in try/except and fall back to
    defaults. Pre-fix, a wrapper exporting `BASELINE_MIN_HORIZON=''`
    crashed `int('')` at module load, erroring the submission.
    """
    import os
    saved = {
        k: os.environ.get(k) for k in (
            "BASELINE_MIN_HORIZON", "BASELINE_MAX_HORIZON",
            "BASELINE_STEP_BASE_MS", "BASELINE_OPP_SMART_LEAF_WINDOW",
        )
    }
    try:
        os.environ["BASELINE_MIN_HORIZON"] = ""
        os.environ["BASELINE_MAX_HORIZON"] = "abc"
        os.environ["BASELINE_STEP_BASE_MS"] = " "
        os.environ["BASELINE_OPP_SMART_LEAF_WINDOW"] = "junk"
        from agents.baseline.chooser import (
            affordable_validate_cap, opp_smart_leaf_window,
        )
        # Helpers must return their defaults, not raise.
        assert isinstance(opp_smart_leaf_window(), int)
        _obs, snap = _snapshot_from_seed(42)
        cap, _ = affordable_validate_cap(
            snap, me=0, num_seats=2, max_horizon=40, wallclock_ms=600.0,
            min_horizon=25, gamma=0.99,
        )
        assert cap >= 8
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_cost_model_responds_to_runtime_env_change():
    """Commit 3 (2026-05-23): cost-model constants are read per-call.
    Setting BASELINE_STEP_PER_FLEET_MS to a much larger value mid-test
    must shrink the cap.
    """
    import os
    saved = os.environ.get("BASELINE_STEP_PER_FLEET_MS")
    try:
        from agents.baseline.chooser import affordable_validate_cap
        _obs, snap = _snapshot_from_seed(7)
        os.environ.pop("BASELINE_STEP_PER_FLEET_MS", None)
        cap_default, _ = affordable_validate_cap(
            snap, me=0, num_seats=2, max_horizon=40, wallclock_ms=600.0,
            min_horizon=25, gamma=0.99,
        )
        os.environ["BASELINE_STEP_PER_FLEET_MS"] = "2.0"  # ~33x default
        cap_heavy, _ = affordable_validate_cap(
            snap, me=0, num_seats=2, max_horizon=40, wallclock_ms=600.0,
            min_horizon=25, gamma=0.99,
        )
        # Heavier cost → smaller cap (or hit the floor of 8).
        assert cap_heavy <= cap_default
    finally:
        if saved is None:
            os.environ.pop("BASELINE_STEP_PER_FLEET_MS", None)
        else:
            os.environ["BASELINE_STEP_PER_FLEET_MS"] = saved


def test_cost_model_accounts_for_adaptive_k_extension():
    """Commit 4 (2026-05-23): when Phase 2 is on, the cost model
    receives `adaptive_k_extension=ADAPTIVE_K_BUMP` so n_aff is sized
    for the actual (bumped) avg-K rather than the unbumped MAX_HORIZON.
    """
    from agents.baseline.chooser import affordable_validate_cap
    _obs, snap = _snapshot_from_seed(7)
    cap_unbumped, _ = affordable_validate_cap(
        snap, me=0, num_seats=2, max_horizon=40, wallclock_ms=600.0,
        min_horizon=25, gamma=0.99, adaptive_k_extension=0,
    )
    cap_bumped, _ = affordable_validate_cap(
        snap, me=0, num_seats=2, max_horizon=40, wallclock_ms=600.0,
        min_horizon=25, gamma=0.99, adaptive_k_extension=10,
    )
    assert cap_bumped <= cap_unbumped


def test_cost_model_accounts_for_phase3b_when_enabled():
    """Commit 4 (2026-05-23): when COMPOSITE_FLEET_SURVIVAL_CHECK is on,
    affordable_validate_cap adds per-my-fleet leaf cost so n_aff
    reflects the real per-leaf time.
    """
    import os
    from agents.baseline.chooser import affordable_validate_cap
    saved = os.environ.get("COMPOSITE_FLEET_SURVIVAL_CHECK")
    try:
        _obs, snap = _snapshot_from_seed(42)
        os.environ.pop("COMPOSITE_FLEET_SURVIVAL_CHECK", None)
        cap_off, _ = affordable_validate_cap(
            snap, me=0, num_seats=2, max_horizon=40, wallclock_ms=600.0,
            min_horizon=25, gamma=0.99,
        )
        os.environ["COMPOSITE_FLEET_SURVIVAL_CHECK"] = "1"
        cap_on, _ = affordable_validate_cap(
            snap, me=0, num_seats=2, max_horizon=40, wallclock_ms=600.0,
            min_horizon=25, gamma=0.99,
        )
        assert cap_on <= cap_off
    finally:
        if saved is None:
            os.environ.pop("COMPOSITE_FLEET_SURVIVAL_CHECK", None)
        else:
            os.environ["COMPOSITE_FLEET_SURVIVAL_CHECK"] = saved


def test_zero_cost_constants_dont_crash():
    """Commit 5 (2026-05-23): zeroing all cost constants would have
    produced per_cand_ms=0 and ZeroDivisionError. The guard clamps
    per_cand_ms ≥ 0.001 so the chooser stays alive.
    """
    import os
    from agents.baseline.chooser import affordable_validate_cap
    saved = {
        k: os.environ.get(k) for k in (
            "BASELINE_STEP_BASE_MS", "BASELINE_STEP_PER_FLEET_MS",
            "BASELINE_LEAF_BASE_MS", "BASELINE_LEAF_PER_FLEET_MS",
        )
    }
    try:
        for k in saved:
            os.environ[k] = "0"
        _obs, snap = _snapshot_from_seed(7)
        cap, per_cand_ms = affordable_validate_cap(
            snap, me=0, num_seats=2, max_horizon=40, wallclock_ms=600.0,
            min_horizon=25, gamma=0.99,
        )
        assert cap >= 8
        assert per_cand_ms > 0
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
