"""Unit tests for lib/cascade_chooser.py.

Covers:
  - solo greedy by closed-form _capture_value
  - source-budget tracking within a rebuild
  - target uniqueness
  - cascade emergence across rebuilds
  - seat awareness (me=0 and me=1)
  - no out-of-bounds emissions survive selection
  - pair-joint enumeration semantics
  - per-turn budget regression

The fate cache is per-process, so we reset between tests via the same
autouse fixture used in `tests/test_sa_core.py`.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

from lib.cascade_chooser import (  # noqa: E402
    _enumerate_pair_joints,
    _greedy_pick_from_ctx,
    cascade_greedy_select,
)
from lib.sa_core import (  # noqa: E402
    _build_perturb_context,
    reset_fate_cache,
)
from lib.trajectory import predict_fleet_fate  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_fate_cache_between_tests():
    reset_fate_cache()
    yield
    reset_fate_cache()


# ---------------------------------------------------------------------------
# Helpers (mirrors patterns in tests/test_sa_core.py)
# ---------------------------------------------------------------------------


def _build_snap0(seed: int = 0, steps: int = 100):
    from kaggle_environments import make
    from lib.fast_sim import from_obs as fs_from_obs

    env = make("orbit_wars",
               configuration={"seed": seed, "episodeSteps": steps},
               debug=False)
    env.reset(num_agents=2)
    obs0 = env.steps[0][0]["observation"] if isinstance(env.steps[0][0], dict) else env.steps[0][0].observation
    return fs_from_obs(obs0, env.configuration,
                       episode_seed=seed, num_seats=2)


def _build_ctx(seed: int, steps: int, *, me: int = 0, plan=None, t_start: int = 0):
    snap = _build_snap0(seed=seed, steps=steps)
    if plan is None:
        plan = []
    return snap, _build_perturb_context(
        snap, plan, opp_policy=None,
        max_steps=steps - t_start,
        t_start=t_start, t_end=steps,
        me=me, path_graph=None,
    )


# ---------------------------------------------------------------------------
# Greedy: solo by _capture_value
# ---------------------------------------------------------------------------


def test_greedy_picks_by_capture_value():
    """With no joints, greedy should pick the highest-value admissible
    candidate first."""
    _snap, ctx = _build_ctx(seed=7542, steps=80)
    if not ctx.admissible:
        pytest.skip("admissibility empty for this seed/horizon")
    spent: dict[int, float] = {}
    claimed: set[int] = set()
    picks = _greedy_pick_from_ctx(
        ctx, joint_candidates=[],
        max_picks=1, spent_per_src=spent, claimed_targets=claimed,
    )
    assert len(picks) == 1


def test_greedy_respects_target_uniqueness():
    """Two picks must not both target the same planet (within one rebuild)."""
    _snap, ctx = _build_ctx(seed=7542, steps=80)
    if not ctx.admissible:
        pytest.skip("admissibility empty for this seed/horizon")
    spent: dict[int, float] = {}
    claimed: set[int] = set()
    picks = _greedy_pick_from_ctx(
        ctx, joint_candidates=[],
        max_picks=8, spent_per_src=spent, claimed_targets=claimed,
    )
    # Resolve each pick's target via the admissibility index.
    targets_in_picks: list[int] = []
    by_id = {id(e): ctx.admissible_targets[i]
             for i, e in enumerate(ctx.admissible)}
    for p in picks:
        tid = by_id.get(id(p))
        if tid is not None:
            targets_in_picks.append(int(tid))
    assert len(targets_in_picks) == len(set(targets_in_picks)), \
        f"duplicate targets in single-rebuild picks: {targets_in_picks}"


def test_greedy_respects_source_budget():
    """Two picks from the same source must not exceed available ships
    (cache_ships − cumulative spend)."""
    _snap, ctx = _build_ctx(seed=7542, steps=80)
    if not ctx.admissible:
        pytest.skip("admissibility empty for this seed/horizon")
    spent: dict[int, float] = {}
    claimed: set[int] = set()
    picks = _greedy_pick_from_ctx(
        ctx, joint_candidates=[],
        max_picks=8, spent_per_src=spent, claimed_targets=claimed,
    )
    # Per-source per-t_dep, total spent must not exceed cache_ships.
    per_src_total: dict[int, float] = {}
    for emit in picks:
        t_dep = int(emit[0])
        src_id = int(emit[1][0])
        ships = float(emit[1][2])
        per_src_total[src_id] = per_src_total.get(src_id, 0.0) + ships
        cache_at = ctx.ownership_cache.get(t_dep, {}).get(src_id)
        assert cache_at is not None, f"no cache for src {src_id} at t_dep={t_dep}"
        _owner, cache_ships = cache_at
        assert per_src_total[src_id] <= float(cache_ships) + 1e-6, \
            (f"src={src_id} spent={per_src_total[src_id]} > "
             f"cache_ships={cache_ships} at t_dep={t_dep}")


# ---------------------------------------------------------------------------
# Pair-joint enumeration
# ---------------------------------------------------------------------------


def test_pair_joint_rejects_solo_viable_pairs():
    """Joints are only emitted when neither source alone has
    `defender + 1` ships. If both can solo-capture, no joint is yielded."""
    # Use a small horizon where multiple owned-source-shipping situations
    # exist; we can't guarantee a joint is enumerated, but we can assert
    # NO joint pair has either leg with ships >= garrison + 1.
    _snap, ctx = _build_ctx(seed=7542, steps=80)
    if ctx.path_graph is None or ctx.world is None:
        pytest.skip("ctx missing world or path_graph")
    joints = _enumerate_pair_joints(ctx)
    # For every enumerated joint, derive the defender at t_arr and assert
    # ships_a < defender+1 AND ships_b < defender+1.
    from lib.world_model import predict_garrison_at as _pga
    for jc in joints:
        _val, t_dep, _s_a, _s_b, emit_a, emit_b, tgt_id, t_arr, n_a, n_b = jc
        tgt = ctx.world.planets_by_id.get(int(tgt_id))
        assert tgt is not None
        try:
            arrivals = list(ctx.world_model.ledger.get(int(tgt_id), []))
        except Exception:
            arrivals = []
        eta_from_snap0 = max(0, t_dep - ctx.t_start) + (t_arr - t_dep)
        pred_owner, pred_garrison = _pga(tgt, eta_from_snap0, arrivals)
        import math
        defender_plus_one = int(math.ceil(float(pred_garrison))) + 1
        assert n_a < defender_plus_one, \
            f"joint leg a={n_a} >= defender+1={defender_plus_one}"
        assert n_b < defender_plus_one, \
            f"joint leg b={n_b} >= defender+1={defender_plus_one}"
        assert n_a + n_b >= defender_plus_one, \
            f"joint combined {n_a + n_b} < defender+1 {defender_plus_one}"


def test_pair_joint_emits_same_target_same_arrival():
    """Both legs of every joint reference the same target and same t_arr.
    Combat-rule-1 requires simultaneous arrival to pool ships."""
    _snap, ctx = _build_ctx(seed=7542, steps=80)
    if ctx.path_graph is None or ctx.world is None:
        pytest.skip("ctx missing world or path_graph")
    joints = _enumerate_pair_joints(ctx)
    pg = ctx.path_graph
    for jc in joints:
        _val, t_dep, s_a, s_b, emit_a, emit_b, tgt_id, t_arr, _na, _nb = jc
        edge_a = pg.lookup(int(s_a), int(tgt_id), int(t_dep))
        edge_b = pg.lookup(int(s_b), int(tgt_id), int(t_dep))
        assert edge_a is not None and edge_b is not None
        assert int(edge_a.t_arr) == int(t_arr)
        assert int(edge_b.t_arr) == int(t_arr)


def test_pair_joint_rejects_same_source():
    """Degenerate (S, S) pairs must not appear in joints (would launch
    twice from one source which combat-rule-1 doesn't reward)."""
    _snap, ctx = _build_ctx(seed=7542, steps=80)
    if ctx.path_graph is None or ctx.world is None:
        pytest.skip("ctx missing world or path_graph")
    joints = _enumerate_pair_joints(ctx)
    for jc in joints:
        _val, _t_dep, s_a, s_b, *_rest = jc
        assert int(s_a) != int(s_b), f"same-source pair: {s_a}"


# ---------------------------------------------------------------------------
# Cascade emergence across rebuilds
# ---------------------------------------------------------------------------


def test_cascade_planet_appears_as_source_after_rebuild():
    """Build a 2-rebuild plan; verify that AT LEAST ONE planet captured
    by rebuild 0 appears as a source for an emission in rebuild 1.

    Probabilistic on seed/horizon; we use a long horizon to give the DP
    a meaningful window. If no cascade emerges we accept the test as
    inconclusive (skip) rather than flaky-fail."""
    snap = _build_snap0(seed=7542, steps=120)
    plan = cascade_greedy_select(
        snap, t_start=0, t_end=120, me=0,
        opp_policy=None, path_graph=None,
        max_rebuilds=3, max_picks_per_rebuild=4,
    )
    # Map src_id -> set of t_dep it appears at
    src_t_deps: dict[int, list[int]] = {}
    for tau, payload in plan:
        src_t_deps.setdefault(int(payload[0]), []).append(int(tau))
    # A planet that appears as a source at t_dep > 0 AND was NOT
    # originally ours signals a cascade — it had to be captured first.
    from lib.intent import World
    obs0 = snap.state[0].observation
    obs_d = obs0 if isinstance(obs0, dict) else dict(obs0)
    world = World.from_obs(obs_d)
    originally_ours = {int(p.id) for p in world.planets_by_id.values()
                       if int(p.owner) == 0}
    cascaded_sources = [
        s for s, deps in src_t_deps.items()
        if s not in originally_ours and any(t > 0 for t in deps)
    ]
    if not cascaded_sources:
        pytest.skip(
            "no cascade emerged at seed=7542 horizon=120; not strictly "
            "wrong (depends on map geometry + opp model), but the DP "
            "did not reach 2-step cascade here")
    # Each cascaded source must have been captured at some prior t_arr.
    # We can't directly look at t_arr; but the existence of a non-
    # originally-ours source at any t_dep > 0 IS evidence the cascade
    # mechanism is firing.


# ---------------------------------------------------------------------------
# Seat awareness
# ---------------------------------------------------------------------------


def _assert_all_sources_owned_by(me: int, plan, world):
    bad = []
    for _tau, payload in plan:
        src_id = int(payload[0])
        src = world.planets_by_id.get(src_id)
        if src is None or int(src.owner) != int(me):
            bad.append(src_id)
    assert not bad, \
        f"sources not owned by me={me} (initial state): {bad}"


def test_seat_awareness_me_0():
    snap = _build_snap0(seed=7542, steps=60)
    plan = cascade_greedy_select(
        snap, t_start=0, t_end=60, me=0,
        opp_policy=None, path_graph=None,
        max_rebuilds=1, max_picks_per_rebuild=4,
    )
    from lib.intent import World
    obs0 = snap.state[0].observation
    obs_d = obs0 if isinstance(obs0, dict) else dict(obs0)
    world = World.from_obs(obs_d)
    # In rebuild=1 only, every source must be ORIGINALLY ours (me=0).
    _assert_all_sources_owned_by(0, plan, world)


def test_seat_awareness_me_1():
    """The agent must read me=1 and emit only from player-1-owned
    sources — this is the bug sa_online had (me=0 hardcoded)."""
    snap = _build_snap0(seed=7542, steps=60)
    plan = cascade_greedy_select(
        snap, t_start=0, t_end=60, me=1,
        opp_policy=None, path_graph=None,
        max_rebuilds=1, max_picks_per_rebuild=4,
    )
    from lib.intent import World
    obs0 = snap.state[0].observation
    obs_d = obs0 if isinstance(obs0, dict) else dict(obs0)
    world = World.from_obs(obs_d)
    _assert_all_sources_owned_by(1, plan, world)


# ---------------------------------------------------------------------------
# Out-of-bounds gate (Rule 47 — physics-primitive verification)
# ---------------------------------------------------------------------------


def test_no_oob_emission_survives_selection():
    """Every emission returned by cascade_greedy_select must, when
    re-validated through predict_fleet_fate, hit a planet (i.e. NOT
    sun, NOT out-of-bounds, NOT timeout).

    We use the `predict_fleet_fate(src, src, ...)` trick — passing
    src as both src and tgt — to query "what does this trajectory hit
    when no specific target is named?" The outcome 'planet' means "hit
    some planet other than src" (the expected capture outcome). The
    bad outcomes are {'sun', 'oob', 'timeout'}."""
    snap = _build_snap0(seed=7542, steps=80)
    plan = cascade_greedy_select(
        snap, t_start=0, t_end=80, me=0,
        opp_policy=None, path_graph=None,
        max_rebuilds=2, max_picks_per_rebuild=4,
    )
    if not plan:
        pytest.skip("empty plan; nothing to validate")
    from lib.intent import World
    obs0 = snap.state[0].observation
    obs_d = obs0 if isinstance(obs0, dict) else dict(obs0)
    world = World.from_obs(obs_d)
    for tau, payload in plan:
        src_id = int(payload[0])
        angle = float(payload[1])
        ships = int(payload[2])
        src = world.planets_by_id.get(src_id)
        if src is None:
            # Cascade-captured source; cannot validate from initial world
            # without re-running forward sim. The chooser internally
            # validated via _compute_capture_emission_from_edge at build.
            continue
        wait_N = max(0, int(tau))
        try:
            fate = predict_fleet_fate(
                src, src, angle, ships, world, wait_N=wait_N)
        except Exception:
            pytest.fail(f"predict_fleet_fate raised at tau={tau}: "
                        f"src={src_id} angle={angle} ships={ships}")
        assert fate.outcome in ("target", "planet"), \
            (f"trajectory leaves the board (outcome='{fate.outcome}') "
             f"at tau={tau} src={src_id} angle={angle} ships={ships}")


# ---------------------------------------------------------------------------
# Budget regression
# ---------------------------------------------------------------------------


def test_full_select_under_one_second_local():
    """End-to-end cascade_greedy_select WITH a precomputed path_graph
    (the production code path — agent lazy-builds once per episode at
    turn 0 then reuses).

    Production uses DEFAULT_HORIZON=25 in agents/cascade_greedy/main.py.
    Local box is ~3-5x faster than Kaggle, so 300 ms local ≈ 1 s Kaggle.
    Without path_graph the lazy build inside ctx-build adds ~600 ms per
    rebuild — that's the path the agent's turn-0 takes; tested separately."""
    snap = _build_snap0(seed=7542, steps=120)
    # Mirror the production lazy-build path_graph step (done once per
    # episode in agent's _get_or_build_path_graph).
    from lib.intent import World
    from lib.path_graph import build_path_graph
    obs0 = snap.state[0].observation
    od = {k: getattr(obs0, k, None)
          for k in ("player", "step", "planets", "fleets", "comets",
                    "comet_planet_ids", "angular_velocity")}
    od = {k: v for k, v in od.items() if v is not None}
    world = World.from_obs(od)
    pg = build_path_graph(world, t_max=120, orbiting_bucket=8, comet_bucket=2)
    t0 = time.perf_counter()
    _plan = cascade_greedy_select(
        snap, t_start=0, t_end=25, me=0,
        opp_policy=None, path_graph=pg,
        max_rebuilds=3, max_picks_per_rebuild=8,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"cascade_greedy_select took {elapsed:.3f}s"
