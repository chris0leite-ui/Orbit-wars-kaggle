"""Pin tests for `lib/joint_solver/opening_search.py` — Phase η.2.

The widened search reads from the trajectory matrix and runs an MILP
over (src ∈ my-planets, tgt ∈ all non-mine non-comet, launch_tick).

Coverage:

1. Opt-in: `opening_search_enabled()` reads env var per call.
2. Returns same `OpeningPlan` shape as `opening_planner.opening_plan`.
3. Planet 16 IS in the candidate set on seed 42 (the motivating case).
4. The MILP picks a schedule (status not 'empty', n_chosen > 0).
5. Chain candidates emit when a high-value parent capture enables them.
6. With `LP_OPENING_SEARCH=1` in `lib/pipeline/opening.py`, the
   dispatch routes to opening_plan_search.
"""

from __future__ import annotations

import os

import pytest

from kaggle_environments import make

from lib.intent import World
from lib.joint_solver.opening_search import (
    _SearchCandidate,
    _build_candidates,
    opening_plan_search,
    opening_search_enabled,
)
from lib.joint_solver.trajectory_matrix import (
    TrajectoryMatrix,
    get_default as get_default_matrix,
)
from lib.pipeline.perception import perception_default
from lib.world_model import WorldModel


@pytest.fixture(autouse=True)
def _isolate_matrix(monkeypatch):
    """Reset the trajectory matrix singleton; clear LP_OPENING_SEARCH."""
    monkeypatch.delenv("LP_OPENING_SEARCH", raising=False)
    get_default_matrix().reset()
    yield
    get_default_matrix().reset()


def _ctx_from_seed(seed: int):
    env = make("orbit_wars", configuration={"seed": int(seed)}, debug=False)
    env.reset()
    obs = env.steps[0][0]["observation"]
    if not isinstance(obs, dict):
        obs = {k: getattr(obs, k) for k in dir(obs) if not k.startswith("_")}
    ctx = perception_default(obs, env.configuration)
    return ctx, env


def _prime_matrix(ctx):
    """Build the matrix for this ctx (no env var; we manually drive)."""
    matrix = get_default_matrix()
    matrix.begin_game(ctx.world, ctx.model, float(ctx.omega), int(ctx.me))


# ---------------------------------------------------------------------------
# Opt-in env var.
# ---------------------------------------------------------------------------


def test_opening_search_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("LP_OPENING_SEARCH", raising=False)
    assert opening_search_enabled() is False
    monkeypatch.setenv("LP_OPENING_SEARCH", "1")
    assert opening_search_enabled() is True
    monkeypatch.setenv("LP_OPENING_SEARCH", "off")
    assert opening_search_enabled() is False


# ---------------------------------------------------------------------------
# OpeningPlan return shape.
# ---------------------------------------------------------------------------


def test_returns_opening_plan_shape():
    """opening_plan_search must return an OpeningPlan with the same
    fields as opening_planner.opening_plan (so the dispatch in
    lib/pipeline/opening.py is transparent)."""
    from lib.joint_solver.opening_planner import OpeningPlan

    ctx, _env = _ctx_from_seed(42)
    if ctx.is_empty_obs or ctx.is_no_targets:
        pytest.skip("seed 42 step 0 has no targets")
    _prime_matrix(ctx)
    op = opening_plan_search(ctx)
    assert isinstance(op, OpeningPlan)
    assert hasattr(op, "schedule")
    assert hasattr(op, "objective")
    assert hasattr(op, "status")
    assert hasattr(op, "pruning_waterfall")


# ---------------------------------------------------------------------------
# The motivating case: planet 16 IS a candidate.
# ---------------------------------------------------------------------------


def test_seed42_planet16_in_candidate_set():
    """The whole point of Phase η: planet 16 from planet 0 must be in
    the candidate set. opening_planner drops it via the K=8 prune;
    opening_search must NOT."""
    ctx, _env = _ctx_from_seed(42)
    if ctx.is_empty_obs or ctx.is_no_targets:
        pytest.skip("seed 42 step 0 has no targets")
    _prime_matrix(ctx)
    candidates = _build_candidates(ctx)
    p16_candidates = [c for c in candidates if c.tgt_id == 16 and c.src_id == 0]
    assert p16_candidates, (
        f"expected ≥1 (0→16) candidate; got 0. "
        f"total candidates={len(candidates)}, "
        f"direct={[c for c in candidates if c.parent_column_id is None][:3]}"
    )


# ---------------------------------------------------------------------------
# MILP picks something.
# ---------------------------------------------------------------------------


def test_milp_picks_a_schedule_on_seed42():
    ctx, _env = _ctx_from_seed(42)
    if ctx.is_empty_obs or ctx.is_no_targets:
        pytest.skip("seed 42 step 0 has no targets")
    _prime_matrix(ctx)
    op = opening_plan_search(ctx)
    assert len(op.schedule) > 0, (
        f"empty schedule on seed 42; status={op.status!r}, "
        f"waterfall={op.pruning_waterfall}"
    )
    # Status string starts with 'search:' to distinguish from opening_planner.
    assert op.status.startswith("search:"), f"unexpected status: {op.status!r}"


# ---------------------------------------------------------------------------
# Chain candidates.
# ---------------------------------------------------------------------------


def test_chain_candidates_emit_when_parent_enables_them():
    """When opening_search finds a viable parent capture
    (my_src → neutral_tgt) with arrival_tick well before OPENING_HORIZON,
    AT LEAST ONE chain candidate (src=captured_tgt → other_tgt) should
    appear in the candidate set."""
    ctx, _env = _ctx_from_seed(42)
    if ctx.is_empty_obs or ctx.is_no_targets:
        pytest.skip("seed 42 step 0 has no targets")
    _prime_matrix(ctx)
    candidates = _build_candidates(ctx)
    chains = [c for c in candidates if c.parent_column_id is not None]
    # If no chains emit, the chain code path is broken (or this seed
    # legitimately has no chain-enabling captures — but seed 42 has
    # several early captures, so we EXPECT chains).
    assert chains, (
        f"expected ≥1 chain candidate; got 0. "
        f"n_direct={len([c for c in candidates if c.parent_column_id is None])}"
    )
    # Each chain must reference a real parent column_id.
    parent_ids = {c.column_id for c in candidates if c.parent_column_id is None}
    for chain in chains:
        assert chain.parent_column_id in parent_ids, (
            f"chain {chain} references non-existent parent_column_id"
        )


# ---------------------------------------------------------------------------
# Status string when matrix empty.
# ---------------------------------------------------------------------------


def test_returns_no_candidates_when_matrix_empty():
    """If the matrix wasn't primed (begin_game never called) AND the
    matrix has no entries for this world, opening_plan_search returns
    an empty plan with status 'search_no_candidates'.
    """
    ctx, _env = _ctx_from_seed(42)
    if ctx.is_empty_obs or ctx.is_no_targets:
        pytest.skip("seed 42 step 0 has no targets")
    # Do NOT call _prime_matrix — matrix is empty.
    get_default_matrix().reset()
    op = opening_plan_search(ctx)
    assert op.status == "search_no_candidates" or len(op.schedule) == 0


# ---------------------------------------------------------------------------
# Pipeline dispatch — LP_OPENING_SEARCH routes to the new search.
# ---------------------------------------------------------------------------


def test_pipeline_dispatch_routes_via_env_var(monkeypatch):
    """When LP_OPENING_SEARCH=1, lib/pipeline/opening.py's opening_default
    should call opening_plan_search and return moves from its schedule."""
    monkeypatch.setenv("LP_OPENING_SEARCH", "1")
    ctx, _env = _ctx_from_seed(42)
    if ctx.is_empty_obs or ctx.is_no_targets:
        pytest.skip("seed 42 step 0 has no targets")
    # Need to prime the matrix manually since perception only does it
    # when env var is read by perception itself; the test goes through
    # opening_default which doesn't prime.
    _prime_matrix(ctx)

    from lib.pipeline.opening import opening_default
    res = opening_default(ctx)
    # Should NOT be a fall-through (committed=None) since the search
    # produces a non-empty schedule on seed 42.
    assert res.committed is not None, (
        f"opening_default fell through despite LP_OPENING_SEARCH=1; "
        f"diagnostics={res.diagnostics}"
    )
    # Status string should reflect the search path.
    status_str = (res.diagnostics or {}).get("status", "")
    assert "search:" in status_str, (
        f"expected 'search:' in status, got {status_str!r}"
    )
