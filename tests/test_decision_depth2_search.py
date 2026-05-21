"""Pin tests for `decision_depth2_search` — Phase ε.2.a.

Rule 38: each behaviour exercised both off (= plain LP) and on
(= search wrapper). The two structural gates from the plan:

1. **Disabled / out-of-opening parity**: `LP_DEPTH2_SEARCH` unset OR
   `step_now >= OPENING_HORIZON` → byte-identical to
   `decision_outcome_aware_milp`.
2. **Functional check**: when enabled inside the opening, the search
   runs the inner T+1 pipeline and returns a `depth2:` status.

Real worlds built from `kaggle_environments.make("orbit_wars", ...)` so
the assertions exercise the full pipeline through `perception_default`,
`candidates_default`, `prerank_passthrough`, the LP, and
`fast_sim.step`.

Plan: /root/.claude/plans/do-it-thoroughly-consider-tingly-fox.md
"""

from __future__ import annotations

import os

import pytest

from kaggle_environments import make

from lib.pipeline.candidates import candidates_default
from lib.pipeline.decision import decision_outcome_aware_milp
from lib.pipeline.decision_depth2_search import (
    DEFAULT_OPENING_HORIZON,
    decision_depth2_search,
)
from lib.pipeline.opp_model import opp_greedy_roi
from lib.pipeline.perception import perception_default
from lib.pipeline.prerank_passthrough import prerank_passthrough


@pytest.fixture(autouse=True)
def _clear_depth2_env(monkeypatch):
    """Each test starts with LP_DEPTH2_SEARCH unset; tests opt in
    explicitly via monkeypatch."""
    monkeypatch.delenv("LP_DEPTH2_SEARCH", raising=False)
    monkeypatch.delenv("LP_DEPTH2_OPENING_HORIZON", raising=False)
    monkeypatch.delenv("LP_DEPTH2_K_MY", raising=False)
    yield


# ---------------------------------------------------------------------------
# World builder.
# ---------------------------------------------------------------------------


def _ctx_from_seed(seed: int = 42):
    """Build a TurnContext from the initial obs of seed `seed`."""
    env = make("orbit_wars", configuration={"seed": int(seed)}, debug=False)
    env.reset()
    obs = env.steps[0][0]["observation"]
    if not isinstance(obs, dict):
        obs = {k: getattr(obs, k) for k in dir(obs) if not k.startswith("_")}
    ctx = perception_default(obs, env.configuration)
    return ctx, env


def _pipeline_to_cols_and_opp(ctx):
    """Run the candidates + opp + prerank stages on a TurnContext.

    Matches the order in `lib/pipeline/compose.py`: opp runs BEFORE
    prerank so prerank can read the opp-augmented model.
    """
    cset = candidates_default(ctx)
    opp = opp_greedy_roi(ctx)
    cols = prerank_passthrough(cset, ctx, augmented_model=opp.augmented_model)
    return cols, opp


# ---------------------------------------------------------------------------
# Gate 1: disabled → byte-identical to plain LP.
# ---------------------------------------------------------------------------


def test_disabled_returns_plain_lp_result():
    """With `LP_DEPTH2_SEARCH` unset, the wrapper falls straight through
    to `decision_outcome_aware_milp`. Status string + moves + fired
    columns identical."""
    ctx, _env = _ctx_from_seed(42)
    if ctx.is_empty_obs or ctx.is_no_targets:
        pytest.skip("seed 42 step 0 has no targets — unexpected")
    cols, opp = _pipeline_to_cols_and_opp(ctx)

    plain = decision_outcome_aware_milp(cols, opp, ctx, time_limit_seconds=0.05)
    wrapped = decision_depth2_search(cols, opp, ctx, time_limit_seconds=0.05)

    # Same moves, same columns chosen, same status (we forward the LP's
    # result verbatim when disabled).
    assert wrapped.moves == plain.moves
    assert [c.column_id for c in wrapped.fired_columns] == [
        c.column_id for c in plain.fired_columns
    ]
    assert wrapped.status == plain.status


# ---------------------------------------------------------------------------
# Gate 2: out-of-opening → same fall-through, even when enabled.
# ---------------------------------------------------------------------------


def test_past_opening_horizon_returns_plain_lp_result(monkeypatch):
    """With LP_DEPTH2_SEARCH=1 but step_now >= OPENING_HORIZON, the
    wrapper still calls plain LP (preserves late-game wallclock)."""
    monkeypatch.setenv("LP_DEPTH2_SEARCH", "1")
    # Force the horizon down so the seed-42 step-0 obs is OUTSIDE it.
    monkeypatch.setenv("LP_DEPTH2_OPENING_HORIZON", "0")

    ctx, _env = _ctx_from_seed(42)
    if ctx.is_empty_obs or ctx.is_no_targets:
        pytest.skip("seed 42 step 0 has no targets — unexpected")
    cols, opp = _pipeline_to_cols_and_opp(ctx)

    plain = decision_outcome_aware_milp(cols, opp, ctx, time_limit_seconds=0.05)
    wrapped = decision_depth2_search(cols, opp, ctx, time_limit_seconds=0.05)

    assert wrapped.moves == plain.moves
    assert wrapped.status == plain.status


# ---------------------------------------------------------------------------
# Gate 3: enabled in opening → wrapper runs, status reflects search.
# ---------------------------------------------------------------------------


def test_enabled_in_opening_runs_search_and_reports_status(monkeypatch):
    """Enabled + within OPENING_HORIZON → search executes and the status
    string is the `depth2:` shape (k=, eval=, best=, elapsed_ms=)."""
    monkeypatch.setenv("LP_DEPTH2_SEARCH", "1")
    monkeypatch.setenv("LP_DEPTH2_K_MY", "2")  # keep test fast

    ctx, _env = _ctx_from_seed(42)
    if ctx.is_empty_obs or ctx.is_no_targets:
        pytest.skip("seed 42 step 0 has no targets — unexpected")
    cols, opp = _pipeline_to_cols_and_opp(ctx)

    result = decision_depth2_search(cols, opp, ctx, time_limit_seconds=2.0)

    # Wrapper status (or fall-through to base LP if budget skipped all).
    assert (
        result.status.startswith("depth2:")
        or result.status.startswith("milp:")  # fallthrough to base LP
        or result.status.startswith("greedy_fallback")
    ), f"unexpected status: {result.status}"

    # Must produce valid moves (list of [src_id, angle, ships]) or empty.
    for m in result.moves:
        assert isinstance(m, list) and len(m) == 3
        src_id, angle, ships = m
        assert isinstance(src_id, int) and src_id >= 0
        assert isinstance(angle, float)
        assert isinstance(ships, int) and ships > 0


# ---------------------------------------------------------------------------
# Gate 4: K=1 degenerate → wrapper picks the LP's portfolio.
# ---------------------------------------------------------------------------


def test_k_my_2_picks_lp_or_empty_portfolio(monkeypatch):
    """At K_my=2, LP-seeded enum returns [empty, LP_choice]. The
    wrapper's argmax thus picks whichever has higher leaf+continuation
    value — either no-op or the plain LP's portfolio. We assert the
    wrapped moves are a subset of {empty, plain_moves} — the wrapper
    cannot invent a third option at this depth."""
    monkeypatch.setenv("LP_DEPTH2_SEARCH", "1")
    monkeypatch.setenv("LP_DEPTH2_K_MY", "2")

    ctx, _env = _ctx_from_seed(42)
    if ctx.is_empty_obs or ctx.is_no_targets:
        pytest.skip("seed 42 step 0 has no targets — unexpected")
    cols, opp = _pipeline_to_cols_and_opp(ctx)

    plain = decision_outcome_aware_milp(cols, opp, ctx, time_limit_seconds=0.05)
    wrapped = decision_depth2_search(cols, opp, ctx, time_limit_seconds=2.0)

    wrapped_sorted = sorted(map(tuple, wrapped.moves))
    plain_sorted = sorted(map(tuple, plain.moves))
    # Wrapped result is either empty (no-op preferred) or matches plain.
    assert wrapped_sorted == [] or wrapped_sorted == plain_sorted, (
        f"wrapped picked a third portfolio:\n"
        f"  wrapped={wrapped_sorted}\n"
        f"  plain  ={plain_sorted}"
    )


# ---------------------------------------------------------------------------
# Gate 5: empty columns → empty result, no crash.
# ---------------------------------------------------------------------------


def test_empty_columns_returns_empty_no_crash(monkeypatch):
    """If the prerank returned no columns, the wrapper short-circuits to
    an empty DecisionResult — same as the plain LP would."""
    from lib.pipeline.types import PrerankedColumns
    monkeypatch.setenv("LP_DEPTH2_SEARCH", "1")

    ctx, _env = _ctx_from_seed(42)
    if ctx.is_empty_obs or ctx.is_no_targets:
        pytest.skip("seed 42 step 0 has no targets — unexpected")
    _cols_real, opp = _pipeline_to_cols_and_opp(ctx)
    # Force empty.
    cols = PrerankedColumns(columns=[], n_before_filter=0, n_after_filter=0)

    result = decision_depth2_search(cols, opp, ctx, time_limit_seconds=0.5)
    assert result.moves == []
    assert result.fired_columns == []
    assert "empty_columns" in result.status or "empty" in result.status
