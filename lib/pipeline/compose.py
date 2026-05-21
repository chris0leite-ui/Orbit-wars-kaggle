"""Pipeline composer.

Wires stage implementations into a callable agent that matches the
standard kaggle signature `agent(obs, configuration) -> moves`.

Execution order (actual; differs from the conceptual stage numbering
because Stage 3's prerank uses Stage 4's opp-augmented model — see
the comment at the prerank call site):
  1. perception(obs, configuration)              → ctx
  2. opening_override(ctx)                       → opening; if committed, return
  3. candidates(ctx)                             → cset
  4. opp_model(ctx)                              → opp        [runs BEFORE prerank]
  5. prerank(cset, ctx, augmented_model=opp.augmented_model) → cols
  6. decision(cols, opp, ctx)                    → decision
  7. commit(decision, ctx)                       → committed
  return committed.moves

The default composition mirrors submission 52857903 bit-exact:
  perception        → perception_default
  opening_override  → opening_default
  candidates        → candidates_default
  opp_model         → opp_greedy_roi
  prerank           → prerank_w1w2_filter
  decision          → decision_outcome_aware_milp
  commit            → commit_stateless

Alternative compositions swap one or more stages from the registry.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from lib.pipeline.candidates import candidates_default
from lib.pipeline.commit import commit_stateless
from lib.pipeline.decision import decision_outcome_aware_milp
from lib.pipeline.opening import opening_default
from lib.pipeline.opp_model import opp_greedy_roi
from lib.pipeline.perception import perception_default
from lib.pipeline.prerank import prerank_w1w2_filter
from lib.pipeline.types import (
    CommittedMoves, OppModelResult, PrerankedColumns,
)


def compose(
    *,
    perception: Callable = perception_default,
    opening_override: Optional[Callable] = opening_default,
    candidates: Callable = candidates_default,
    opp_model: Callable = opp_greedy_roi,
    prerank: Callable = prerank_w1w2_filter,
    decision: Callable = decision_outcome_aware_milp,
    commit: Callable = commit_stateless,
    time_limit_seconds: float = 0.3,
    return_trace: bool = False,
) -> Callable:
    """Compose stage implementations into a callable agent.

    Returns `agent(obs, configuration)` matching the kaggle signature.
    If `return_trace=True`, the returned callable yields `(moves, trace)`
    where `trace` is a dict containing each stage's output (debug aid;
    NOT part of the bit-parity contract).
    """

    def agent_fn(obs, configuration=None):
        # Stage 1: perception
        ctx = perception(obs, configuration)
        trace: dict = {"ctx": ctx} if return_trace else {}

        # Empty obs short-circuit (parity with mpc.solve_turn:148).
        if ctx.is_empty_obs or ctx.is_no_targets:
            moves: list = []
            if return_trace:
                trace["short_circuit"] = (
                    "empty_obs" if ctx.is_empty_obs else "no_targets"
                )
                return moves, trace
            return moves

        # Opening override
        if opening_override is not None:
            opening = opening_override(ctx)
            if return_trace:
                trace["opening"] = opening
            if opening.committed is not None:
                if return_trace:
                    return opening.committed.moves, trace
                return opening.committed.moves

        # Stage 2: candidates
        cset = candidates(ctx)
        if return_trace:
            trace["candidates"] = cset

        # Stage 4: opp model (runs BEFORE Stage 3 because prerank's value
        # function needs the opp-augmented model — matches mpc.solve_turn).
        opp = opp_model(ctx)
        if return_trace:
            trace["opp"] = opp

        # Stage 3: prerank (with opp-augmented model)
        cols = prerank(cset, ctx, augmented_model=opp.augmented_model)
        if return_trace:
            trace["cols"] = cols

        # Stage 5: decision rule
        decision_result = decision(
            cols, opp, ctx, time_limit_seconds=time_limit_seconds,
        )
        if return_trace:
            trace["decision"] = decision_result

        # Stage 7: commit
        committed = commit(decision_result, ctx)
        if return_trace:
            trace["committed"] = committed
            return committed.moves, trace
        return committed.moves

    return agent_fn


def default_composition(time_limit_seconds: float = 0.3) -> Callable:
    """Convenience: the bit-parity reference composition (submission 52857903)."""
    return compose(time_limit_seconds=time_limit_seconds)
