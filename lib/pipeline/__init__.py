"""Modular analytical-agent pipeline.

Seven stages, each a callable with a documented signature. Implementations
are swappable from a registry without rebuilding the rest of the agent.
The current analytical agent (submission 52857903) is one specific
composition; alternative agents are different compositions of the same
registry.

See `/root/.claude/plans/spicy-marinating-token.md` for the architectural
intent and phase plan.

Stage contracts (see types.py for full signatures):
  1. Perception      — obs                → TurnContext
  2. Candidate gen   — TurnContext        → CandidateSet
  3. Pre-rank        — CandidateSet       → PrerankedColumns
  4. Opp model       — TurnContext        → OppModelResult
  5. Decision rule   — (cols, opp, ctx)   → DecisionResult
  6. Leaf evaluator  — used inside Stage 5
  7. Commit          — (decision, ctx)    → CommittedMoves

Plus an OpeningOverride stage that short-circuits Stages 2-7 when an
opening MILP schedule is in force.

Composition:
  agent = compose(perception=..., candidates=..., prerank=...,
                  opp_model=..., decision=..., commit=...,
                  opening_override=...)
  moves = agent(obs, configuration)
"""

from lib.pipeline.compose import compose, default_composition

__all__ = ["compose", "default_composition"]
