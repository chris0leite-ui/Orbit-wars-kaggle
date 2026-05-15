"""`AnalyticStrategy` — Phase A.

Strategy-agnostic beam search over `(src, target)` atomic launches,
scored by a Tier-1 mirror K-step rollout followed by a JAX-pure
value head that includes future production (held planets are valued
at `production × (500 - state.step)`, not just current ship count).

Pipeline:
1. `enumerate_atomic_launches(state, my_id)` — Cartesian product over
   owned-planet × every-other-alive-planet × ship-fractions in
   `{0.5, 1.0}`. ETA-filtered. ~200-600 atoms typical.
2. `beam_search(...)` — vmap'd per-level expansion of the top-`width`
   partial action sets; adaptive width shrinks under budget pressure.
   Returns the winning `list[ActionSpec]`.
3. Pack the winning set into an `ActionTensor` via `specs_to_tensor`.

Phase A omissions (intentional, addressed in Phase B):
- No `MissionMemory` — every turn re-enumerates and re-searches.
- Single beam-start (the empty action set); no diversified
  constructors. Diversification is a Phase B fallback if beam
  collapses to greedy-equivalent picks.
- No live A/B harness; locally tested via the smoke + perf tests.

Registers `"v8_analytic"` on import.
"""

from __future__ import annotations

from lib.foundation.actions import ActionTensor, specs_to_tensor
from lib.foundation.memory import Memory
from lib.foundation.strategies.analytic_score import enumerate_atomic_launches
from lib.foundation.strategies.beam_search import beam_search
from lib.foundation.strategy import StrategyCtx, register_strategy
from lib.game.jax.jax_types import GameState


class AnalyticStrategy:
    """Strategy-agnostic analytical lookahead.

    `width` / `depth` / `K` / `budget_ms` are passed through to
    `beam_search`. Defaults match the Phase A target: 8×8 beam,
    K=5 mirror rollout, 800 ms wall-clock cap inside the 1000 ms
    turn budget.
    """

    name = "v8_analytic"

    def __init__(
        self,
        *,
        width: int = 4,
        depth: int = 4,
        K: int = 5,
        budget_ms: float = 800.0,
        opp_aggressive: bool = True,
    ) -> None:
        self._width = width
        self._depth = depth
        self._K = K
        self._budget_ms = budget_ms
        self._opp_aggressive = opp_aggressive

    def emit(
        self,
        state: GameState,
        my_id: int,
        ctx: StrategyCtx,
        memory: Memory,
    ) -> tuple[ActionTensor, Memory]:
        atomics = enumerate_atomic_launches(state, my_id)
        num_agents = int(state.num_agents)

        # Cap beam wall-clock by what's left of the turn budget
        # (subtract ~100 ms for env round-trip + post-processing).
        effective_budget = min(self._budget_ms, max(50.0, ctx.turn_budget_ms - 100.0))

        winning_set = beam_search(
            state,
            atomics,
            my_id,
            width=self._width,
            depth=self._depth,
            K=self._K,
            num_agents=num_agents,
            opp_aggressive=self._opp_aggressive,
            budget_ms=effective_budget,
        )

        tensor = specs_to_tensor([winning_set], horizon=1)
        return tensor, memory


register_strategy("v8_analytic", AnalyticStrategy())
