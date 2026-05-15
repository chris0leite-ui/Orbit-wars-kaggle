"""Strategy protocol — pluggable per-agent policy.

A `Strategy` is a pure function from `(state, my_id, ctx, memory)` to
`(action_tensor, new_memory)`. Heuristics, search-based, and learned
policies all conform.

The registry (`register_strategy` / `get_strategy`) lets the live
agent, scripts, and tests A/B by name — the handle the PI asked for
in the design phase.

Pure-function design choice: memory threads through (`emit` returns
`new_memory`) rather than being mutated in place. Keeps strategies
JAX-vmap-compatible and stateless-test-friendly.

Step 1 of the plan; Step 8 adds the concrete strategy adapters that
wrap existing agents (v3.5.1, v7_pv, roi_greedy, search_brute).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from lib.foundation.actions import ActionTensor
from lib.foundation.memory import Memory


@dataclass(frozen=True)
class StrategyCtx:
    """Per-turn scratch context passed to `Strategy.emit`.

    Fields:
        turn_budget_ms : wall-clock budget for this turn's emit call
                         (1000 ms on the live ladder).
        rng_key        : JAX PRNG key, or None for deterministic
                         strategies.
        world_model    : optional precomputed `WorldModel` (or JAX
                         equivalent) handle to avoid rebuilding inside
                         the strategy.
    """

    turn_budget_ms: float = 1000.0
    rng_key: Optional[Any] = None
    world_model: Optional[Any] = None


@runtime_checkable
class Strategy(Protocol):
    """Pluggable agent policy."""

    name: str

    def emit(
        self,
        state: Any,
        my_id: int,
        ctx: StrategyCtx,
        memory: Memory,
    ) -> tuple[ActionTensor, Memory]:
        """Choose an action for `my_id` on the current turn.

        Returns:
            action_tensor : `ActionTensor` of shape `(1, T, A, L)` —
                            one candidate (the chosen action). The
                            leading C axis is kept at 1 so this slots
                            directly into the batched evaluator's
                            candidate axis downstream.
            new_memory    : updated memory to thread to the next turn.
        """
        ...


# -- Registry ---------------------------------------------------------------

_REGISTRY: dict[str, Strategy] = {}


def register_strategy(name: str, strategy: Strategy) -> None:
    """Register `strategy` under `name`. Re-registration overwrites
    (intended for tests and development)."""
    _REGISTRY[name] = strategy


def get_strategy(name: str) -> Strategy:
    """Look up a strategy by name. Raises `KeyError` if not
    registered."""
    if name not in _REGISTRY:
        known = sorted(_REGISTRY.keys())
        raise KeyError(f"Strategy {name!r} not registered. Known: {known}")
    return _REGISTRY[name]


def list_strategies() -> list[str]:
    """Return all registered strategy names, sorted alphabetically."""
    return sorted(_REGISTRY.keys())


def clear_registry() -> None:
    """Remove all registered strategies. For test isolation."""
    _REGISTRY.clear()
