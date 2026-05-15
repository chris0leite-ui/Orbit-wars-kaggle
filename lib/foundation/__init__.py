"""Foundation layer — analytical predictor, JAX batched evaluator,
strategy plugin system, and cross-turn memory.

Step 1 of the plan at
`/root/.claude/plans/be-a-senior-software-expressive-ember.md`. Adds
the protocol surface and action types; subsequent steps fill in the
Predictor, JAX engine wrapper, memory implementations, and strategy
adapters.

Public API:
    from lib.foundation import (
        ActionSpec, ActionTensor, specs_to_tensor, tensor_to_specs,
        Memory, EmptyMemory,
        Strategy, StrategyCtx,
        register_strategy, get_strategy, list_strategies, clear_registry,
    )
"""

from lib.foundation.actions import (
    ActionSpec,
    ActionTensor,
    specs_to_tensor,
    tensor_to_specs,
)
from lib.foundation.memory import EmptyMemory, Memory
from lib.foundation.memory_impls import (
    CommittedMission,
    CompositeMemory,
    JitCacheMemory,
    MissionMemory,
)
from lib.foundation.strategy import (
    Strategy,
    StrategyCtx,
    clear_registry,
    get_strategy,
    list_strategies,
    register_strategy,
)

__all__ = [
    "ActionSpec",
    "ActionTensor",
    "specs_to_tensor",
    "tensor_to_specs",
    "Memory",
    "EmptyMemory",
    "JitCacheMemory",
    "MissionMemory",
    "CommittedMission",
    "CompositeMemory",
    "Strategy",
    "StrategyCtx",
    "register_strategy",
    "get_strategy",
    "list_strategies",
    "clear_registry",
]
