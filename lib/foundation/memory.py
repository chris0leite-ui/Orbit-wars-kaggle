"""Memory protocol — persists across turns within a single game process.

Kaggle agents are invoked as stateless `agent(obs, configuration)`
calls. Module-level singletons survive within one game subprocess but
reset between games — `Memory` is the formal interface for that
cross-turn state.

Threading vs mutation: strategies receive `memory` as an input and
return a new `Memory` (along with their action). This keeps strategies
functionally pure so the same code path works under `jax.vmap` (where
mutation is impossible) and in stateless eval.

Concrete implementations land in `lib.foundation.memory_impls` (Step 7):
- `JitCacheMemory` — caches `jax.jit`'d closures so the cold compile
  on turn 1 is amortised over turns 2-500.
- `WarmStartMemory` — search subtree / PV from previous turn.
- `OppModelMemory` — running posterior over opponent archetypes.
- `MissionMemory` — strategy-level intent across turns (3-wave plans,
  drain-then-snipe sequences, etc.); prunes stale missions whose
  target was captured or source was lost.
- `CompositeMemory` — combines the above; what the live agent uses.

`EmptyMemory` is defined here as the no-op baseline so Step-1
strategies have a working default before Step 7 lands.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Memory(Protocol):
    """Across-turn state. Returned from `Strategy.emit` for the next
    turn; consumed by the next call. Strategies that don't need memory
    pass and return `EmptyMemory()`.

    Implementations should make `update` and `reset` cheap — they're
    called once per turn (`update`) or once per game (`reset`).
    """

    def update(self, state: Any, action_taken: Any) -> "Memory":
        """Called once per turn AFTER the strategy emits. Returns a
        new `Memory` reflecting the played action and resulting state.

        `state` is the GameState / scalar obs the strategy saw.
        `action_taken` is what the strategy chose to play (typically
        an `ActionTensor` slice).
        """
        ...

    def reset(self) -> "Memory":
        """Called once at game start (or between games). Returns a
        fresh `Memory` with no carried state.
        """
        ...


class EmptyMemory:
    """No-op `Memory`. Use for stateless strategies."""

    def update(self, state: Any, action_taken: Any) -> "EmptyMemory":
        return self

    def reset(self) -> "EmptyMemory":
        return EmptyMemory()

    def __repr__(self) -> str:
        return "EmptyMemory()"
