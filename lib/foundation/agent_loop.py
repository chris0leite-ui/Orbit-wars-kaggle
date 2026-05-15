"""Kaggle entry-point wrapper that threads `Memory` across turns.

Kaggle invokes `agent(obs, configuration)` once per turn — stateless
on the surface. But module-level singletons survive within one game
subprocess, so we can hold a `Memory` instance there and rely on its
`update / reset` lifecycle to handle cross-turn state.

This module is the minimal cross-turn pipe:
- `get_memory()` — return the current singleton, lazily creating
  a `CompositeMemory` on first call.
- `set_memory(memory)` — replace the singleton (Step 8 strategy
  adapters call this after `emit` returns `new_memory`).
- `reset_memory()` — clear the singleton; called automatically
  between games when the agent detects a `step=0` obs (game restart
  marker).

Step 8 will add the actual `agent(obs, configuration)` function that
dispatches to a registered strategy. For Step 7, we just provide the
memory plumbing and tests verify it across simulated turns.

Thread-safety: Kaggle agents are single-threaded per subprocess, so
the lock here is defensive. The cost is negligible.
"""

from __future__ import annotations

import threading
from typing import Optional

from lib.foundation.memory import Memory
from lib.foundation.memory_impls import CompositeMemory


# Module-level singleton. None until first `get_memory()` call.
_MEMORY: Optional[Memory] = None
_LOCK = threading.Lock()


def get_memory() -> Memory:
    """Return the current `Memory`, lazily initialising a fresh
    `CompositeMemory` on first call within this subprocess.

    Strategies / agent wrappers call this at the top of each turn
    to read the carried-over state."""
    global _MEMORY
    with _LOCK:
        if _MEMORY is None:
            _MEMORY = CompositeMemory()
        return _MEMORY


def set_memory(memory: Memory) -> None:
    """Replace the singleton `Memory`. The Kaggle wrapper calls this
    after a strategy's `emit()` returns `new_memory`."""
    global _MEMORY
    with _LOCK:
        _MEMORY = memory


def reset_memory() -> None:
    """Clear the singleton. Called between games (e.g., on detecting
    a fresh `step=0` obs after a previously-running game) or in
    tests for isolation."""
    global _MEMORY
    with _LOCK:
        _MEMORY = None


def maybe_reset_on_new_game(obs) -> None:
    """Reset the singleton if `obs` looks like the start of a fresh
    game (step == 0). Idempotent.

    The Kaggle agent wrapper calls this at the top of every turn —
    the first `step=0` obs of a new game triggers reset; subsequent
    turns leave it alone."""
    step = 0
    if isinstance(obs, dict):
        step = int(obs.get("step", 0) or 0)
    else:
        step = int(getattr(obs, "step", 0) or 0)
    if step == 0:
        reset_memory()
