"""Strategy adapters built on top of the foundation layer.

Each module in this package registers a `Strategy`-protocol implementation
under a unique name via `lib.foundation.register_strategy`. Live agents
in `agents/v8_*/` import these modules to trigger registration, then
call `get_strategy(name).emit(...)` per turn.

The smoke baseline is `greedy_roi` — a thin wrapper around the
pre-existing `agents/simple/roi` heuristic. Phase A's `v8_analytic`
lands alongside it.
"""
