"""Strategy adapters built on top of the foundation layer.

Each module in this package registers a `Strategy`-protocol implementation
under a unique name via `lib.foundation.register_strategy`. Live agents
import these modules to trigger registration, then call
`get_strategy(name).emit(...)` per turn.

Modules:
- `analytic_fastsim` — fast_sim K-step rollout value-head pattern with
  reactive `lite_greedy` opp + F1+F2 favor leaf. Extracted from
  `claude/space-fleet-physics-engine-lrLE6` as the substrate for
  Step 3 of the strategy plan (learned planet-value head). The
  orphaned Phase A/C beams from that branch are intentionally NOT
  included here — see the plan's Step 0 skip list.
"""
