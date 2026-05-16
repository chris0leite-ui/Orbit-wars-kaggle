"""v21_ae — v21 patches A + E1 only (E2 hold-check disabled).

Diagnostic variant for the iteration-1 fallback protocol: if
v21 (all three patches) fails to clear the h2h gate vs v15, this
variant isolates whether Patch E2 (rollout-based hold-check) is the
drag. If v21_ae clears the gate but v21 doesn't, E2 is the
regression. If neither clears, E1 or A is the issue.
"""
from __future__ import annotations

import agents.v21.main as _v21

# Override the flag BEFORE any agent() call. Module-level constant —
# v21's agent() reads it via name lookup so this redefinition takes
# effect for all subsequent calls in this process.
_v21.CAPTURE_HOLD_CHECK = False

# Re-export everything the harness needs.
agent = _v21.agent
_INSTRUMENT_COUNTERS = _v21._INSTRUMENT_COUNTERS
