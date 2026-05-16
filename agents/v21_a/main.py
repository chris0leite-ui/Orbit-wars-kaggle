"""v21_a — v21 Patch A only (E1 prefilter + E2 hold-check both disabled).

Second-tier diagnostic for the iteration-1 fallback protocol: if
v21_ae (A + E1) also fails to clear the h2h gate vs v15, this variant
isolates whether Patch A alone is the regression by stripping E1 too.
If v21_a clears the gate, E1 is the drag (over-filtering valid captures).
If v21_a fails too, Patch A's joint-emit pipeline itself doesn't lift.
"""
from __future__ import annotations

import agents.v21.main as _v21

_v21.CAPTURE_HOLD_PREFILTER = False
_v21.CAPTURE_HOLD_CHECK = False

agent = _v21.agent
_INSTRUMENT_COUNTERS = _v21._INSTRUMENT_COUNTERS
