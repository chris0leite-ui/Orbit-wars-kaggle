"""baseline_joint_aggr_noaggr — AGGR-flag OFF (test for 4P double-count bug).

Code-review agent (2026-05-21) found that BASELINE_JOINT_AGGR=1 silently
double-counts capture EV in 4P: the JOINT enumeration is gated to 2P,
but the env var ALSO lifts the used_tgts lock on the SOLO emit loop.
In 4P this means multiple solo launches stack on the same target, but
each was scored in an independent rollout assuming it was alone.

This variant turns off BASELINE_JOINT_AGGR. If it wins more 4P seeds
than AGGR-with-the-flag, the double-count hypothesis is confirmed.
"""
from __future__ import annotations
import os
# JOINT enumeration kept (2P-gated anyway). AGGR-flag OFF.
os.environ.setdefault("BASELINE_JOINT", "1")
from agents.baseline.main import agent  # noqa: E402
