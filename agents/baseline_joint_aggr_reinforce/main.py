"""baseline_joint_aggr_reinforce — AGGR variant + reinforce-emit post-pass.

Extends `baseline_joint_aggr` with `BASELINE_REINFORCE_EMIT=1`, which wires
`lib.missions.reinforce.propose_reinforce_missions` into the chooser's
emit path. Defense-directed: fires ONLY for OUR planets predicted to flip
to enemy within model.horizon. Distinct from `drain_idle_rear` (blanket
forced emits, falsified 2026-05-18).

Origin: PI live-game observation (4P seed 914393430), a +5 prod planet
of ours fell while rear sources held reserves.
"""

from __future__ import annotations

import os

# Inherit AGGR env vars.
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
# Enable reinforce-emit post-pass.
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")

from agents.baseline.main import agent  # noqa: E402
