"""baseline_joint_aggr_reinforce_d — AGGR + reactive + anticipated + 4P leader-focus.

Stacks on top of reinforce_b: also enables LEADER_FOCUS_WEIGHT=1.5 so
the chooser prefers attacks on the currently-leading opp in 4P.
Disabled (no-op) in 2P automatically.
"""
from __future__ import annotations
import os
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_LEADER_FOCUS", "1.5")
from agents.baseline.main import agent  # noqa: E402
