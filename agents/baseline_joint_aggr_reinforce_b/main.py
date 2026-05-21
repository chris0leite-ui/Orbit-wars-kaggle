"""baseline_joint_aggr_reinforce_b — AGGR + reactive reinforce + anticipated reinforce.

Adds preemptive forward-staging on top of baseline_joint_aggr_reinforce:
fires reinforce launches for friendly destinations with inbound enemy
fleets that thin defenders below safety margin, even if T_loss isn't
predicted. Implements PI 2026-05-21 direction (b) — mobilize idle
planets toward planets that need them.
"""

from __future__ import annotations
import os
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
from agents.baseline.main import agent  # noqa: E402
