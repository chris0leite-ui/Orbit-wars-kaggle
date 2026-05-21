"""baseline_joint_aggr_proposerfix — AGGR + analytical proposer fix ONLY.

No post-pass reinforce. Isolates the impact of the proposer fix
(STRATEGIC_DEFENSE_PROD stockpile + bundle blind spot) on baseline AGGR.
"""
from __future__ import annotations
import os
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
from agents.baseline.main import agent  # noqa: E402
