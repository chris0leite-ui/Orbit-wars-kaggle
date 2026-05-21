"""baseline_joint_aggr_open — AGGR + proposer fix + opening MILP.

Includes:
- AGGR (BASELINE_JOINT_AGGR=1 + tuned joint settings).
- analytical-track proposer fix (in agents/baseline/proposer.py always).
- Opening MILP for step < 30 (multi-turn schedule, then fall through to chooser).
"""
from __future__ import annotations
import os
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_OPENING_MILP", "1")
from agents.baseline.main import agent  # noqa: E402
