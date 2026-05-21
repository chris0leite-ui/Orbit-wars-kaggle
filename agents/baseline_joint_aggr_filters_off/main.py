"""baseline_joint_aggr_filters_off — AGGR + proposer filters disabled.

Disables PROPOSER_COST_PARITY and PROPOSER_HOLD_FEASIBILITY. Critique
agent (2026-05-21) flagged these as 2P-tuned filters that over-reject
in 4P (~3x reject rate due to 3 opps' "could-reactor" universe being
3x larger). phase_c also turns these off.
"""
from __future__ import annotations
import os
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("PROPOSER_COST_PARITY", "off")
os.environ.setdefault("PROPOSER_HOLD_FEASIBILITY", "off")
from agents.baseline.main import agent  # noqa: E402
