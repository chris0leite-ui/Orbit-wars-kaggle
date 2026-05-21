"""baseline_joint_aggr_consolidated_drain — consolidated + smart stagnant drain.

Adds STAGNANT_DRAIN to the live-submitted consolidated variant
(sub 52882014). The drain hard-gates on zero-inbound-enemy + dynamic
production-scaled reserve + meaningful action-distance improvement,
distinct from the 2026-05-18-falsified drain_idle_rear.

PI 2026-05-21 directive: unlock the giant ship reserves that 37/40
planets are sitting on (50+ idle ships for 20+ turns in the trace
of the consolidated agent).
"""
from __future__ import annotations
import os
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
os.environ.setdefault("BASELINE_STAGNANT_DRAIN", "1")
from agents.baseline.main import agent  # noqa: E402
