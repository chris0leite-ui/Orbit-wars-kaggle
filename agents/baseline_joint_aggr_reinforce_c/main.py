"""baseline_joint_aggr_reinforce_c — AGGR + reactive + anticipated + tight-stagnation-drain.

Stacks on top of reinforce_b: also enables drain_idle_rear with much
tighter thresholds (threshold=80 vs falsified-audit's 30, reserve=30).
Catches the pure-stagnation case where there's no enemy threat anywhere
but a source has accumulated > 80 idle ships and a high-prod friendly
destination exists.
"""
from __future__ import annotations
import os
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_IDLE_DRAIN", "1")
os.environ.setdefault("BASELINE_IDLE_DRAIN_THRESHOLD", "80")
os.environ.setdefault("BASELINE_IDLE_DRAIN_RESERVE", "30")
os.environ.setdefault("BASELINE_IDLE_REAR_THRESHOLD", "45.0")
from agents.baseline.main import agent  # noqa: E402
