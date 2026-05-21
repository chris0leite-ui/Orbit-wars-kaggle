"""consolidated + JOINT TOP_K=8 (was 5), MAX_PAIRS=100 (was 60).

Targets PI 2026-05-21 open question: "should AGGR TOP_K lift for
high-prod enemy targets?" — image trace showed a 319-ship planet
sitting idle adjacent to combat while the chooser attacked from
elsewhere; hypothesis is that the big idle source wasn't in the
top-5 ranked solos for the target's joint pool.

Unconditional lift (cheaper than per-target gating): TOP_K 5 -> 8
expands per-target enumeration from C(5,2)=10 pairs to C(8,2)=28
pairs. MAX_PAIRS 60 -> 100 keeps total budget commensurate.
"""
from __future__ import annotations
import os
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "8")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "100")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
from agents.baseline.main import agent  # noqa: E402
