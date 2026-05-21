"""baseline_joint_aggr_full — the kitchen sink: AGGR + proposer fix + opening + reinforce.

Full stack:
- AGGR (joint enumeration, AGGR settings).
- analytical-track proposer fix (strategic stockpile + bundle blind spot).
- Opening MILP for step < 30.
- Reactive reinforce post-pass.
- Anticipated reinforce post-pass.
"""
from __future__ import annotations
import os
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_OPENING_MILP", "1")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
from agents.baseline.main import agent  # noqa: E402
