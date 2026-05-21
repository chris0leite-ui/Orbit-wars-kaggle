"""baseline_joint_aggr_reinforce_f — AGGR + analytical proposer fix + (a)+(b).

Stack:
- Proposer fix (cherry-picked from analytical track adbfb5c):
  * Partial-budget candidates (bundle blind spot fix)
  * Strategic stockpile for high-prod own planets (capture_size floor)
- Reactive reinforce post-pass (a)
- Anticipated reinforce post-pass (b)

Tests both "PI's defense intuition expressed at proposer level" and
"my post-pass defense" simultaneously.
"""
from __future__ import annotations
import os
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
from agents.baseline.main import agent  # noqa: E402
