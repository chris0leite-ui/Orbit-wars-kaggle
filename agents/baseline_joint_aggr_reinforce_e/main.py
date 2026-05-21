"""baseline_joint_aggr_reinforce_e — aggressive reinforce + leader-focus stack.

Combines:
- Reactive reinforce (T_loss < horizon)
- Anticipated reinforce with WIDER margin (1.7 vs 1.3 default)
- Leader-focus bonus 1.5x in 4P
"""
from __future__ import annotations
import os
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE_MARGIN", "1.7")
os.environ.setdefault("BASELINE_REINFORCE_MAX", "4")
os.environ.setdefault("BASELINE_LEADER_FOCUS", "1.5")
from agents.baseline.main import agent  # noqa: E402
