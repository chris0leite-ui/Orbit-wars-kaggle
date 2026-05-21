"""baseline_joint_aggr_consolidated_bothdrain — consolidated + both drains.

Both drain_stagnant_rear (drain to closer-to-front friendly) AND
drain_combat_stack (drain to non-our planet with friendly inbound).
Drains are sequential and idempotent on used_srcs, so a src consumed
by stagnant drain won't be re-drained by combat stack.
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
os.environ.setdefault("BASELINE_COMBAT_STACK", "1")
from agents.baseline.main import agent  # noqa: E402
