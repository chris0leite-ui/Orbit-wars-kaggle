"""orbitfix_kt_p23_smart — orbitfix_kt_p23 + Phase 4 smart-opp leaf reaction.

Same env-var stack as `agents/orbitfix_kt_p23/`, plus
`BASELINE_OPP_SMART_LEAF=1` so the FINAL N rollout steps swap in
`top_tier_mirror_policy` instead of `lite_greedy_policy`. The smart-opp
window is `BASELINE_OPP_SMART_LEAF_WINDOW` ticks (default 5) — wide
enough for opp's defensive launches to propagate to their targets before
the leaf is evaluated.

A/B target: `orbitfix_kt_p23` (both share Phase 1 deterministic chooser
+ Phase 2 adaptive K + Phase 3 leaf fate check; only Phase 4 differs).
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
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
os.environ.setdefault("KINEMATIC_TABLE_ENABLED", "1")
os.environ.setdefault("BASELINE_ADAPTIVE_K", "1")
os.environ.setdefault("COMPOSITE_FLEET_SURVIVAL_CHECK", "1")
os.environ.setdefault("BASELINE_OPP_SMART_LEAF", "1")

from agents.baseline.main import agent  # noqa: E402
