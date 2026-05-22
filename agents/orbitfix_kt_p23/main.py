"""orbitfix_kt_p23 — orbitfix_kt + Phase 2 (adaptive K) + Phase 3 (leaf
in-flight fate check).

Same env-var stack as `agents/orbitfix_kt/`, with two additional flags
flipped ON so the chooser bumps rollout horizon on critical turns and
the composite leaf calls predict_fleet_fate for every credited fleet.

A/B target: orbitfix_kt (Phase 1 deterministic budget shared via the
chooser module; Phase 2 + 3 are the only differences).
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

from agents.baseline.main import agent  # noqa: E402
