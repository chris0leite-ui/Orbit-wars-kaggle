"""orbitfix_kt_p23_nokt — orbitfix_kt_p23 env stack with KINEMATIC_TABLE OFF.

Diagnostic variant (2026-05-23): the KT bit-parity verification showed
that BASELINE_KINEMATIC_TABLE_ENABLED=1 produces 81% turn divergence
from OFF on seed 7 with Phase 2/3/4 disabled — meaning the substrate
itself silently changes the agent's policy, not just its speed. Live
ladder evidence is consistent: orbitfix vanilla μ=1165, our orbitfix_kt_*
submissions stuck at μ=994-1037.

This variant keeps every Phase 1+2+3 + code-review-fix layer that the
production orbitfix_kt_p23 has, EXCEPT KINEMATIC_TABLE_ENABLED. If this
variant beats orbitfix_kt_p23 locally, the table is the culprit.
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
# NOTE: KINEMATIC_TABLE_ENABLED deliberately NOT set (defaults OFF).
os.environ.setdefault("BASELINE_ADAPTIVE_K", "1")
os.environ.setdefault("COMPOSITE_FLEET_SURVIVAL_CHECK", "1")

from agents.baseline.main import agent  # noqa: E402
