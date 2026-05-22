"""orbitfix_kt — orbitfix (sub 52912707, μ=1175 on Kaggle) + kinematic table.

Same env-var stack as `agents/baseline_joint_aggr_consolidated_orbitfix`
(the current ladder ceiling), plus `KINEMATIC_TABLE_ENABLED=1`. The
kinematic-table priming hook lives in `agents/baseline/main.py:863-870`
and the Phase 3a wait_N filter fix lives in
`agents/baseline/proposer.py:996-1011` — both fire automatically for any
agent that sets the env var. Brain otherwise identical to orbitfix.

This is the clean A/B target for evaluating Phase 3a: same flag stack,
only differences are the kinematic table substrate + the closed H44
wait_N gap.
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

from agents.baseline.main import agent  # noqa: E402
