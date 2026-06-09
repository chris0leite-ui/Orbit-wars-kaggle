"""State-driven-K corpus-gen wrapper — VH OFF, Tier-2 OFF.

Used by `scripts/gen_b2_corpus.py` for Phase D self-play (2026-06-03).
The corpus must be generated WITHOUT the value head active (otherwise
the trained head would be optimizing against its own predictions —
circular). It must also be generated WITHOUT Tier-2 in rollouts
(Phase 1 ablation showed Tier-2 contributes ~50pp regression).

Env block mirrors the live champion `baseline_state_driven_k` minus
the VH and Tier-2 levers. BASELINE_VH_TRACE_FEATURES is NOT set here —
the gen-script subprocess sets it externally so this wrapper stays
safe to use as a plain self-play agent if needed elsewhere.
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
os.environ.setdefault("BASELINE_PV_ETA", "1")
os.environ.setdefault("BASELINE_LAUNCH_RULES", "1")
os.environ.setdefault("BASELINE_CAPTURE_HORIZON_K", "10")
os.environ.setdefault("BASELINE_KINEMATIC_TABLE", "1")
os.environ.setdefault("BASELINE_STATE_DRIVEN_K", "1")
os.environ.setdefault("BASELINE_STATE_K_CEIL", "30")

from agents.baseline.main import agent  # noqa: E402
