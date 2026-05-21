"""baseline_joint_aggr_neutral — AGGR + proposer fix + reinforce + neutral-grab bonus.

Adds NEUTRAL_BONUS_WEIGHT=2.0 (and EXTRA=1.5 for step<50) to the
chooser. Hypothesis from seed=5 trace (loss): we capture only 6
neutrals vs 73 enemy planets; phase_c snowballs via aggressive
neutral grab. Tilting the chooser toward neutrals should accelerate
territorial expansion in the opening + early mid-game.
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
from agents.baseline.main import agent  # noqa: E402
