"""baseline_joint_aggr_unleashed — full coherent expansion theory.

AGGR + proposer filters off (let candidates through) + reactive +
anticipated reinforce (defense post-pass to handle the false-counter
fallout) + neutral-capture bonus (tilt toward neutrals, exactly the
mechanism phase_c uses to snowball).
"""
from __future__ import annotations
import os
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("PROPOSER_COST_PARITY", "off")
os.environ.setdefault("PROPOSER_HOLD_FEASIBILITY", "off")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
from agents.baseline.main import agent  # noqa: E402
