"""baseline_speedrun — favor_speedrun value head (planet-count-first).

The speedrun frame: capture all planets in minimum turns. No opp model;
all non-mine planets are obstacles with growing garrisons. Every
capture of a non-mine planet adds K_PLANET=50 to the leaf regardless
of who owned it — this gives the chooser a HARD positive Δ for every
capture, fixing the flat-Δ chooser-idle pathology the unified-favor
model exhibited in 4P.

PI directive (2026-05-26 PM, learning-iteration on sub 53032723):
"Go build the speedrunner. Aim for elimination in most games."
"""
from __future__ import annotations
import os
# Orbitfix env stack (byte-equivalent prologue).
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
# Speedrun head + calibrated knobs.
os.environ.setdefault("BASELINE_VALUE_HEAD", "speedrun")
os.environ.setdefault("BASELINE_SPEEDRUN_K_PLANET", "50")
os.environ.setdefault("BASELINE_SPEEDRUN_K_ACQUIRE", "0.1")
os.environ.setdefault("BASELINE_SPEEDRUN_K_SHIPS", "0.5")
os.environ.setdefault("BASELINE_SPEEDRUN_HOLD_HORIZON", "20")
os.environ.setdefault("BASELINE_SPEEDRUN_REACH_HORIZON", "25")
from agents.baseline.main import agent  # noqa: E402
