"""baseline_joint_aggr_consolidated_orbitfix — consolidated + orbital arrival safety.

Adds BASELINE_ORBITAL_SAFETY=1 to the live-submitted consolidated variant
(sub 52882014). Fixes a silent scoring bug where the chooser/proposer
treated orbiting targets as safe based on their CURRENT position even
when they would rotate into enemy territory by our arrival time —
PI 2026-05-21 observation: "attack rotating planets that rotate in...
at the time that we will hit with our fleet, the planet will be close
to opponents so that they can easily anticipate what we are doing."

The fix: when scoring a capture's expected hold post-arrival, pass
`arrival_eta` to `time_to_enemy_threat` so target & enemy positions
are predicted at the time of our arrival via `predict_relative`. Bug
fix in `lib/world_model.py:time_to_enemy_threat`, propagated via
`lib/scoring.py:expected_hold` and `agents/baseline/proposer.py:_quick_score`.
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
from agents.baseline.main import agent  # noqa: E402
