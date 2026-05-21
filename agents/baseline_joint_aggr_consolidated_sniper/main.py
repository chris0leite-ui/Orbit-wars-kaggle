"""baseline_joint_aggr_consolidated_sniper — consolidated + sniper bundle.

PI 2026-05-21 directive: "when we have idle planets and when it's clear
that they can bundle to really shoot even across the whole map, fast to
attack one of the biggest opponent planets, then do it."

Adds emit_sniper_strikes post-chooser: when total reserve > 300 ships
AND a source has >= 80 idle ships AND a non-our planet with production
>= +4 is reachable, the largest idle source fires a sized strike
(margin 1.2x predicted garrison) with optional follow-on reinforcement
from other idle sources to bolster the post-capture garrison.
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
os.environ.setdefault("BASELINE_SNIPER", "1")
from agents.baseline.main import agent  # noqa: E402
