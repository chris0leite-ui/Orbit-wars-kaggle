"""baseline_joint_aggr_consolidated — coherent 4P stack.

Composes:
- AGGR (BASELINE_JOINT_AGGR=1 + AGGR caps) — multi-source same-target.
- JOINT-in-4P fix — lifts the 2P-only gate so the joint scoring runs
  in 4P, eliminating the silent EV double-count flagged by code review.
- Analytical-track proposer fix (strategic stockpile + bundle blind
  spot) — in agents/baseline/proposer.py.
- Reactive + anticipated reinforce post-pass — defends predicted-to-fall
  and inbound-enemy-thinned friendly planets.
- Neutral-capture bonus — tilts chooser toward early-game territorial
  grab; phase_c's snowball mechanism.

Drops (proved regressive or neutral):
- Opening MILP (regression in n=16).
- Leader-focus (bounce-asymmetry; net 0).
- Drain_idle_rear (audit-falsified 2026-05-18).
- Tight stagnation drain (no-op at threshold 80).
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
