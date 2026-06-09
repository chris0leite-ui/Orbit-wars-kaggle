"""Diagnostic sibling: state-driven-K + VH on SOLO candidates only.

Same env block as `baseline_champion_stateDrivenK_vh`, but sets
BASELINE_VH_JOINT=0 to disable the Phase A joint-VH boost. The chooser
falls back to applying VH on solo prerank rows only — pre-Phase-A
behavior on the post-Phase-A code path.

Purpose: A/B against `launch_rules_universal_local` to attribute the
VH collapse (0/32 = 0.0% with joint VH on) between:
  (a) the shipped VH being miscalibrated for the state-K base, OR
  (b) the Phase A joint-leg aggregation (λ · sum(per-leg head_outs))
      being too aggressive.

Decision:
  - Solo-only VH ≈ baseline (62.5%) → the issue is JOINT aggregation;
    tune Phase A (mean instead of sum, or halve λ for joints).
  - Solo-only VH also collapses (≤30%) → the shipped VH is the issue;
    proceed to Phase D (retrain on state-K self-play).
  - Mid-range → both contribute; retrain takes priority, then revisit.

NOT a submission candidate. This wrapper exists for the diagnostic.
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

# VH on solo, OFF on joints (the diagnostic).
os.environ.setdefault("BASELINE_VH_LAMBDA", "1.0")
os.environ.setdefault("BASELINE_VH_JOINT", "0")

from agents.baseline.main import agent  # noqa: E402
