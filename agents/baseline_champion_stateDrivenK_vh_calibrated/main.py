"""Calibrated VH wrapper: state-driven-K + centered + attenuated VH.

Diagnosis (2026-06-02 session): the raw VH head output has mean +102
and std 78 on plausible candidate features (probe: 500 rows of
N(0,1)-distributed 14-d features). The chooser's base `j_score` is
O(10-100). Adding `λ=1.0 · head_out` swamps the base score — every
candidate gets +90..+225 added, unlocking the emit gate for almost
everything. Result: 5x launch-volume blowup, 0/32 vs the simplest
opponent panel.

Fix:
  1. CENTERING — subtract BASELINE_VH_BIAS=102 inside both VH call
     sites in chooser_trajectory.py, so the boost is mean-zero
     instead of strongly positively biased. (Done as a code change
     this session; BIAS=0.0 default preserves the pre-fix behavior.)
  2. ATTENUATION — λ=0.1 so the std of the centered correction is
     ~7.8, in the same order of magnitude as base score noise. The
     VH becomes a real correction signal, not a dominant override.

Expected: launch volume returns to the no-VH baseline (62.5%-class)
while VH discrimination contributes a small lift. If the wrapper
clears Wlo ≥ 0.55 vs launch_rules_universal, the original VH
"transferability" failure was a wiring bug, not a model-quality
issue; the shipped VH can stay live with this calibration.

If still ≤ baseline, the model itself is broken (likely a
training-corpus or sidecar-stale artifact issue — model.txt was
overwritten 2026-06-02 13:51 UTC but meta/history are May 28-29);
proceed to Phase D retrain on state-driven-K self-play.
"""
from __future__ import annotations
import os

# State-driven-K live champion (μ ≥ 1153.6) env block.
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

# Calibrated VH: attenuated and centered.
os.environ.setdefault("BASELINE_VH_LAMBDA", "0.1")
os.environ.setdefault("BASELINE_VH_BIAS", "102.0")
# Phase A joint VH stays on — the bug is integration, not the joint
# fix. With centering+attenuation, joint and solo paths are parity.

from agents.baseline.main import agent  # noqa: E402
