"""orbitfix_kt_p23 — orbitfix_kt + Phase 2 (adaptive K) + Phase 3 (leaf
in-flight fate check).

Same env-var stack as `agents/orbitfix_kt/`, with two additional flags
flipped ON so the chooser bumps rollout horizon on critical turns and
the composite leaf calls predict_fleet_fate for every credited fleet.

A/B target: orbitfix_kt (Phase 1 deterministic budget shared via the
chooser module; Phase 2 + 3 are the only differences).
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
os.environ.setdefault("BASELINE_ADAPTIVE_K", "1")
os.environ.setdefault("COMPOSITE_FLEET_SURVIVAL_CHECK", "1")
# Forward-deployment incentive (2026-05-23): pull ships toward enemy
# planets to address the passivity diagnosis. Replaces the positionless
# `favor` head with `favor_hybrid_attack_pull` which adds a per-ship
# weight by closeness to the nearest enemy planet.
os.environ.setdefault("BASELINE_VALUE_HEAD", "attack_pull")
os.environ.setdefault("BASELINE_ATTACK_PULL_WEIGHT", "0.5")
os.environ.setdefault("BASELINE_ATTACK_PULL_DECAY", "30.0")
# Distance-aware min fleet size filter (2026-05-23): rejects offensive
# launches sized too small for the target distance. Addresses the "many
# small fleets to far targets" pattern observed in the JM replay (sub
# 52959167 vs JM: 5-10 ship launches at distance 46-53 all wasted).
# Slope 0.15: d=30→5 ships, d=50→8 ships, d=70→11 ships.
os.environ.setdefault("BASELINE_MIN_FLEET_BY_DISTANCE", "1")
os.environ.setdefault("BASELINE_MIN_FLEET_SLOPE_PER_UNIT", "0.15")

from agents.baseline.main import agent  # noqa: E402
