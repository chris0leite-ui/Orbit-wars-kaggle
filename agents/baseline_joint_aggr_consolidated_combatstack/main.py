"""baseline_joint_aggr_consolidated_combatstack — consolidated + combat-stack drain.

Stacks excess from idle-rear sources directly onto NON-OUR planets that
already have friendly inbound (we're attacking them). Directly addresses
PI 2026-05-21 image observation: "our large planet sits fleets away
from combat, we do not cluster at combat." Distinct from drain_stagnant_rear
(target = closer-to-front friendly) — this drains TO the fight, not near it.
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
os.environ.setdefault("BASELINE_COMBAT_STACK", "1")
from agents.baseline.main import agent  # noqa: E402
