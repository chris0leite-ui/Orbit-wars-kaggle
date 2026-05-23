"""orbitfix_kt_p23_snowball — orbitfix_kt_p23 + anti-fragmentation knobs.

Same env-var stack as `agents/orbitfix_kt_p23/`, plus the three Change
A/B/C gates from the 2026-05-23 anti-fragmentation plan:

- BASELINE_MIN_FLEET_BY_ETA=1
    Per-ETA min ship floor: candidates dropped if ships < schedule(eta).
    eta <= 5  : 2 ships (no-op)
    eta 6-15  : 5 ships
    eta > 15  : 10 ships

- BASELINE_MIN_SOURCE_SHIPS_TO_EMIT=5
    Source garrison floor for FIRE-NOW candidates: planets below 5 ships
    don't emit immediate launches (wait_N candidates still allowed so the
    source can plan a delayed launch once it grows).

- BASELINE_JOINT_TARGET_PRIORITY=1
    Joint pair enumeration ordered by target value (enemy-owned-fattest
    first, then aggregate cheap_delta) so the JOINT_MAX_PAIRS budget
    concentrates on highest-value targets rather than dict-insertion order.

A/B target: `orbitfix_kt_p23` (both share Phase 1/2/3 + code-review fixes).
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
# Snowball knobs (Change A/B/C, 2026-05-23):
os.environ.setdefault("BASELINE_MIN_FLEET_BY_ETA", "1")
os.environ.setdefault("BASELINE_MIN_SOURCE_SHIPS_TO_EMIT", "5")
os.environ.setdefault("BASELINE_JOINT_TARGET_PRIORITY", "1")

from agents.baseline.main import agent  # noqa: E402
