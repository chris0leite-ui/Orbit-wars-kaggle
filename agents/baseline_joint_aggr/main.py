"""baseline_joint_aggr — variant: structural change A/B vs iter_baseline.

Three changes from baseline (all gated by env vars):
  - `BASELINE_JOINT_AGGR=1` lifts the `used_tgts` lock on solo + joint
    emits in `agents.baseline.chooser_trajectory.choose_trajectory`.
    Multi-source-same-target SOLO emits become legal (combat rule 1:
    same-owner same-step arrivals stack).
  - `BASELINE_JOINT_TOP_K=5` (default 3): JOINT enumeration considers
    top-5 solos per target instead of top-3.
  - `BASELINE_JOINT_MAX_PAIRS=60` (default 20): global JOINT pair cap
    raised 3x. With more pairs evaluated, joint coverage of the failing-
    solo bucket is wider.

`used_srcs` lock is RETAINED — relaxing it would over-budget the source
planet's ships (chooser scoring doesn't account for shared ship budget).

Per-A/B-call env override: set BEFORE the import below so the chooser's
module-level constants pick them up at first import.
"""

from __future__ import annotations

import os

# Set BEFORE importing baseline so module-level constants bake in.
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")

from agents.baseline.main import agent  # noqa: E402
