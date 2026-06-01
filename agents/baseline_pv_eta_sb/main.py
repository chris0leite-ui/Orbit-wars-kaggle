"""baseline + pv_eta + launch_rules + size_balance — minimal stack.

Diagnostic: jsr (with slot_res + joint_sync) lost 2/32 vs champion locally.
This variant strips slot_res and joint_sync to isolate whether the base
(pv_eta + launch_rules + size_balance) is the regression source, or if
slot_res / joint_sync are.

Mirror of 00JzI's size_balance bundle (which settled live mu=1097) plus
our pv_eta foundation.

Env vars set:
- Standard peak-orbitfix preamble.
- BASELINE_PV_ETA=1 (foundation lock).
- BASELINE_LAUNCH_RULES=1, BASELINE_CAPTURE_HORIZON_K=10 (champion mechanism).
- BASELINE_SIZE_BALANCE=1 (00JzI's live mu=1097 mechanism).
- KINEMATIC_TABLE_ENABLED=0 (mutable-singleton bug fix).
"""

from __future__ import annotations

import os as _os

# peak orbitfix preamble
_os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
_os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
_os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
_os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
_os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
_os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
_os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")

# Foundation.
_os.environ.setdefault("BASELINE_PV_ETA", "1")

# Champion mechanism.
_os.environ.setdefault("BASELINE_LAUNCH_RULES", "1")
_os.environ.setdefault("BASELINE_CAPTURE_HORIZON_K", "10")

# Live-evidenced attack fix (00JzI sub 53248277 mu=1097).
_os.environ.setdefault("BASELINE_SIZE_BALANCE", "1")

# Kinematic-table bug fix.
_os.environ["KINEMATIC_TABLE_ENABLED"] = "0"

from agents.baseline.main import agent  # noqa: E402,F401
