"""baseline_kt — baseline_full brain + kinematic-table substrate.

Same brain as `baseline_full` (sub 52893236, μ≈1078, rolling-pair
floor). One difference: `KINEMATIC_TABLE_ENABLED=1` flips the env-
gated kinematic-table priming inside `agents/baseline/main.py:agent`
(default OFF for every other agent).

The freed predict_relative cycles flow into the chooser's wallclock-
adaptive candidate loop (`agents/baseline/chooser.py`,
WALLCLOCK_BUDGET_MS=600); no manual K-bump.

Bundling note: this wrapper is for LOCAL EVAL via fast.py only. The
project bundler refuses wrapper-style cross-agent imports (friction
tag `bundle-agent-doesnt-inline-from-baseline-main`). For an eventual
submission, bundle `agents/baseline` directly and prepend the env-var
setdefaults to the output — see `submissions/baseline_full.py:6-19`
for the established post-injection pattern.
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
os.environ.setdefault("BASELINE_STAGNANT_DRAIN", "1")
os.environ.setdefault("BASELINE_COMBAT_STACK", "1")
os.environ.setdefault("BASELINE_SNIPER", "1")
os.environ.setdefault("KINEMATIC_TABLE_ENABLED", "1")

from agents.baseline.main import agent  # noqa: E402
