"""orbitfix_kt_deep — orbitfix_kt + deeper rollout horizon (Direction A).

Same brain as `agents/orbitfix_kt` (and same env-var stack as the
ladder ceiling sub 52912707, μ=1175), plus a K-bump:

    MIN_HORIZON: 25 -> 40
    MAX_HORIZON: 40 -> 60

The kinematic-table substrate (Phase 1+2) cut per-position cost by
~60% on this brain. Direction A converts those freed cycles into
deeper rollout rather than wider candidate breadth. The hypothesis
the H44 audit suggested but Phase 3a could not test: the K=10-ish
rollout was structurally blind to wait_N + flight_eta deaths that
happen past its horizon. K=60 sees most of them.

`MIN_HORIZON` / `MAX_HORIZON` are read from env at
agents/baseline/proposer.py:29-30 (defaults preserved when env
unset, so every other agent — orbitfix, baseline_full, etc — keeps
its shipped horizon).
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
os.environ.setdefault("BASELINE_MIN_HORIZON", "40")
os.environ.setdefault("BASELINE_MAX_HORIZON", "60")

from agents.baseline.main import agent  # noqa: E402
