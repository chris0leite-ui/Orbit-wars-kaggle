"""baseline + pv_eta time-discount + ACCEPTED-candidate tracing.

Reframe B.1 diagnostic probe (2026-05-29). Wraps
`agents.baseline.main.agent` with the pv_eta foundation locked AND the
chooser's accepted-candidate set persisted to a per-game JSONL via
`BASELINE_ACCEPTED_TRACE`. The runner script sets that env var
per-worker before importing this agent.

Invariant: `BASELINE_ML_LAMBDA=0.0` here, so the score persisted in the
accepted trace is the raw pv_eta γ-discounted delta. If you flip ML on
in this wrapper, the trace will record the ml-adjusted score and break
the B.1 analysis.

Reads:
- `BASELINE_ACCEPTED_TRACE` (set by the runner) — JSONL path.
- `BASELINE_WALLCLOCK_MS` (default 100; runner can override).
"""

from __future__ import annotations

import os as _os

# peak orbitfix preamble (mirrors submissions/_imported/baseline_pv_eta.py:7-15)
_os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
_os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
_os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
_os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
_os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
_os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
_os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
_os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")

# pv_eta time-discount (foundation lock).
_os.environ.setdefault("BASELINE_PV_ETA", "1")

# Lambda-0 invariant: never enable ML for the probe.
_os.environ.setdefault("BASELINE_ML_LAMBDA", "0.0")

# Kinematic-table bug fix (d50654a / 232307c). Explicit hard override
# guards local A/Bs where the opponent bundle setdefault-enables the
# table — opponent's setdefault wins by default because we don't set
# the var. Explicit assignment ensures it's "0" regardless of opponent
# behavior. No-op on Kaggle (isolated processes).
_os.environ["KINEMATIC_TABLE_ENABLED"] = "0"

# Re-export top-level agent.
from agents.baseline.main import agent  # noqa: E402,F401
