"""baseline + pv_eta time-discount + per-shot ML logit chooser term.

Reframe A (2026-05-29). Wraps `agents.baseline.main.agent` with three
env-var defaults:

- `BASELINE_PV_ETA=1`  — enable the pv_eta γ-discount on candidate Δ
  (the live champion mechanism at μ=1154.8). Verbatim port living in
  `agents/baseline/chooser_trajectory.py`.
- `BASELINE_ML_LAMBDA=<sweep value>` — λ for the additive centered-logit
  term `λ * (logit(P_success) - logit(0.5))` added to each candidate's
  score. The Booster underneath is the same one
  `agents/baseline_validated` uses as a filter (45-d schema, 1000ms
  wallclock training corpus).
- Peak orbitfix preamble (joint AGGR, neutral bonus, orbital safety,
  reinforce) — matches the bundle preamble in
  `submissions/_imported/baseline_pv_eta.py:7-15`.

At `BASELINE_ML_LAMBDA=0` (default) this agent is byte-equivalent to
bare pv_eta — parity-test gate per the Reframe A plan.

Bundling: `scripts/bundle_validator.py`-style — inline the inner bundle,
patch `agents.baseline._ml_logit._BOOSTER_B64` with the gzip+base64
LightGBM dump, rename `def agent(` → `def _inner_agent(` inside the
inner. See plan file for the workflow.
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

# pv_eta time-discount (Reframe A foundation lock)
_os.environ.setdefault("BASELINE_PV_ETA", "1")

# ML logit additive term — default OFF (parity with bare pv_eta).
# Sweep per `/root/.claude/plans/squishy-bouncing-hickey.md`:
#   λ ∈ {0.1, 0.3, 1.0} × σ(delta)/σ(logit P) from the Step-0 probe.
_os.environ.setdefault("BASELINE_ML_LAMBDA", "0.0")

# Re-export top-level agent.
from agents.baseline.main import agent  # noqa: E402,F401
