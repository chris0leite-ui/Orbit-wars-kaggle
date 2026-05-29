"""baseline + pv_eta time-discount + per-target value-head chooser term.

Reframe B.2 (2026-05-29). Wraps `agents.baseline.main.agent` with the
pv_eta foundation lock plus the value-head additive coefficient.

Env-var defaults:

- `BASELINE_PV_ETA=1` — enable pv_eta γ-discount on candidate Δ (live
  champion mechanism at μ=1154.8; foundation lock 2026-05-29).
- `BASELINE_VH_LAMBDA=1.0` — λ for the additive `λ · vh_output` term
  added to each solo candidate's score. The head is a 15-d LightGBM
  regressor trained to predict K=10 ship-delta; its output IS in
  ship units, so λ=1.0 is the natural scale.
- Peak orbitfix preamble (joint AGGR, neutral bonus, orbital safety,
  reinforce) — matches `submissions/_imported/baseline_pv_eta.py:7-15`.

At `BASELINE_VH_LAMBDA=0`, this agent is byte-equivalent to bare pv_eta
— same parity check as Reframe A's λ=0 invariant.

Bundling: `scripts/bundle_pv_eta_vh.py` inlines the inner bundle and
patches `agents.baseline._value_head._VH_MODEL_B64` with the gzip+base64
regressor dump.
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

# pv_eta time-discount (Reframe B foundation lock).
_os.environ.setdefault("BASELINE_PV_ETA", "1")

# Value-head additive term — default ON at λ=1.0 (head's natural scale).
# Set BASELINE_VH_LAMBDA=0 at invocation time for the byte-equivalent
# parity check vs bare pv_eta.
_os.environ.setdefault("BASELINE_VH_LAMBDA", "1.0")

# Re-export top-level agent.
from agents.baseline.main import agent  # noqa: E402,F401
