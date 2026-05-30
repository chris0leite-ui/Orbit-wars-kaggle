"""baseline + pv_eta time-discount + Tier-2 learned opp model.

Tier 2 (2026-05-31). Wraps `agents.baseline.main.agent` with the
pv_eta foundation lock + the Tier-2 opp policy
(`lib.opp_model.trained_logreg_policy`). The Tier-2 opp policy runs
inside every `fast_sim.rollout` step the chooser uses for candidate
leaf scoring — replacing the cheap `lite_greedy_policy` default with
shot-validator-filtered Tier-1 candidates.

Env-var defaults:

- `BASELINE_PV_ETA=1` — enable pv_eta γ-discount on candidate Δ
  (foundation lock 2026-05-29).
- `BASELINE_OPP_TIER=2` — Tier 2 learned opp model in the chooser's
  rollout (PI-ratified 2026-05-30 PM after B.3 18/32 = 56.2 % marginal).
- `BASELINE_VH_LAMBDA=0` — value-head OFF for the first cut so the
  Tier-2 lift is attributed cleanly (not mixed with B.3 head lift).
  Flip to 1.0 in a follow-up if Tier 2 alone clears.
- `BASELINE_OPP_FILTER_THRESHOLD=0.30` — booster P(success) cutoff
  for filtering opp candidates (PM5 value).
- Peak orbitfix preamble (joint AGGR, neutral bonus, orbital safety,
  reinforce) — matches `submissions/_imported/baseline_pv_eta.py:7-15`.

At `BASELINE_OPP_TIER=0`, this agent reverts to bare pv_eta behaviour
(byte-equivalent parity check). The Tier-2 booster artifact lives in
`data/shot_validator/validator_booster.txt`; the bundler patches its
gzip+base64 dump into `lib.opp_model._OPP_BOOSTER_B64`.
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

# Tier-2 learned opp model in the chooser's fast_sim rollout. The
# policy lives in lib.opp_model.trained_logreg_policy and uses the
# shot-validator booster (data/shot_validator/validator_booster.txt) to
# filter Tier-1 candidates per opp step. Threshold below.
_os.environ.setdefault("BASELINE_OPP_TIER", "2")
_os.environ.setdefault("BASELINE_OPP_FILTER_THRESHOLD", "0.30")

# Value head OFF for clean Tier-2 attribution (the B.3 head's marginal
# lift is what we are trying to unblock by upgrading the opp model;
# layering both at once would confound the A/B).
_os.environ.setdefault("BASELINE_VH_LAMBDA", "0.0")

# Kinematic-table bug fix (d50654a / 232307c). The table
# (lib/kinematic_table.py) is a module-global singleton whose mutable
# state leaks across seats in any in-process play, silently regressing
# winrate ~25 pp per the SEU7P isolation A/B (31% -> 56%). This wrapper
# never enables it, but opponent bundles in local A/Bs may setdefault
# it to "1" in the shared process — that wins because we never set it.
# Explicit set (not setdefault) overrides any opponent-side setdefault.
# No-op on Kaggle (agents run in isolated processes).
_os.environ["KINEMATIC_TABLE_ENABLED"] = "0"

# Re-export top-level agent.
from agents.baseline.main import agent  # noqa: E402,F401
