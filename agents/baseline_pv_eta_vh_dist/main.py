"""baseline + pv_eta time-discount + Tier-2 distilled-ladder opp model.

Tier 2 v2 (2026-05-31). REPLACES the falsified filter-based Tier-2 design
(audit/2026-05-31-postmortem-tier2-falsification.md, Rule 37 axis closed).

New Tier 2 = sub-ms opp predictor distilled from top-10% Kaggle leaderboard
2P replays. Architecture: lite_greedy-style candidate enumeration +
LightGBM booster scoring + threshold. Lives in
`lib.opp_model.trained_logreg_policy` (function name preserved so the
chooser-side selector at `agents/baseline/chooser.py:32` and the bundler
patch site work unchanged).

Env-var defaults:

- `BASELINE_PV_ETA=1` — enable pv_eta γ-discount on candidate Δ
  (foundation lock 2026-05-29).
- `BASELINE_OPP_TIER=2` — Tier 2 distilled opp model in the chooser's
  rollout.
- `BASELINE_VH_LAMBDA=0` — value-head OFF for the first cut so the
  Tier-2 lift is attributed cleanly (not mixed with B.3 head lift).
  Flip to 1.0 in a follow-up if Tier 2 alone clears the A/B gate.
- `BASELINE_OPP_FILTER_THRESHOLD=0.30` — booster P cutoff for emit
  decision (training-time default; tune per per-Phase-6 A/B).
- Peak orbitfix preamble (joint AGGR, neutral bonus, orbital safety,
  reinforce) — matches `submissions/_imported/baseline_pv_eta.py:7-15`.

At `BASELINE_OPP_TIER=0`, this agent reverts to bare pv_eta behaviour.
On any booster load/scoring failure the Tier 2 policy falls back to
`lite_greedy_policy(obs)` — never silent garbage launches.

The booster artifact lives in `data/opp_distill/distill_booster.txt`;
`scripts/bundle_pv_eta_vh_dist.py` patches its gzip+base64 dump into
`lib.opp_model._OPP_BOOSTER_B64`.
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

# Tier-2 v2 distilled-ladder opp model in the chooser's fast_sim rollout.
# Predictor: `lib.opp_model.trained_logreg_policy` (rewritten 2026-05-31).
# Booster: `data/opp_distill/distill_booster.txt`.
_os.environ.setdefault("BASELINE_OPP_TIER", "2")
_os.environ.setdefault("BASELINE_OPP_FILTER_THRESHOLD", "0.30")

# Value head OFF for clean Tier-2 attribution. Phase 6c composes this
# wrapper's distilled opp WITH the B.3 head (flip lambda to 1.0).
_os.environ.setdefault("BASELINE_VH_LAMBDA", "0.0")

# Kinematic-table bug fix (d50654a / 232307c). The table is a module-
# global singleton whose mutable state leaks across seats in any
# in-process play, silently regressing winrate ~25 pp per the SEU7P
# isolation A/B. Explicit set (not setdefault) overrides any opponent-
# side setdefault. No-op on Kaggle (agents run in isolated processes).
_os.environ["KINEMATIC_TABLE_ENABLED"] = "0"

# Re-export top-level agent.
from agents.baseline.main import agent  # noqa: E402,F401
