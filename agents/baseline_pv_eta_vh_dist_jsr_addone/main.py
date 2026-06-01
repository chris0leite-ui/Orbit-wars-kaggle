"""baseline jsr + buildup→aggression mode switch (trajectory → v7_search add_one).

Extends `baseline_pv_eta_vh_dist_jsr` with a per-turn chooser override:
when our total ships > BASELINE_ROI_SWITCH_RATIO × max opponent seat's
ships, the dispatch in `agents/baseline/main.py` hands off to
`lib.v7_search.choose(enumerator_mode="add_one", K=10)` instead of the
trajectory chooser. add_one extends the incumbent by ONE more launch
from an idle source, K=10 rolls each variant out, and the parity floor
refuses to add anything that loses ground.

PI direction (2026-06-01): the closed-form ROI handoff was falsified
(A/B 43.8% vs jsr at n=16, wallclock max=1445ms) — closed-form ROI is
myopic about cumulative cross-turn source-drain. v7_search.choose's
K=10 rollout sees the drain; add_one is structurally protected from
greedy over-commit by the parity floor.

Default behavior with BASELINE_ROI_SWITCH_ENABLED unset: byte-identical
to baseline_pv_eta_vh_dist_jsr. Rule 46 parity preserved.
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

# Foundation: pv_eta time-discount.
_os.environ.setdefault("BASELINE_PV_ETA", "1")

# Composite layer 1: distilled-ladder opp model (Tier 2 v2).
_os.environ.setdefault("BASELINE_OPP_TIER", "2")
_os.environ.setdefault("BASELINE_OPP_FILTER_THRESHOLD", "0.15")

# Composite layer 2: B.3 CRN-advantage value head.
_os.environ.setdefault("BASELINE_VH_LAMBDA", "1.0")

# Attack fix 1: per-class scoring slots.
_os.environ.setdefault("BASELINE_SLOT_RESERVATION", "3/2/2")

# Attack fix 2: synchronized multi-source coalitions.
_os.environ.setdefault("BASELINE_JOINT_SYNC", "1")
_os.environ.setdefault("BASELINE_JOINT_SYNC_MAX_PAIRS", "5")
_os.environ.setdefault("BASELINE_JOINT_SYNC_SRC_K", "1")

# Attack fix 3: arrival-correct + source-safe sizing.
_os.environ.setdefault("BASELINE_SIZE_BALANCE", "1")

# Champion mechanism: universal K=10 launch-discipline validator.
_os.environ.setdefault("BASELINE_LAUNCH_RULES", "1")
_os.environ.setdefault("BASELINE_CAPTURE_HORIZON_K", "10")

# Systematic attack feature A — surplus-aggression bias (soft transition).
_os.environ.setdefault("BASELINE_SURPLUS_AGGRESSION", "1")
_os.environ.setdefault("BASELINE_SURPLUS_BOOST", "1.5")
_os.environ.setdefault("BASELINE_SURPLUS_RATIO", "1.3")

# Systematic attack feature B — persistent target injection.
_os.environ.setdefault("BASELINE_PERSISTENT_ATTACK", "1")
_os.environ.setdefault("BASELINE_PERSISTENT_ATTACK_MAX_INJECT", "1")

# Buildup chooser: trajectory (ML composite).
_os.environ.setdefault("BASELINE_CHOOSER", "trajectory")

# Mode-switch — hand off to v7_search add_one when ship advantage
# crosses RATIO. add_one extends incumbent by ONE more launch from an
# idle source; K=10 rollout scores each variant; parity floor refuses
# regressions. Structurally protected from over-commit (the ROI failure
# mode). Threshold 1.5 sits above SURPLUS_AGGRESSION soft band (1.3).
_os.environ.setdefault("BASELINE_ROI_SWITCH_ENABLED", "1")
_os.environ.setdefault("BASELINE_ROI_SWITCH_RATIO", "1.5")
_os.environ.setdefault("BASELINE_AGGRESSION_CHOOSER", "v7_add_one")

# v7_search budget — tight to leave margin for trajectory chooser's
# setup, ledger tick, propose pre-pass, decorators, and launch_rules.
# v7's parity floor returns incumbent on watchdog timeout — safe to
# budget tightly. Origin: ROI A/B v2 wallclock max=1445ms over the
# 1000ms cap; this run has to stay under.
_os.environ.setdefault("BASELINE_V7_WALLCLOCK_MS", "500.0")
_os.environ.setdefault("BASELINE_V7_K", "10")

# Kinematic-table bug fix (non-negotiable).
_os.environ["KINEMATIC_TABLE_ENABLED"] = "0"

from agents.baseline.main import agent  # noqa: E402,F401
