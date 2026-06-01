"""baseline jsr + buildup→aggression mode switch (ML chooser → ROI on ship advantage).

Extends `baseline_pv_eta_vh_dist_jsr` with a per-turn chooser override:
when our total ships > BASELINE_ROI_SWITCH_RATIO × max opponent seat's
ships, the dispatch in `agents/baseline/main.py` hands off to the
closed-form ROI chooser (`agents/baseline/chooser_roi.py`) instead of
the trajectory chooser. ROI bypasses the value-head's per-turn-EV veto
and emits attack-favoring closed-form solo + coalition launches.

PI direction (2026-06-01): the ML-enriched jsr agent is strong at
buildup but fails to convert advantage into wins. The composite stack's
K-step rollout vetoes attacks whose per-turn EV is negative even when
campaign EV is positive — buildup compounds but never gets cashed in.
This wrapper keeps jsr's buildup strength and adds the conversion arm.

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
# Boosts attack-class scores within the trajectory chooser when our ships
# > 1.3 × max(opp). Fires in the band [1.3, 1.5) before the hard switch.
_os.environ.setdefault("BASELINE_SURPLUS_AGGRESSION", "1")
_os.environ.setdefault("BASELINE_SURPLUS_BOOST", "1.5")
_os.environ.setdefault("BASELINE_SURPLUS_RATIO", "1.3")

# Systematic attack feature B — persistent target injection.
_os.environ.setdefault("BASELINE_PERSISTENT_ATTACK", "1")
_os.environ.setdefault("BASELINE_PERSISTENT_ATTACK_MAX_INJECT", "1")

# Buildup chooser: trajectory (ML composite).
_os.environ.setdefault("BASELINE_CHOOSER", "trajectory")

# Mode-switch — hand off to ROI chooser when ship advantage crosses
# RATIO. ROI bypasses the value-head veto; structurally attack-capable
# via closed-form solo + coalition scoring. Threshold 1.5 sits above
# the SURPLUS_AGGRESSION soft-boost band (1.3) so trajectory gets one
# ratio band to convert with boosted scoring before we override.
_os.environ.setdefault("BASELINE_ROI_SWITCH_ENABLED", "1")
_os.environ.setdefault("BASELINE_ROI_SWITCH_RATIO", "1.5")

# Kinematic-table bug fix (non-negotiable).
_os.environ["KINEMATIC_TABLE_ENABLED"] = "0"

from agents.baseline.main import agent  # noqa: E402,F401
