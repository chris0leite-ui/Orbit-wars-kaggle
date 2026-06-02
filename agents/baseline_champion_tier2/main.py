"""baseline_champion_tier2 — launch_rules_universal champion + Tier-2 opp.

The first single-variable graft of hqNVM's ML stack onto 00JzI's
champion strategy substrate. The champion (launch_rules_universal,
historical peak μ=1183.7) sits unmodified; this wrapper adds ONLY
the distilled Tier-2 opponent model (`lib.opp_model.trained_logreg_policy`,
booster at `data/opp_distill/distill_booster.txt`). Value head stays OFF
(BASELINE_VH_LAMBDA defaults to 0.0) so this is a single-variable A/B
vs the un-ported champion.

Hypothesis under test: the +14 μ Tier-2 lift attributed in hqNVM-isolated
A/B transfers when the underlying chooser is the joint-aggressive
champion stack instead of pv_eta. If μ at first settle ≥ ~1130 (champion
historical peak − Wilson noise), Tier-2 transfers cleanly; submit
`baseline_champion_ml_vh` next (champion + Tier-2 + retrained VH).
If μ settles ≤ ~1080, escalate to PI before any VH retrain.

Source-config provenance: `scripts/_build_state_driven_k_bundle.sh`
header (launch_rules_universal config block), with
BASELINE_STATE_DRIVEN_K omitted (state-driven horizon is a separate
lever and is not part of the historical peak config).
"""
from __future__ import annotations
import os

# Champion (launch_rules_universal) config — historical peak μ=1183.7.
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
os.environ.setdefault("BASELINE_PV_ETA", "1")
os.environ.setdefault("BASELINE_LAUNCH_RULES", "1")
os.environ.setdefault("BASELINE_CAPTURE_HORIZON_K", "10")
os.environ.setdefault("BASELINE_KINEMATIC_TABLE", "1")

# Distilled-ladder Tier-2 opp model.
os.environ.setdefault("BASELINE_OPP_TIER", "2")
os.environ.setdefault("BASELINE_OPP_FILTER_THRESHOLD", "0.15")

# Value head stays OFF — single-variable A/B for Tier-2 attribution.
# BASELINE_VH_LAMBDA unset → defaults to 0.0 in agents/baseline/_value_head.py.

from agents.baseline.main import agent  # noqa: E402
