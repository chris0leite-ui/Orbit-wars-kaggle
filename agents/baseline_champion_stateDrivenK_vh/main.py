"""baseline_champion_stateDrivenK_vh — live champion + value head.

Builds on the live ladder champion `baseline_state_driven_k` (sub
#53280733, μ ≥ 1153.6) by activating the trained value head
(`BASELINE_VH_LAMBDA=1.0`). No Tier-2: this session's diagnostic
proved Tier-2 in rollouts is a ~50pp regressor on this base (see
audit/2026-06-02-tier2-in-rollouts-regression.md when written).

The value head's additive correction is now applied to BOTH solo and
joint candidates as of the Phase A patch
(`agents/baseline/chooser_trajectory.py` 2026-06-03). Pre-Phase-A, the
joint path silently skipped the VH boost — submitting VH-on with the
champion's `BASELINE_JOINT_AGGR=1` would have under-rated the
agent's core attack pattern.

The currently-shipped VH model (`data/value_head/value_head_model.txt`)
was trained on `baseline_pv_eta` self-play. This wrapper's first A/B
will read whether that model transfers to the state-driven-K base —
if yes, submit; if no, retrain on state-driven-K self-play (Phase D
in the plan) and re-A/B.

Env block source-of-truth: `scripts/_build_state_driven_k_bundle.sh`
header + `BASELINE_VH_LAMBDA=1.0`.
"""
from __future__ import annotations
import os

# Live-champion (state_driven_k) config — μ ≥ 1153.6.
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

# State-driven horizon K (per-target predictability, ceil 30 / floor 10).
os.environ.setdefault("BASELINE_STATE_DRIVEN_K", "1")
os.environ.setdefault("BASELINE_STATE_K_CEIL", "30")

# Value head active. Phase A patch (2026-06-03) ensures the boost
# applies to joint candidates too, not just solo.
os.environ.setdefault("BASELINE_VH_LAMBDA", "1.0")

# NO Tier-2 (BASELINE_OPP_TIER unset → defaults to "0" → lite_greedy
# in rollouts). Tier-2-in-rollouts axis is closed (Rule 37) — see
# session 2026-06-02 ablation A/B (12.5% vs 62.5% on state-driven-K).

from agents.baseline.main import agent  # noqa: E402
