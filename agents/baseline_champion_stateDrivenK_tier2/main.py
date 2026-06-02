"""baseline_champion_stateDrivenK_tier2 — live champion + Tier-2 opp.

Builds on the current live ladder champion `baseline_state_driven_k`
(sub #53280733) by adding the distilled Tier-2 opponent model. The K
lever is **state-driven** (per-target predictability horizon, not the
step-decay version): for each target `K_target = clamp(floor=10,
ceil=30, opp_earliest_contest_tick)`. Uncontested target → ceil
(commit long); contested at tick T → K=T. Documented as "the
principled form of the adaptive step-schedule" — see
`agents/baseline/launch_rules.py:capture_horizon_k` design §3 Lever A
and `knowledge-base/concepts/contest-aware-conversion-design.md`.

This is the FIRST agent in the project whose chooser actually exercises
`lib.opp_model.trained_logreg_policy` during rollout — the
chooser._select_opp_policy() dispatch bug that silently routed tier-2
through lite_greedy was fixed alongside this wrapper (Phase 0,
2026-06-02). Sub #53295205 (baseline_champion_tier2) ran the buggy code
path; whatever μ it settles to is NOT a Tier-2 transfer signal.

Per PI directive (2026-06-02): this wrapper's first A/B captures the
JOINT signal of (a) Tier-2 finally being active + (b) the upgrade from
launch_rules_universal base → state-driven-K base. Attribution between
the two is conflated on this A/B; the strategic question is "does the
combined stack beat the live state_driven_k champion?".

Live calibration prior: state_driven_k (sub #53280733) μ ≥ 1153.6 as
of 2026-06-02. Upper bound for this wrapper is roughly state_driven_k's
μ plus whatever Tier-2 contributes when actually live.

Env block source-of-truth: `scripts/_build_state_driven_k_bundle.sh`
header (the bundle script that produced the live champion).
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

# Distilled-ladder Tier-2 opp model.
os.environ.setdefault("BASELINE_OPP_TIER", "2")
os.environ.setdefault("BASELINE_OPP_FILTER_THRESHOLD", "0.15")

# Value head stays OFF — Phase 1 isolates state-driven-K + (now-live)
# Tier-2 transfer; VH retrain + VH-on wrapper come in Phases 2-4.
# BASELINE_VH_LAMBDA unset → defaults to 0.0 in agents/baseline/_value_head.py.

from agents.baseline.main import agent  # noqa: E402
