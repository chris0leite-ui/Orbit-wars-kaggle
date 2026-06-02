"""baseline_champion_stateDrivenK_tier2_t07 — calibration variant of
`baseline_champion_stateDrivenK_tier2` with the Tier-2 emission filter
raised from 0.15 -> 0.7.

Diagnostic finding (2026-06-02 seed=0 vs launch_rules_universal): the
0.15 threshold makes trained_logreg_policy emit ~dozens of low-
confidence predicted opp launches per rollout step. Inside the
chooser's K-step rollout, that floods the opp's hand with "credible"
counter-attacks at virtually every target, which combined with
state-driven-K (K=opp_earliest_contest_tick) and launch_rules (drop
launches arriving after K) compresses K and nukes most of our own
candidate launches. Result: 482 idle ships at step 90, zero launches
emitted. Pan-game n=32 A/B vs launch_rules_universal: 4/32 = 12.5%.

Hypothesis: raising the threshold to 0.7 reduces trained_logreg_policy
to ONLY its high-confidence predictions (typically 0-1 launches per
opp turn). Rollout opp looks like a sane player rather than a frenzied
attacker; chooser stops planning against phantom counter-launches;
under-launch failure mode lifts.

Single-variable A/B vs baseline_champion_stateDrivenK_tier2: ONLY
BASELINE_OPP_FILTER_THRESHOLD differs (0.7 here vs 0.15 there).
Same Tier-2 dispatch (active post-Phase 0), same state-driven K, same
launch_rules.
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

# Tier-2 with HIGH-CONFIDENCE-ONLY threshold (0.7 vs sibling's 0.15).
os.environ.setdefault("BASELINE_OPP_TIER", "2")
os.environ.setdefault("BASELINE_OPP_FILTER_THRESHOLD", "0.7")

from agents.baseline.main import agent  # noqa: E402
