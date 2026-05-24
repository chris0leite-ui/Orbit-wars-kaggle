"""CONSOLIDATION phase: chooser_trajectory + defensive reinforce.

Delegates to `agents.baseline.main.agent` with the env-var profile of
the live team-peak agent `baseline_joint_aggr_consolidated_orbitfix`
(sub 52912707, μ=1165.4). That agent already wires:

  - BASELINE_CHOOSER=trajectory       (baseline default)
  - BASELINE_JOINT_AGGR=1              (multi-source same-target lift)
  - BASELINE_REINFORCE_EMIT=1          (defensive reinforce post-pass)
  - BASELINE_REINFORCE_ANTICIPATE=1    (preemptive defensive reinforce)
  - BASELINE_NEUTRAL_BONUS=2.0         (early-game neutral capture bias)
  - BASELINE_ORBITAL_SAFETY=1          (B1-B7 orbital arrival fix)

…and explicitly does NOT enable the offensive post-passes
(BASELINE_IDLE_DRAIN, BASELINE_STAGNANT_DRAIN, BASELINE_COMBAT_STACK,
BASELINE_SNIPER) — those are a separate axis per the plan.

The env vars are set with `setdefault` so a local A/B harness can
override any of them via the process environment.
"""
from __future__ import annotations

import os

# Match `agents/baseline_joint_aggr_consolidated_orbitfix/main.py` exactly.
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
# NOTE: BASELINE_VALUE_HEAD=phi setdefault lives in
# agents/buildup_planner/main.py (which runs BEFORE the baseline import
# in this module, so the bundle's earlier-running baseline setdefault to
# "hybrid" loses).

# Ablation knob (2026-05-23): switch CONSOLIDATION's chooser between
# trajectory (default) and roi (closed-form ROI prior + sniper/drain post-
# passes). Default trajectory matches the previously-shipped configuration;
# set BUILDUP_PLANNER_CHOOSER=roi to test the ROI chooser as a different
# equilibrium against full-strength baseline. The env var IS NOT read by
# baseline.main directly — we set BASELINE_CHOOSER via setdefault below
# so the actual switch happens in baseline.main's dispatcher. NOTE: this
# DOES leak to baseline-as-opp via env var, so the matchup stays symmetric
# in chooser choice. The hypothesis being tested is whether the ROI+sniper+
# drain equilibrium gives more room for FINISHER's edge to dominate.
_CHOOSER = os.environ.get("BUILDUP_PLANNER_CHOOSER", "trajectory").strip().lower()
if _CHOOSER == "roi":
    os.environ.setdefault("BASELINE_CHOOSER", "roi")
    # ROI chooser path in baseline.main also applies the offensive post-
    # passes: idle drain, stagnant drain, combat stack, sniper strikes.
    # Default OFF in trajectory mode; ON in roi mode (no env flip needed
    # — the post-passes are unconditionally called inside the roi branch
    # at agents/baseline/main.py:988-992).
else:
    os.environ.setdefault("BASELINE_CHOOSER", "trajectory")

from agents.baseline.main import agent as _baseline_agent  # noqa: E402


def step(obs, configuration=None) -> list[list]:
    """Run one CONSOLIDATION turn via the validated baseline pipeline."""
    return _baseline_agent(obs, configuration)
