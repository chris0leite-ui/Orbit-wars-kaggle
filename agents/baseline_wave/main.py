"""baseline_wave v2 — orbitfix + three wave-attack incentive terms.

v1 shipped four terms (coord penalty + bleed + inflight-HHI + stockpile);
A/B vs orbitfix at default coefficients was 0/8 elim, 1/8 win-by-reward.
Root cause (see /root/.claude/plans/do-1-2-rosy-popcorn.md "Plan v2"):
- Coord penalty had the wrong sign — it penalised joints vs solos.
- Inflight-at-leaf HHI measured the wrong distribution — fleets had
  already arrived by leaf time for typical eta < horizon.
- Bleed and stockpile-pressure were directionally OK.

v2 fixes (env-var renames included):
- BASELINE_COORD_BONUS    α·Σ(cohort_ships²)/Σ(cohort_ships) over per-
                          arrival-step cohorts in a joint coalition,
                          ADDED to joint score. Combat-rule-1 stacking.
- BASELINE_BLEED_PENALTY  β·min(excess, P·t−s)·γ^t per emission
                          (unchanged). Synergises with the coord bonus:
                          bleed lowers solo cheap_delta, joints get a
                          clean rollout score, so the joint > solo
                          gradient widens.
- BASELINE_STOCKPILE_PENALTY  ε·Σ excess² per owned planet, ε dropped
                              from 0.005 → 0.001 so it provides a
                              gentle "drain pressure" floor rather than
                              dominating the leaf signal.

Rollback: delete this directory; the orbitfix bundle is unaffected.
"""
from __future__ import annotations
import os

# Inherit the orbitfix peak stack (sub 52912707, μ=1165.4).
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")

# Wave-incentive layer (v2).
os.environ.setdefault("BASELINE_COORD_BONUS", "1")
os.environ.setdefault("BASELINE_COORD_ALPHA", "0.5")
os.environ.setdefault("BASELINE_BLEED_PENALTY", "1")
os.environ.setdefault("BASELINE_BLEED_BETA", "0.05")
os.environ.setdefault("BASELINE_STOCKPILE_PENALTY", "1")
os.environ.setdefault("BASELINE_STOCKPILE_EPS", "0.001")
os.environ.setdefault("BASELINE_STOCKPILE_TARGET", "50")

from agents.baseline.main import agent  # noqa: E402
