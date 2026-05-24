"""baseline_wave — orbitfix + four wave-attack incentive terms.

Adds four env-var-gated incentives on top of the orbitfix peak agent
(sub 52912707, μ=1165.4) so wave behavior emerges from differentiable
incentives instead of a hardcoded phase machine:

- BASELINE_COORD_PENALTY      α·spread² on joint-coalition arrival times
                              (combat-rule-1 stacking incentive).
- BASELINE_BLEED_PENALTY      β·min(excess, P·t−s)·γ^t per emission
                              (opportunity-cost of dribbling from
                              stockpiled planets).
- BASELINE_HHI_BONUS          δ·HHI·Σs on inflight fleet sizes
                              (concentration reward).
- BASELINE_STOCKPILE_PENALTY  ε·(ships−target)² per owned planet
                              (forces drainage of large idle stockpiles
                              — the "inactive planets" hedge).

All four default OFF in core; this shim turns them ON with calibration
defaults documented in /root/.claude/plans/do-1-2-rosy-popcorn.md. The
base orbitfix stack (BASELINE_JOINT_AGGR, BASELINE_ORBITAL_SAFETY,
BASELINE_NEUTRAL_BONUS, BASELINE_REINFORCE_EMIT, …) is inherited so the
agent never regresses below the orbitfix path's structural strength.

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

# Wave-incentive layer.
os.environ.setdefault("BASELINE_COORD_PENALTY", "1")
os.environ.setdefault("BASELINE_COORD_ALPHA", "0.2")
os.environ.setdefault("BASELINE_BLEED_PENALTY", "1")
os.environ.setdefault("BASELINE_BLEED_BETA", "0.05")
os.environ.setdefault("BASELINE_HHI_BONUS", "1")
os.environ.setdefault("BASELINE_HHI_DELTA", "0.1")
os.environ.setdefault("BASELINE_STOCKPILE_PENALTY", "1")
os.environ.setdefault("BASELINE_STOCKPILE_EPS", "0.005")
os.environ.setdefault("BASELINE_STOCKPILE_TARGET", "50")

from agents.baseline.main import agent  # noqa: E402
