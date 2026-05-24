"""baseline_wave v3 — orbitfix + multi-source wave proposer.

v1 (4 terms) + v2 (coord-bonus on arrival cohorts) both 0/8 elim vs
orbitfix; the alpha-sweep (0.5 / 2 / 5) was non-monotonic and never
cleared the gate. Root cause documented in
/root/.claude/plans/do-1-2-rosy-popcorn.md "Plan v4": the existing
proposer enumerates (src, tgt, ships, wait_N) per source independently
— wave-shaped actions don't exist in the candidate set, so reshaping
the value head can't pick them. The chooser can only pick what the
proposer enumerates.

v3 introduces a WAVE PROPOSER (proposer.enumerate_wave_candidates) that
directly emits (target, [(src, ships, angle, wait_N), ...]) coalitions
where multiple sources fire at different times so all arrivals land on
the same step — exploiting combat rule 1 (additive stacking).

Layer summary:
- BASELINE_WAVE_PROPOSER       turn on the new proposer pathway
- BASELINE_BLEED_PENALTY       lowers solo cheap_delta from stockpiled
                               sources (widens solo<wave gradient)
- BASELINE_STOCKPILE_PENALTY   gentle drainage floor in the value head
- BASELINE_COORD_BONUS=0       subsumed by the wave proposer; keeping
                               both would double-credit the same waves

Rollback: delete this directory + revert the three chunks in
proposer.py/main.py/chooser_trajectory.py — orbitfix bundle unaffected.
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

# Wave-proposer layer (v3).
os.environ.setdefault("BASELINE_WAVE_PROPOSER", "1")
os.environ.setdefault("BASELINE_WAVE_MAX_PER_TURN", "8")
os.environ.setdefault("BASELINE_WAVE_K", "4")
os.environ.setdefault("BASELINE_WAVE_MARGIN", "2")
os.environ.setdefault("BASELINE_WAVE_TEMPO_GUARD", "8")

# Supporting wave-incentive terms (v2 carryovers).
os.environ.setdefault("BASELINE_COORD_BONUS", "0")  # subsumed by wave proposer
os.environ.setdefault("BASELINE_BLEED_PENALTY", "1")
os.environ.setdefault("BASELINE_BLEED_BETA", "0.05")
os.environ.setdefault("BASELINE_STOCKPILE_PENALTY", "1")
os.environ.setdefault("BASELINE_STOCKPILE_EPS", "0.001")
os.environ.setdefault("BASELINE_STOCKPILE_TARGET", "50")

from agents.baseline.main import agent  # noqa: E402
