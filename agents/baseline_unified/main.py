"""baseline_unified — strategic head (favor_strategic) with unified leaf.

The unified strategic value head (commit 3a054c7):
- F1 = my_ships − max_o(opp_ships_o)              [max-of-opps in any P]
- F2 = (my_prod_disc − max_o(opp_prod_o_raw)) · pv [asymmetric Term A]
- Term B = capture-feasible forward-reach
- Term C = per-opp finishing pressure + dead-slot credit
- Discrete elim_bonus when FINISH_BONUS=0 in 4P+

One leaf for both 2P and 4P. No mode switch on num_seats. The asymmetric
Phase-F discount on my_prod (calibrated defensive "fear" gradient) is
preserved across modes; max-of-opps controls F2 scale so the asymmetric
form works without a symmetric counterpart in 4P.

Learning-submit (PI explicit sign-off 2026-05-26 PM, Rule 1):
"submit. I want to observe what actually happens. We need to gather
data. It's about learning."
"""
from __future__ import annotations
import os
# Orbitfix env stack (byte-equivalent prologue, matches baseline_ev_per_ship).
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
# Strategic head with calibrated knobs (matches the experiments we just ran).
os.environ.setdefault("BASELINE_VALUE_HEAD", "strategic")
os.environ.setdefault("BASELINE_HOLD_HORIZON", "20")
os.environ.setdefault("BASELINE_FORWARD_REACH_WEIGHT", "0.5")
os.environ.setdefault("BASELINE_FORWARD_REACH_HORIZON", "15")
os.environ.setdefault("BASELINE_FINISH_BONUS", "50")
os.environ.setdefault("BASELINE_FINISH_THRESHOLD", "200")
from agents.baseline.main import agent  # noqa: E402
