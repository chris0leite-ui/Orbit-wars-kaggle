"""baseline_ev_per_ship — orbitfix env stack + EV-per-ship chooser sort.

Adds BASELINE_SORT_BY_EV_PER_SHIP=1 on top of the byte-equivalent
orbitfix env-var set. The chooser code (chooser_trajectory.py) was
extended in commit 0a8308f to honour this var; default OFF preserves
orbitfix behaviour. With this var ON, the final candidate sort uses
score/ships instead of score, prioritising EV-per-ship over total EV.

Probe evidence (panel A/B 2026-05-25): 15/20 = 75% pooled win-rate
across {orbitfix, baseline_wave, v7_0_drop_one, v4_planner} at 5 games
each (PI standard procedure, 250-step cap, no seat switch).
Per-opponent: 4/5, 3/5, 4/5, 4/5.

Diagnostic deltas (variant vs orbitfix baseline, same focal code):
- 4P launches/turn 0.23 -> 1.68 (7x increase)
- 4P owned planet-turns 243 -> 3154 (13x ownership across game)
- `ranked_out` drop 28.4% -> 3.4% (per-ship sort converts wait-N
  commits into fire-now launches that succeed)

PI directive 2026-05-25 PM: "submit then run 4P".
"""
from __future__ import annotations
import os
# Orbitfix env stack (byte-equivalent prologue).
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
# New: EV-per-ship sort key (commit 0a8308f, default OFF in chooser).
os.environ.setdefault("BASELINE_SORT_BY_EV_PER_SHIP", "1")
from agents.baseline.main import agent  # noqa: E402
