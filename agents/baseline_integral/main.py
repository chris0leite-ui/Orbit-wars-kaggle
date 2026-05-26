"""baseline_integral — terminal-ship-integral leaf (favor_integral_ships).

Closed-form predictor of terminal ship-count differential at T_END
assuming current ownership freezes:

    V = [my_ships + Σ_{p∈mine}    π_p · (T_END - t)]
      − max_o [ships_o + Σ_{p∈o} π_p · (T_END - t)]

Intentionally minimal — no MIN_FLEET_SIZE, no wait-N timers, no
threat-ETA discount, no forward-reach, no finishing pressure. PI
2026-05-26: strategy emerges from the rollout, not the leaf
(CLAUDE.md Rule 40, modeling-correctness over restriction-tuning).

Keeps the orbitfix chooser-side env stack (JOINT_AGGR / REINFORCE /
NEUTRAL_BONUS / ORBITAL_SAFETY) — those shape candidate enumeration,
not the leaf. Drops the strategic-head knobs (HOLD_HORIZON /
FORWARD_REACH_* / FINISH_*) because the integral leaf does not read
them.

The solo benchmark can override `INTEGRAL_T_END` via subprocess env to
match the game's `episodeSteps` cap.
"""
from __future__ import annotations
import os
# Orbitfix chooser stack (byte-equivalent prologue to baseline_unified).
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
# Integral-ships leaf — v2 is physics-aware (uses
# lib.trajectory.walk_existing_fleet_fate to discount in-flight fleets
# predicted to die in transit). v1 fallback via BASELINE_VALUE_HEAD=
# integral_ships still works.
os.environ.setdefault("BASELINE_VALUE_HEAD", "integral_v2")
os.environ.setdefault("INTEGRAL_T_END", "500")
from agents.baseline.main import agent  # noqa: E402
