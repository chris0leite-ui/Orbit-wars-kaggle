"""baseline_joint_aggr_consolidated_orbitfix — consolidated + orbital arrival safety.

Adds BASELINE_ORBITAL_SAFETY=1 to the live-submitted consolidated variant
(sub 52882014). Fixes a silent modeling bug where the chooser/proposer
treated orbiting targets as safe based on their CURRENT position even
when they would rotate into enemy territory by our arrival time —
PI 2026-05-21 observation: "attack rotating planets that rotate in...
at the time that we will hit with our fleet, the planet will be close
to opponents so that they can easily anticipate what we are doing."

Fix scope (f1774a7 + 2026-05-22 audit pass — bugs B1-B7):
- `lib/world_model.py:time_to_enemy_threat` predicts target + enemy
  positions at our arrival; in-flight fleet filter is strict `>`
  (simultaneous-arrival resolved by combat); LATER inbound waves now
  surface via `incoming_enemy_eta_after`; orbiting targets get a
  5-iteration fixed-point on `enemy_eta_travel`.
- `lib/scoring.py:expected_hold` threads `arrival_eta` through.
- `agents/baseline/proposer.py:cheap_marginal_value`,
  `_target_holdable_after_capture`, `_target_cost_parity_ok` predict
  target/opp/ally positions at arrival_step.
- `lib/missions/snipe.py:_followon_hold_estimate`, `_best_followon`
  predict followon + enemy positions at our arrival.

Every site is gated on BASELINE_ORBITAL_SAFETY=1; default OFF preserves
backwards compat with sub 52882014.
"""
from __future__ import annotations
import os
os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
from agents.baseline.main import agent  # noqa: E402
