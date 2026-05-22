# Q: Is orbitfix's μ=1165 a real ceiling or just a recent peak?

Submitted 2026-05-22 04:56. μ=1165 settled. +63μ over the prior
strongest local-tested variant (`baseline_joint_aggr_consolidated`
at μ=1102) AND over the prior team peak (`composite_a2_hybrid` at
μ=1149.2, EVICTED).

The lift came from a B1-B7 modeling-bug sweep on
`time_to_enemy_threat`, `_target_holdable_after_capture`,
`_target_cost_parity_ok`, `_followon_hold_estimate` etc. — NOT
from value function or search architecture.

Open questions:
1. Are there more sibling modeling bugs of the same class (other
   primitives that mis-handle entity types)?
2. Does the 2P-only restriction on `BASELINE_ORBITAL_SAFETY`
   matter — would a 4P-aware extension lift further?
3. Is the top-10 cliff (μ≈1440) reachable by stacking more
   physics-modeling fixes, or does it require a different agent
   class (precision-artillery / shot-validator)?

## Why this matters

The LP-family value-function axis is now saturated below μ=1165
per the 2026-05-23 A/Bs. The next reach is either:
- More orbitfix-class physics audits (high precedent, +60μ band).
- A different agent class entirely (precision-physics / ML
  validator — H14 in `state/hypothesis-board.md`).

Without an answer to (1), we don't know which off-ramp pays off.
A 1-day Rule-47 sweep through `lib/world_model.py`,
`lib/joint_solver/predicate.py`, `lib/scoring.py`, `lib/missions/`,
and the chooser path for similar "primitive assumes wrong entity
behaviour" patterns would resolve the question cheaply.
