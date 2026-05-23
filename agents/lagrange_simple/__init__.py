"""lagrange_simple — the simplest precision-physics Lagrangian agent.

Three files:
  score.py    enumerate (src, tgt, launch_tick) candidates via
              aim_and_eta + predict_fleet_fate + predict_garrison_at,
              with B1-B7 orbital safety semantics (BASELINE_ORBITAL_SAFETY=1).
  dual.py     3-sweep Lagrangian: per-target argmax under shadow prices
              λ_s on per-source ship budgets, subgradient update each sweep.
  main.py     agent entry point.
"""
