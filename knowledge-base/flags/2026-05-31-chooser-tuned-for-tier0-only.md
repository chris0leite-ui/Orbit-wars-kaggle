# 2026-05-31 — Persistent flag: chooser is tuned for Tier 0 opp policy only

The chooser's wallclock budget (~1000 ms) is structurally calibrated
for a cheap opp policy (`lite_greedy`, ~0.5 ms/call). Any heavier opp
model in `opp_actions_for_snap` cuts per-candidate validation count
6-8× and emit count 3-4×, regressing win rate by ~50 percentage points
vs `launch_rules`.

This is not Tier 2-specific. Any future RL/IL/distilled opp model
that costs > ~1 ms/call will hit the same wall unless either:
- The chooser is restructured (event-driven horizon, fewer opp calls
  per rollout), or
- The opp model is engineered to stay at ~0.5 ms/call regardless of
  sophistication (distillation, tiny architecture)

Flag tag: `chooser-budget-binds-opp-model-quality`.
Owners: any branch attempting opp-model upgrades inside the chooser.
Re-evaluate when: chooser architecture changes, or wallclock cap
raised by the comp / by a chooser refactor.
