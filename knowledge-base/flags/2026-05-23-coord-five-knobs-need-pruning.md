# 2026-05-23 — flag: coord has 5 new knobs in 2 days; prune the
# ones that don't work

## State after sub coord v3 submit

Live rolling pair when sub coord v3 lands:
- **Newest:** sub coord v3 (just submitted) — Day-13 stack:
  deadline-bounded enumerate + smooth-ΔW endgame bonus +
  COORD_LEAF_FLOOR=2.0 + COORD_REDUCED_FLOOR=2.0 + demand-spread
  mixing (Option 3 LITE)
- **Older half:** sub 52935965 (orbitfix_kt_p23, μ=1091.3)

Self-evicted sub 52936894 (coord v2, never settled — was submitted
~10 min before v3).

## What changed in 2 days

5 env-var-gated features added between 2026-05-22 and 2026-05-23:

| Knob | Default | Behavior |
|---|---|---|
| `COORD_DELTA_W` | "1" | smooth-ΔW endgame bonus (λ_W=0.002) |
| `COORD_ATTACK_BONUS` / `COORD_DEFEND_BONUS` | "1" | per-kind gates |
| `COORD_LEAF_FLOOR` | 2.0 | reject bundles with tier2 < floor |
| `COORD_REDUCED_FLOOR` | 2.0 | Lagrangian breaks on reduced ≤ floor |
| `COORD_DEMAND_SPREAD` | "1" | mixing_weight = capacity/demand |

Day 11 coord was simpler and settled at μ=905.6. We've changed five
things at once; if v3 lifts/regresses, attributing to a single
feature is hard.

## Action required next session

Once sub coord v3's μ settles (~12-24h post-submit):

1. **Read v3 μ.** Single command, blocks all decisions.
2. **If v3 > 1100**: features collectively help. Tune
   COORD_OPP_CAPACITY_FACTOR and DEMAND_REACH_WINDOW.
3. **If v3 ∈ [900, 1100]**: at-parity with old coord (μ=905). The
   pile-up cancelled out. Need to PRUNE.
4. **If v3 < 900**: features collectively hurt. Need to PRUNE
   AGGRESSIVELY.

## Pruning order (in cases 3 and 4)

Highest suspicion → lowest:
- `COORD_DEMAND_SPREAD=0` (most fundamental change; revert to pure
  tier2 scoring)
- `COORD_LEAF_FLOOR=0` (revert to admit any-tier2 bundle)
- `COORD_REDUCED_FLOOR=0` (revert to original Lagrangian break)
- `COORD_DELTA_W=0` (disable endgame bonus; back to leaf-only)
- Keep the deadline fix (measured win) and code-review fixes
  (correctness only, no behavior change)

Method: n=4 swapped A/B vs orbitfix per knob, single env var
override at a time. If win rate within ±25pp of v3 baseline, the
knob is null → disable in production default.

## Persistent flag

This flag is "active" until v3 settles AND we've pruned. Reading the
flag at session-start is the trigger to check the Kaggle leaderboard
before any other decision.
