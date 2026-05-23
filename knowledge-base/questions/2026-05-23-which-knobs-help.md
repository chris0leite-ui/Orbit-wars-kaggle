# 2026-05-23 — open question: which of coord's 5 new knobs help vs
# hurt vs are null?

The single fact that picks the next axis. Answer requires v3's μ AND
individual-knob A/Bs.

## Predicted ranges per knob (rough)

- `COORD_DEMAND_SPREAD=1` (demand-spread mixing): **directional
  unknown**. Theory says it helps when our demand > opp capacity
  (late game with many planets). Could hurt if opp's actual defense
  is concentrated and the mixing rewards bundles that genuinely
  fail in isolation. n=0 A/Bs so far — only this submission tests it.

- `LEAF_FLOOR=2.0`: **likely helps**. Sub 52936894 (FLOOR=0) had the
  "too many small/far fleets" symptom the PI observed; FLOOR=2.0
  blocks tactical-marginal bundles. Risk: if it filters too many,
  the agent goes back to idling.

- `REDUCED_FLOOR=2.0`: **likely helps**, same reasoning as
  LEAF_FLOOR. May overlap with LEAF_FLOOR in practice (both block
  marginal bundles from different angles).

- `COORD_DELTA_W=1` (smooth-ΔW endgame bonus, λ=0.002): **A/B null
  at n=4**. Outcomes unchanged from off; only game lengths shifted.
  Possibly does nothing useful at this λ.

- Per-kind gates `COORD_ATTACK_BONUS`/`COORD_DEFEND_BONUS`:
  **diagnostic only**. Default ON; we never observed differences
  from disabling.

## Branch decisions (post v3 settle)

- **v3 μ ≥ 1100**: keep all features; tune
  COORD_OPP_CAPACITY_FACTOR + DEMAND_REACH_WINDOW. Then upgrade
  Option 3 from LITE to canonical (per-opp shadow price).

- **v3 μ ∈ [900, 1100]**: at-parity with old coord. PRUNE order:
  DEMAND_SPREAD off → REDUCED_FLOOR=0 → LEAF_FLOOR=0 → DELTA_W=0.
  Stop at first knob whose removal improves A/B.

- **v3 μ < 900**: features collectively HURT. Revert to
  pre-Day-13 coord (keep ONLY the deadline fix and code-review
  correctness fixes). Re-A/B from a clean baseline.

## Secondary questions

- **Does the floor at 2.0 over-block?** With LEAF_FLOOR=0, coord
  emitted small/far bundles wastefully. With LEAF_FLOOR=2.0, does
  it now under-emit on legitimately-good captures whose leaf is
  ~1? Calibration probe with mid-game obs would tell us.

- **Is demand-spread mixing's "capacity = opp.ships" too generous?**
  Real defenders allocate maybe 30% to defense, 70% to offense. If
  so, COORD_OPP_CAPACITY_FACTOR=0.3 would be more realistic.

- **Does DEMAND_REACH_WINDOW=12 capture real responders?** Minimal's
  response horizon is ~25 turns. We may be UNDER-counting responders
  → over-estimating mixing_weight → too defended-pessimistic in
  practice.
