# 2P A/B — favor vs projected_rank_diff

Branch: `claude/strategy-framework-design-OyoYR` @ `71da9da`
Date: 2026-05-19
Tooling: `python -m scripts.ab_variants --variant favor VALUE_HEAD_CHOICE=0
--variant projected VALUE_HEAD_CHOICE=2 --agent agents/baseline --seeds 32
--workers 8 --candidate projected`
Wallclock: 1187.8 s

## Result

| Variant | W/L/D | Winrate | Wilson 95% |
|---|---:|---:|---|
| favor | 31/33/0 | 48.4% | [36.6%, 60.4%] |
| **projected** | **33/31/0** | **51.6%** | **[39.6%, 63.4%]** |

Per-anchor gate (Wilson lo ≥ 0.55): **FAIL** (Wlo=0.396 vs favor).

Tournament JSONs:
- `audit/tournaments/20260519T084505Z.json` (per-game outcomes)
- `audit/tournaments/ab-20260519T084505Z.json` (summary + gate verdict)

## Interpretation

Statistical tie. Projected has a +3.2 pp point estimate over favor but the
Wilson CI brackets 50%; we cannot distinguish from random play.

**This was the expected outcome.** In 2P, `agents/baseline/value.favor` uses
`max` over the single opponent (same as projected) and a γ-discounted PV
horizon. With `PV_GAMMA = 1.0` (the current default in `lib/scoring.py:89`)
the pv_horizon collapses to `EPISODE_STEPS - step` — a linear horizon, the
same shape as projected's `λ × P × (T − step)` term. Modulo the λ scaling
(0.05) and the in-flight composite credit (which projected adds on top),
favor and projected are parameterising nearly the same scalar in 2P.

The unification claim (single state-function objective replacing favor +
A2-hybrid graft) was always going to be most visible **in 4P**, where:
- `favor` switches to `sum_of_opps` (over-penalises us — opp1+opp2+opp3
  aggregated).
- `projected` keeps `max` (only the leader matters; opps fighting each
  other shrinks our cost for free).

The 2P A/B doesn't exercise that semantic gap. A flat result is consistent
with "the new head doesn't break 2P" and orthogonal to "the new head helps
in 4P."

## Tooling note

`scripts/ab_variants.py` was extended to include `lib/value_heads.py` in
`PATCHABLE_PATHS` (`71da9da`). The value-head selector was converted from
the string env var `BASELINE_VALUE_HEAD` to a numeric constant
`VALUE_HEAD_CHOICE` in `lib/value_heads.py` so the regex-based patcher
can swap variants cleanly. Env var remains as a back-compat fallback when
`VALUE_HEAD_CHOICE == 0`.

The numeric constant deliberately omits its type annotation
(`VALUE_HEAD_CHOICE = 0` not `VALUE_HEAD_CHOICE: int = 0`); the patcher's
regex matches `NAME = number` but not annotated assignments. First A/B
attempt failed with `not found as a top-level assignment` until the
annotation was removed.

## Next experiments

1. **4P A/B** — the actual unification test. `scripts/ffa_panel.py` or
   `scripts/ffa_tournament.py` with the two bundles (`submissions/_ab/
   favor.py`, `submissions/_ab/projected.py`).
2. **Cross-opponent panel** — `fast.py eval agents/baseline --vs-panel
   default` with `VALUE_HEAD_CHOICE` patched in each run. Measures the
   tournament-style lift against the v7_0 / v4_planner / v3.5.1 anchors,
   which is closer to live-ladder calibration.
3. **λ sweep** — if 4P also lands flat, try `PROJECTION_LAMBDA` ∈
   {0.1, 0.2, 0.5} via ab_variants. λ=0.05 is a guess; the in-flight
   credit (CAPTURE_REWARD_WEIGHT=0.05) was the calibration anchor.

## Verdict

Do not submit. Run 4P A/B first; that's the actual hypothesis test.
