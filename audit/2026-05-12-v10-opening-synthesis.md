# 2026-05-12 — v10_opening: synthesis of opening-fix + drop-one search

## TL;DR

After comparing v9_opening (opening-conditional heuristic, σ-equiv,
v3.4 base) against the parallel team's live submissions (v4_planner
μ≈1058 + v7_0_drop_one μ≈1030), we built **v10_opening** which
combines both threads:

- **Architecture**: v7_0_drop_one — v3.5.1 incumbent + drop-one
  enumeration + fast_sim K=10 rollout + parity-floor fallback.
- **Opening layer** (step < 15, 2P only): NEUTRAL_BONUS=1.5×,
  COMET_BONUS=1.3×, opp-aware demotion 0.1× for targets opp reaches
  first. Applied to MISSION SCORES before settle_plan, so the
  incumbent the drop-one enumerator explores around is already
  opening-tuned.
- **σ-equiv layer**: reverted (v7.6 bisect: regresses drop-one by
  ~54pp per audit/2026-05-12-v4-planner-receding-horizon-pathology.md).

## Why this and not v9_opening alone

v9_opening was built on the σ-equiv branch with v3.4 base. Compared
against the live rolling-last-2:

| Aspect | v9_opening | v10_opening |
|---|---|---|
| Base | v3.4 | v3.5.1 |
| Search | none (heuristic only) | drop-one × fast_sim K=10 |
| σ-equiv | active (sym_hypot + tie-break) | reverted |
| Opening fix | yes | yes |
| Local μ proxy vs v3.4 | 93.8% W/D | (pending) |
| Estimated live μ | ~v3.4 (995) + ~10μ opening | ~v7_0 (1030) + ~10μ opening |

## Architecture map

```
agent(obs):
  world, model ← v3.5.1 base
  missions ← snipe(aggressive=True) + reinforce
  if step < 15 AND 2P:
      opp_targets ← predict_opp_first_targets(obs)  # v3.5.1 from opp POV
      missions    ← apply_opening_adjustments(missions, opp_targets)
  incumbent ← settle_plan(missions) + realize()
  if 4P:
      return incumbent
  candidates ← drop_one(incumbent)
  for c in candidates:  # watchdog 700ms
      score = fast_sim K=10 vs top_tier_mirror_policy opp
  return argmax (tie → incumbent)
```

The opening layer is a 30-LOC injection BEFORE settle_plan. Outside
the opening window v10 ≡ v7_0_drop_one.

## Gates

1. v10 vs v3.5.1, 8 seeds × 2 sides: **16/16 = 100% W/D** ✓
2. v10 vs v7_0_drop_one, 8 seeds × 2 sides: gate ≥50% W/D (in progress)
3. v10 self-play 4 seeds × 2 sides: gate ≥80% draws (in progress)
4. Bundle smoke: parity vs source — passes (no σ-equiv-affected paths)

## Eviction calculus

Rolling-last-2 currently:
- v4_planner #52579863 μ≈1058.1
- v7_0_drop_one #52588156 μ≈1030.4 (PENDING)

Pushing v10 evicts v4_planner. Net gain if v10 settles ≥ 1058. If v10
matches v7_0's 1030 plus opening bonus, expected ~1040-1050 — likely
NEUTRAL or slightly NEGATIVE on the rolling pair.

Better play: wait for v7_0 to settle. If v7_0 lands ≥1058, v10 makes
sense (replaces older v4_planner). If v7_0 underperforms, v10 still
has the v7_0 base plus the opening lift; submit anyway.

## What v10 is NOT

- Not a Nash approximation. Still heuristic + 1-ply search.
- Not opening-only — search layer applies all 500 turns.
- Not σ-equiv. Two-seat asymmetry in self-play is plausible (env
  P0/P1 tie-break bias documented separately).
- Not a recapture agent. v7_0_drop_one base does NOT include
  recapture missions (v7.6 found recapture regresses).
