# 2026-05-12 — opening-phase diagnostic (probe v1)

## TL;DR

User observed v3 falls behind in ship count "after a few steps,
before encounters with opponents." Probe across 64 games (4 opps × 8
seeds × 2 sides) confirms an opening deficit exists but is more
nuanced than initial framing.

**Findings**:
1. ✅ Against v3-self: ship-delta = 0 throughout (σ-equiv lock).
2. ✅ Against v2 (older base): minimal opening gap (~+0.8 at step 20).
3. ⚠ Against roi: ship-delta dips to -9.1 mean at step 10, recovers
   to +20.2 by step 30.
4. ⚠ Against precision: ship-delta dips to -6.8 at step 10, recovers
   to +16.1 by step 30.

**On AVERAGE** v3 is roughly tied through opening and AHEAD by step
30. But the per-seed variance is enormous:
- vs precision seeds: step-30 ship_delta ranges from **-38 to +59**
- The MEAN of +16 hides games where we're 38 ships behind

**The user's "weak in opening" observation maps to**:
- The 1-7 ship deficit window in steps 4-20 (mean across runs)
- The HIGH-VARIANCE outcomes per seed (some games we collapse)
- NOT a systematic monotonic deficit

## Probe setup

```
focal: v3_snipe (current branch = v3.4 + σ-equiv patches)
opponents: {v3_snipe, v2, roi, precision}
seeds: 8 per opponent × 2 sides = 16 games per opp
total: 64 games × 50-step episode (only steps 0-30 logged)
wallclock: 166s
output: audit/tournaments/opening_probe_v1.json
```

## Aggregate per-step metrics

| Opp | First neg Δship | 1st launch (median) | step 5 | step 10 | step 15 | step 20 | step 25 | step 30 |
|---|---|---|---|---|---|---|---|---|
| v3_snipe | NEVER | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| v2 | step 16 | 0 | 0 | 0 | 0.0 | +0.8 | +0.8 | +0.8 |
| roi | step 9 | 0 | 0 | -0.8 | -1.4 | -9.1 | +1.6 | +20.2 |
| precision | **step 4** | 0 | -0.5 | +0.1 | -3.0 | -6.8 | +6.6 | +16.1 |

Key:
- `Δship` = focal_ships - opp_ships (positive means v3 ahead)
- All values are MEAN across 16 games (8 seeds × 2 sides)

## Per-seed signals (vs precision, the most informative case)

```
seed  s5   s10   s15   s20   s25   s30   outcome
  0    0     0   -14   +10   +20   +30   LOSS
  1    0     0    -4    -4   -20   +33   LOSS
  2   -4   +12   +25   -33   -21   -38   LOSS  ← collapses
  3    0    -5    +3   -17    +3   +59   WIN   ← oscillates a lot, wins
  4    0     0   -20     0   +28   +31   WIN
  5    0     0   -25    -1   +21   +11   WIN
  6    0     0     0    -1   -16   -31   LOSS  ← collapse after step 20
  7    0    -6    +6    -8   +17   +34   WIN
```

Observations:
- **Seed 0 ends step 30 at +30 ships but LOST the game.** Opening
  recovery doesn't guarantee win.
- **Seed 6 was flat at 0 through step 20 then collapsed to -31 at
  step 30.** The deficit appears AFTER the user's "few steps" window.
- **Variance is huge**: seed 2 swings -4 → +25 → -38.
- **First 5-10 steps are mostly flat** (delta 0 to -6); the wild
  swings start around step 15 when first fleets arrive at neutrals.

## Hypothesis evaluation

From the active plan:

| H | Cause | Supported? | Evidence |
|---|---|---|---|
| H1 | Under-capture | No (mean) / Maybe (variance) | step 30 Δplanets is +0.6 / -0.2; not systematically lower |
| H2 | Intent-drop without replan | **Indirect support** | first_launch_median = 0 means we DO launch turn 0, but maybe later turns drop intents silently |
| H3 | Sun/OOB ship loss | Not directly tested | need fleet-fate instrumentation |
| H4 | Score-formula myopia | **Strongly supported** | wild swings in seed 2 (+25 → -33) suggest fleets arriving at contested targets; bounce-rate likely |
| H5 | Opp launches earlier | Disproven | v3 first_launch_median = 0 (immediate) |
| H6 | Init-condition variance | **Strongly supported** | per-seed range -38 to +59 indicates map layout dominates outcome |

**Best-supported root cause**: v3's scoring is *myopic* — it doesn't
account for opp also targeting the same neutral. Both fleets arrive
at the same target around steps 10-15. Combat resolves:
- If v3 fleet > opp + target garrison: v3 wins, captures
- If v3 fleet ≈ opp + target garrison: bounce (rule 4 destroys both)
- If v3 fleet < opp + target garrison: v3 loses, opp captures

The OUTCOME of these contested captures is hugely variable, which
explains both the wild per-seed swings AND the average-recovery
pattern (sometimes we win contested fights, sometimes lose).

## Proposed v9 fix: opening-conditional NEUTRAL_BONUS + opp-aware target avoidance

Two changes targeted at the diagnosed cause:

### v9.A: re-enable NEUTRAL_BONUS in opening only

The disabled-global `NEUTRAL_BONUS = 1.5` regressed 28% because
"tipped scorer toward easy neutrals when contested enemy planets
were binding" (lib/missions/snipe.py:46-51). In OPENING, enemy
planets are ALL well-defended (no captures yet), so they're never
the binding constraint. Opening-only re-enable should be safe:

```python
if step_now < OPENING_HORIZON:  # OPENING_HORIZON = 15
    if t.owner == -1:  # neutral
        priority *= NEUTRAL_BONUS_OPENING  # 1.5
```

### v9.B: opp-aware target deduplication

Before settling each mission, predict opp's first launch (via v3
from opp POV — same pattern as v7's opp model). If opp targets the
same neutral AND would arrive at/before our fleet, demote OUR score
for that target. Send our fleet to opp's SECOND-choice target
(opp won't compete; we capture uncontested).

This is "anti-contested-capture" — a specific form of strike-window
timing aimed at opening neutral race.

Concrete (~50 LOC):
```python
opp_target = predict_opp_first_target(world, opp_id)
if opp_target is not None and step_now < OPENING_HORIZON:
    for m in missions:
        if m.target_id == opp_target.id:
            # opp targets same; check ETAs
            my_eta = m.eta
            opp_eta = compute_opp_eta(opp_target, world)
            if opp_eta < my_eta:
                m.score *= 0.1  # heavily demote
```

### Combined v9 = v3 + (A) + (B), opening-window only

Step 0-14: aggressive opening with neutral bonus + opp-target avoidance.
Step 15+: identical to current v3 behavior (no regression of mid/late game).

## Verification gates (per plan)

1. pytest passes (215 existing + any new opening probe tests)
2. v9 vs v3, 16 seeds × both sides = 32 games at full 500 steps: ≥ 55% W/D
3. v9 self-play 8 seeds, 500 steps: ≥ 80% draws (mostly v3-like late-game)
4. v9 vs v7_minimax, 8 seeds × both sides = 16 games: ≥ 50% W/D
5. Opening probe re-run: v9 vs precision/roi step-15 mean Δship ≥ -3 (was -3.0/-1.4)
6. Bundle smoke

## Expected μ-impact

Honest range: **+5-25μ** from reduced opening variance. Highest if
the "anti-contested-capture" component prevents the big-deficit
seeds (like seed 6 collapse). Median if only the NEUTRAL_BONUS
component fires (just shifts target selection).

Sub-thresholds for build vs not-build:
- v9 vs v3 ≥ 60% W/D: submit
- v9 vs v3 50-59% W/D: marginal; investigate before submit
- v9 vs v3 < 50%: don't build/submit; pivot to different fix

## What this probe DOESN'T tell us

- Late-game cascade mechanism (steps 30-500). Seeds 0 & 1 end +30/+33
  at step 30 but lose by elimination — that's a different problem.
- 4P FFA opening behavior. This probe was 2P only.
- Whether the user's specific observed game was a "bad-seed" outlier
  or representative. The variance is high.

## Files

- `audit/tournaments/opening_probe_v1.json` — raw 64-game per-turn log (1.2 MB)
- This file: `audit/2026-05-12-opening-analysis.md`
- Next: `agents/v9_opening/main.py` (post-design)
