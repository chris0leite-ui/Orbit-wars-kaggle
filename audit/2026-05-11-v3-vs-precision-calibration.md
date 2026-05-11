# 2026-05-11 — v3 vs precision_v3 cross-class calibration

## Setup

Calibration test: v3_snipe (current HEAD, with σ-equivariance patches +
merged main's v3.4 WorldModel arrival_size + LEADER_MULTIPLIER) vs
precision_v3 (from `origin/merge-precision-to-main`, validated as
live submission #52552139). Both sides, 8 seeds × 500 steps = 16 games.

Cross-class test — precision is a fundamentally different strategy
class: deterministic intercept solver + 2-source wave bundling +
strike-window timing + depth-2 enemy minimax + robust min-max scoring.

## Result

```
v3 vs precision: 8W / 0D / 8L = 50.0% W/D over 16 games
  P0: 4W / 0D / 4L
  P1: 4W / 0D / 4L
  Mean game length: 192 steps (all eliminations, none reached 500)
```

**Exact 50/50 split.** Perfectly seat-symmetric — every seed's outcome
is identical regardless of which agent plays P0 vs P1. This confirms
both v3 (post-σ-fix) and precision are seat-symmetric (group-equivariant
under the seat-swap).

Per-seed breakdown:

| Seed | Outcome (both sides) | Avg steps |
|---|---|---:|
| 0 | v3 wins | 197 |
| 1 | v3 wins | 178 |
| 2 | precision wins | 161 |
| 3 | v3 wins | 236 |
| 4 | v3 wins | 169 |
| 5 | precision wins | 215 |
| 6 | precision wins | 156 |
| 7 | precision wins | 197 |

v3 wins seeds {0, 1, 3, 4}; precision wins seeds {2, 5, 6, 7}.

## Interpretation

**The cannot-lose lock doesn't generalize across strategy classes.**
v3-vs-v3 self-play = 100% draws (provable by σ-equivariance). v3 vs
precision = 50% wins (different policies; competing legitimately).
This is the expected behavior — the symmetric-game value theorem
guarantees value 0 against ANY opponent only at Nash equilibrium,
which v3 is NOT (v3 is just an approximate Nash in its strategy
class).

**The 50/50 result is empirically the cannot-lose floor.** v3
matches precision on aggregate. Our v3 is in the same strength
tier as precision (which is a strong agent — #52552139 reached
ladder).

**Strict per-seed determinism** means there's a clear axis to
investigate: WHY does v3 win seeds {0,1,3,4} but lose {2,5,6,7}?
The board configurations must favor one strategy over the other.
Diffing the seeds would tell us what advantages each class has —
likely a directly-learnable pattern.

## Learnable techniques from precision

`agents/precision/README.md` documents 7 techniques v3 doesn't have:

1. **Precision intercept**: 100% land-rate inverse solver. v3 uses
   forward lead_aim (5-iter fixed-point + search_safe_intercept).
   Difference: precision verifies every shot lands; v3 has ~3% sun
   loss + 0.3% OOB.
2. **Strike-window timing**: shot arrives 1-3 ticks AFTER projected
   enemy capture. 2× ROI claim (minimum-defender window).
3. **Wave bundling**: 2-source synchronized arrival on hard targets.
4. **Depth-2 enemy minimax**: 2-turn lookahead under both enemy
   hypotheses.
5. **Post-commitment enemy re-projection**: account for enemy
   response to our weakened state after a wave.
6. **Fast event-driven sim**: 24× speedup over step-loop.
7. **Robust min-max score**: 0.7·greedy_theirs + 0.3·worst_for_us.

The 50/50 result suggests v3 doesn't strictly need any of these
to be competitive at precision's level. But each has an empirically
documented win-rate impact.

## Next iteration candidates (ranked by leverage)

1. **Strike-window timing on v3** (high impact, ~1-2 weeks).
   Requires: enemy launch projection + new mission class
   (`strike_window`) + damped scoring (α=0.7) + integration with
   settle_plan. v3 already has `WorldModel.owner_at` timeline; the
   strike-window logic plugs in cleanly.

2. **Enemy launch projection** as a standalone (foundational; ~1 week).
   Run v3's mission proposers with `my_id=opp_id` to predict opp's
   next launch(es). Useful both for strike-window AND for richer
   reinforce scoring (know which planets opp targets).

3. **Per-seed difference analysis** (~1-2 days).
   Diff seed 0 (v3 win) vs seed 2 (precision win) — what board
   feature predicts which class wins? May reveal a CHEAP
   distinguishing improvement (e.g., a specific mission class v3
   misses on some configurations).

4. **Investigate v2 anomaly** (~1 day).
   v3 vs v2 = 53.1% (audit/2026-05-11-calibration-post-merge.md);
   confirm at 64 seeds, isolate the cause.

## Recommendation

**Build strike-window mission class as the next iteration** (option 1).
Highest claimed ROI (2× on individual shots), buildable in our budget
(43 days), doesn't require precision's full inverse solver. Integrates
with our existing σ-equivariance work — strike-window missions go
through settle_plan, which we already made group-equivariant.

If after strike-window v3 still ties precision at 50%, move to wave
bundling. If strike-window pushes us to 60-70% against precision,
we've climbed above precision's class.

## Files

- `audit/tournaments/v3-vs-precision.json` — raw tournament data
- `agents/precision/` — gitignored (separate-branch agent); kept on
  disk for calibration purposes only
