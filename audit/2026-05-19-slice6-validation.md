# Slice 6 validation — 2026-05-19 — NEGATIVE RESULT

> Commit `49f572f` — strategic LP per-turn assignment as a commit
> reason. Per Rule 41: inspect first, small A/B. Both indicators
> evaluated.

## Single-game introspect (seed 42, vs trajectory baseline)

**Outcome: WIN +1/-1** (same as Slice 4)

| Metric | Slice 4 | Slice 5 | Slice 6 |
|---|---|---|---|
| Inner emits/turn | 0.91 | 0.68 | 0.47 |
| Backstop/turn | 0.32 | 0.18 | **0.88** |
| Backstop rate | 20.1% | 18.1% | **40.6%** |
| Total moves/turn | 1.23 | 0.86 | 1.36 |

LP fires aggressively — 5× more backstop emits than Slice 5,
65% of total emits from L0.

## Small A/B (n=16, vs trajectory baseline)

| | Slice 4 | Slice 5 | Slice 6 |
|---|---|---|---|
| Wins | 9/16 (56.2%) | 9/16 (56.2%) | **8/16 (50.0%)** |
| Wlo | 0.332 | 0.332 | **0.280** |
| max-ms | 1535 | 861 | **1832** |

**Slice 6 regresses on both axes**:
- One fewer win (8/16 vs 9/16) on the same 16 seeds.
- Wallclock max blew up to 1832ms (over the 1000ms env cap).

## Diagnosis

The LP backstops ~5× more launches than Slice 5 (0.88/turn vs
0.18/turn). Most of those are emits the inner explicitly didn't
pick. The LP's static assignment doesn't see:

- **Production accumulation over multiple turns**: LP only looks at
  current ship counts and capture-time matrix. A source the inner
  is "saving" for a 3-turn-out capture gets LP-committed to a
  immediate-but-suboptimal launch.
- **Defensive needs**: LP doesn't model reinforce vs attack
  tradeoffs.
- **Inner chooser's strategic reasoning**: when the inner skips a
  source, it's usually for a reason (the rollout sees a better
  future use). LP overrides that judgment.

The extra emits behave as noise — they're launches the inner
explicitly rejected, now appended via backstop. The win-rate drop
and wallclock blowup are both consistent with this interpretation.

## Decision

**STOP. LP wiring disabled by default** (gated behind
`BASELINE_LP_COMMIT=1`). The `strategic_lp.py` module stays in
the codebase for future research uses (audit-replay analysis,
training-data labelling) but doesn't fire in production.

Do **not** proceed to Slice 7. Two reasons:
1. Per plan §12 stop-rule: Wlo < 0.45 → STOP this slice line.
2. The Slice 6 result casts doubt on the "stack more commits"
   thesis generally. Slice 7 (W3 fork detector) would add yet
   another commit reason; if Slice 6's LP commits are noise, why
   would W3 forks not be?

The best-confirmed configuration is **Slice 5** (Slice 4 backstop
+ Slice 5 dominance, no LP). It at least matches the trajectory
baseline (Wlo=0.332 at n=16, identical wins to Slice 4) with
clean wallclock (max=861ms).

## Recommendations (PI to choose)

Three honest paths forward:

1. **Ship Slice 5 as opt-in** — validates the layered architecture
   without claiming a win-rate improvement. Production default stays
   on trajectory; layered available via `BASELINE_CHOOSER=layered`
   for further research.
2. **Park the whole layered axis as research-only**. The 4-slice
   push (4+5+6+7) was an ambitious experiment; Slices 4-5 stay
   neutral, Slice 6 regressed, Slice 7 carries the same architectural
   risk. Document negative-result lessons and pivot to a different
   axis (e.g., trajectory leaf-eval cache, faster value head).
3. **Try Slice 7 anyway** — the W3 fork detector is on a different
   axis (tactical pattern, not strategic optimization). If forks are
   real and rare, the per-commit harm might be small. But the
   "stack more commits" thesis is weakening; expect at most a small
   lift.

Production unchanged: `BASELINE_CHOOSER=trajectory` remains default.
Rolling-pair floor μ=1118.8 preserved.
