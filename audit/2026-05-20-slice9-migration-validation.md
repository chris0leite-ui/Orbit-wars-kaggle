# Slice 9 validation — 2026-05-20 — migration partial recovery, still below

> Commit `175f790` — `agents/baseline/migration_solver.py` +
> chooser_differential integration. Verified per Rule 41.

## Single-game introspect (seed 0 vs trajectory)

**Outcome: WIN +1/-1** (was TIE (1,1) under Slice 8c).

Trace can't show migration scores directly (introspect's scoring
helper uses `score_candidate_differential` which returns 0 for
own→own — that's the whole reason the solver was needed). The
outcome flip is the meaningful signal at single-seed scale.

## Small A/B (n=16, vs trajectory baseline)

```
n=16  wins=4/16  (25.0%)  Wlo=0.102  Whi=0.495  FAIL  (Whi<0.55)
focal turn-ms  p50=55  p95=302  max=734
total elapsed 338.7s
```

| | Slice 8 | Slice 8c | **Slice 9** |
|---|---|---|---|
| Wins | 6/16 (37.5%) | 3/16 (18.8%) | 4/16 (25.0%) |
| Wlo | 0.185 | 0.066 | 0.102 |
| max-ms | 810 | 704 | 734 |

Migration partially recovered from the Slice 8c regression
(+1 win vs Slice 8c), but didn't restore parity with Slice 8
(which itself was below the 0.30 keep threshold).

Per plan §15: Wlo=0.102 < 0.30 → **STOP**.

## Diagnosis

The migration solver IS doing analytical work — the unit tests
(14/14) pin its math, and the single-seed WIN suggests it's
making some difference. But the differential chooser as a whole
remains below baseline.

Two possibilities:

1. **Migrations are firing but their value isn't realised**:
   we send ships to a "high-potential" planet, but by the time
   they arrive the situation has changed (opp captured nearby
   targets, our destination's target list shrank, etc). The
   closed-form EV-projection is static; the game isn't.

2. **The differential chooser lacks defensive reasoning**: when
   the trajectory chooser's rollout simulates `lite_greedy_policy`,
   it produces some defensive opportunities indirectly (opp's
   simulated launches trigger our W2 reinforces). Differential's
   projection assumes opp does nothing → no defensive cues.
   Migration doesn't address this.

The "missing piece" the user identified (repositioning) was real,
but it isn't the LAST missing piece. The candidate space has more
gaps — particularly around opp counter-modelling.

## Decision

**STOP this slice.** Per the preservation strategy:

- `migration_solver.py` stays in tree, opt-out via
  `BASELINE_MIGRATION=0` (default on under differential chooser).
- `chooser_differential.py` stays gated on `BASELINE_CHOOSER=differential`.
- All Slice 8/8c/9 infrastructure preserved as research-only.

Production unchanged: `BASELINE_CHOOSER=trajectory` remains default.
Rolling-pair floor μ=1118.8 preserved.

## What this tells us about the full-analytic vision

The user's full-analytic roadmap is correct in principle but the
PATH is harder than "add one more candidate class." Even with
two analytical pieces in place (closed-form leaf eval + migration
solver), the differential chooser loses to trajectory's noisy
rollout.

The likely missing pieces, in order of expected impact:

1. **Opp counter-projection**: Slice 9's value formula assumes
   opp does nothing. A Stackelberg-one-step injection (closed-
   form opp counter-launch in the projection) would give the
   chooser realistic counter-aware scores.
2. **Multi-turn capture chains**: a single migration that
   enables N future captures should be valued at N × that
   capture's EV, not just one. The current formula computes
   only "next-capture-EV unlocked."
3. **Defensive priors**: with no opp policy, the chooser doesn't
   anticipate threats. W2 fires only on already-inbound threats.
   A speculative "opp will probably attack X next" inference
   would unlock pre-emptive reinforcement.

Each of these is its own analytical primitive. The vision is
sound; the road is longer than 4 slices.

## Recommendations

**Don't iterate further on differential as a chooser.** Two
intentional attempts (8c filter, 9 migration) confirm the
underlying chooser has multiple missing pieces. The substrate
(closed-form leaf eval + capture-EV solver + migration solver)
stays valuable for future use; the chooser as a standalone
production agent doesn't work.

**Next session strategic question for PI**:
- Continue analytical-substrate work on a DIFFERENT axis
  (opp-counter-projection, multi-turn chains)? OR
- Pivot back to the trajectory chooser and explore non-
  analytical improvements (better value head, better opp
  policy, better leaf-eval cache)? OR
- Accept current trajectory baseline (μ=1118.8) as the live
  agent and focus on submission selection / ladder strategy?

Production unchanged. No submission. All slices preserved as
opt-in research code.
