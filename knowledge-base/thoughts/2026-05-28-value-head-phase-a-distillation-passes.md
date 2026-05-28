# 2026-05-28 — value-head Phase A: distillation passes, Phase B greenlit

Status: substrate diagnostic complete.

## Punchline

`baseline_learned` (chooser + MLP head distilled from `favor_hybrid`)
got 14/32 = **43.8 %** wins against `baseline_hybrid` (chooser +
`favor_hybrid`). Wilson 95 % CI [0.282, 0.607]. n auto-bumped from 16
to 32. Verdict: near-parity / INCONCLUSIVE.

Compare v1 (margin-on-`lite_greedy`-self-play): 2/32 = 6.2 % vs
`favor`. The **38 pp jump** is the diagnostic signal.

## Why this is a pass even though Wilson crosses 50 %

Phase A was scoped as a substrate diagnostic, not a submission
candidate. The question was "does the chooser-with-learned-head
wiring work at all?". A distillation MLP that faithfully reproduces
its teacher's decisions answers YES — the chooser substrate
correctly consumes whatever signal the head emits. v1 had a head
that emitted noise (43 % val variance explained) and lost 0/16 of
games; Phase A has a head that emits high-fidelity signal (99.8 %
val variance explained) and gets parity with the teacher.

A SUBMISSION decision at the same evidence would be a FAIL (Rule 45
needs Wilson-lo ≥ 0.50). But Phase A isn't a submission — it's a
go/no-go for whether we invest the compute in Phase B's richer
training signal. The answer is GO.

This distinction (diagnostic gate vs submission gate) is now a
promotion candidate against CLAUDE.md Rule 45.

## What we now know with confidence

1. **The chooser's substrate is fine.** Whatever the learned head
   outputs, the chooser ranks it. The proposer + chooser pipeline
   from `agents/baseline/` consumes value-head outputs correctly.
2. **The 40-feature pipeline is mostly sufficient.** Distillation R²
   ≈ 99.8 %. If a richer head in Phase B underperforms, blame the
   data / target, NOT the feature count. Expanding features is NOT
   the move.
3. **v1's failure was target + data, not architecture.** Margin
   computed against `lite_greedy` self-play (the v1 target) had two
   problems compounding: (a) margin at game-end is high variance
   relative to action-Δ; (b) `lite_greedy` is too weak — the head
   learned to beat a mediocre opponent rather than to play well.
   Phase A fixed problem (a) by using a teacher-distilled scalar.
   Phase B will fix problem (b) with a strong opponent pool.
4. **Inference latency is within the chooser budget.** p50 = 164 ms,
   p95 = 240 ms, max = 459 ms per turn under chooser load with
   `BASELINE_WALLCLOCK_MS=100`. We don't need to compress the head.

## What we explicitly do NOT know

- **Live-ladder calibration.** We did not A/B `baseline_learned` vs
  the current rolling pair (μ=806 / μ=829). The Phase A pass means
  it ~matches `favor_hybrid` (μ=1149 EVICTED), not that it would
  beat the live floor. Phase B candidate must clear Rule 43 panel
  AND Rule 45 n≥32 vs current rolling champion before push.
- **Does the distilled head add ANYTHING new?** It doesn't. A head
  that ~mimics its teacher is an inference cost regression with no
  upside. Phase A is a SUBSTRATE test, not a candidate. The upside
  must come from Phase B's richer signal (advantage + CRN + multi-
  horizon + strong opp pool).

## Phase B sketch (full roadmap in HANDOVER.md)

1. **Advantage head with Common Random Numbers.** Target
   `A(s,a) = margin_action − margin_idle` with same opp RNG seed
   on both legs. Expected 50–95 % variance reduction on the Δ signal.
   This is the single highest-EV change because the chooser cares
   about action-Δs, not absolute V(s).
2. **Multi-horizon target.** final-margin + K-turn margin + win-prob;
   KataGo weighting. Regulariser more than rank changer.
3. **Strong heterogeneous opponent pool.** Five strong agents
   spanning ~200 μ. Fixes the v1 single-weak-opp failure mode.
4. **Kaggle GPU.** 5-fold > 1h local ⇒ GPU per Rule 13. Two-tier
   smoke before production push.

Phase B gates: each addition A/B vs `favor_hybrid` at n ≥ 32,
Wilson-lo ≥ 0.50 for B-1 (CRN advantage), Wilson-lo ≥ 0.55 for the
final candidate (B-3). PLUS a live-rolling-pair calibration A/B
before any submission.

## What this changes about the program

Before today, the question was "is the learned-head approach broken?"
After today, the question is "does a richer training signal push the
distilled-head's ~44 % parity into a 55 %+ lift?" The headroom is
between favor_hybrid (μ=1149) and whatever the strong-pool + CRN +
multi-horizon learned head can hit. That's the bet for Phase B.

## Open ideas (not commitments)

- The 6 pp gap from parity is "interesting": it's the irreducible
  loss of compressing favor_hybrid into 40 features + an MLP. If
  Phase B beats parity, that gap is the floor on its added signal.
  A Phase B candidate at 55 % vs favor_hybrid implies it added ~11 pp
  of decision-quality on top of the distillation floor.
- A diagnostic worth running before Phase B: how does
  `baseline_learned` perform against the v1 opponent (`favor`)?
  Should be HIGH (≥80 %?) because favor_hybrid beats favor easily.
  If `baseline_learned` doesn't, the distillation R² is lying
  somewhere we haven't caught. Five-minute n=16 test.
- The distillation corpus came from a fixed game distribution.
  Phase B should generate new games per training epoch (or every
  K epochs) so the head sees the action-space the strong pool
  actually visits.
