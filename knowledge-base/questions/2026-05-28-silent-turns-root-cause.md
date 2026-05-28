# 2026-05-28 PM — silent-turns root-cause question

## The question

Our chooser emits ZERO launches for 13-29 consecutive mid-game turns
on contested-expansion seeds (verified seed=2 vs v4_planner; suspected
across the seed-2 archetype). The opponent emits every turn or near.
This happens at PV_ETA=0 (peak) AND PV_ETA=1. Why does the chooser
stall?

## What we know

- Per-turn the chooser receives a candidate list from `propose_*`,
  scores via `score_candidate_v4`, emits everything with `Δ > MIN_DELTA=0`.
- On silent turns, the chooser ran (no crash, no wallclock-exhaust
  evidence) but emitted `[]`.
- Therefore every candidate produced Δ ≤ 0.

## Three hypotheses (ranked by suspicion)

1. **Rollout opp-model pessimism (~70% confidence).** The
   `lite_greedy_policy` simulated inside per-candidate K-step rollouts
   counter-captures near-neutrals too aggressively, so the leaf state
   shows "P1 grabs target first" for every candidate → Δ ≈ 0 or
   negative. Test by replacing single greedy with mixture
   {greedy, do-nothing, sniper, defender}.

2. **Admissibility filters out the escape valve (~20% confidence).**
   The only candidates that COULD score positively (cross-board grabs
   that bypass the contested middle) get filtered by sun/OOB/path-
   blocked checks before scoring. Test by running with
   `TRAJECTORY_SKIP_ADMISSIBILITY=1` on a silent turn.

3. **MIN_DELTA=0 is too strict (~10% confidence).** Candidates are
   producing tiny positive Δs that exactly equal 0 from float-precision
   rounding. Test by ablating `BASELINE_MIN_DELTA=-5.0` and seeing if
   P0 emits.

## How to answer

See `knowledge-base/thoughts/2026-05-28-silent-turns-pre-existing-weakness.md`
for the probe plan. First step: instrument `score_candidate_v4` on
seed=2 t=22 (a known silent turn) and dump every candidate's
pre-and-post-discount Δ. The raw distribution tells us which of the
three hypotheses fits.

## Why it matters

If we can fix this for a single seed-class (geometries that force
contested mid-game expansion), it likely lifts the 80% panel rate
closer to the 95%+ that the underlying game-strategic merit deserves.
The seed-2 loss is a peak-class deficit not introduced by any recent
change.

## Status

Open. Next session priority 1 (PI directive 2026-05-28 PM:
"begin in the first session with further investigation and hard
thinking").
