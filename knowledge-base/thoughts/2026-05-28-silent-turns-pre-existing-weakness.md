# 2026-05-28 PM — silent-turns mid-game is a pre-existing peak weakness

## What we found

While investigating "why does PV_ETA lose seed=2 vs every panel opponent",
the smoking gun turned out to NOT be PV_ETA at all. Both peak (PV_ETA=0)
and PV_ETA=1 lose the same seed=2 game vs v4_planner. The shared
mechanism is that our chooser **emits zero launches for 13-29 consecutive
mid-game turns** while the opponent keeps the board pressured every turn.

## The numbers

| Run | Outcome | Steps | P0 emit % | Mid-game max silent streak | Late-tail streak (post-decision) |
|---|---|---:|---:|---:|---:|
| PV_ETA=0 (peak) seed=2 vs v4_planner | LOSS | 159 | 35% | 14 | **29** (post-elimination) |
| PV_ETA=1 seed=2 vs v4_planner | LOSS | 126 | 38% | **13** | 11 |
| PV_ETA=1 seed=1 vs peak anchor (control) | **WIN** | 220 | 44% | ~8 | 21 (post-decision tail) |

The 29-turn streak at peak is post-elimination noise (P0 had zero planets,
literally couldn't emit). The diagnostic signal is **mid-game streak**:
losses run 13-14, the winning game runs ≤ 8.

## Why this matters

The 80% panel result for PV_ETA submission (sub 53111837) is honest.
The 4 losses across opponents all fell on seed=2, where peak also loses.
PV_ETA neither caused this weakness nor cured it — it modestly reduced
the worst mid-game silent streak (14 → 13) but the structural failure
mode is the same.

This redirects the next-session investigation from "tune PV_ETA" to
"why does the chooser stall in contested mid-game?"

## Suspected mechanism (to verify)

The chooser's emit gate is `Δ > MIN_DELTA=0` AND admissibility filter.
On the silent turns, every candidate must be producing Δ ≤ 0. Three
forces likely conspire:

1. **Rollout pessimism.** `lite_greedy_policy` is the opp-model inside
   the per-candidate K-step rollout. If it's too aggressive at near-
   neutral counter-capture, the simulated P1 grabs every contested
   target before we arrive — leaf shows "no capture happened" → Δ ≈ 0
   or negative. The chooser concludes "nothing to do."

2. **Admissibility filters out the long-shot escape.** Cross-board grabs
   that BYPASS the contested middle (and thus don't trigger the rollout
   pessimism) get filtered by sun/OOB/path-blocked checks. The only
   candidates that COULD score positively are removed before scoring.

3. **Wallclock compounds it.** With ~4 sources × proposer fanout ×
   wait-grid + joint enumeration, the chooser may not score every
   candidate in 600ms. Biased toward cheap-to-score, which are also
   the contested ones.

(1) is the most parametrically testable. If swapping `lite_greedy_policy`
for a mixture of {greedy, do-nothing, defender} causes the chooser to
emit during the silent stretches, rollout pessimism is the dominant
factor.

## Next-session probes (queued, ranked by EV)

1. **Instrument `score_candidate_v4` on a silent turn.** Run seed=2
   game vs v4_planner, intercept at the silent-streak window (t=22 or
   so), dump every (src, tgt, ships) candidate's raw `leaf - baseline`
   pre-and-post-PV_ETA. We need to see: are all candidates Δ ≤ 0, or
   are some positive-but-tiny that get killed by MIN_DELTA=0 ?

2. **Opp-model mixture in the rollout.** Replace single greedy policy
   with sample-from-mixture {greedy, sniper, do-nothing, defender} at
   each rollout step. Hypothesis: a less-confident opp-model would
   let our captures look positive-EV in the rollout's leaf state,
   restoring emit frequency.

3. **Single-knob ablation: `BASELINE_MIN_DELTA=-5`.** If the gate is
   filtering Δ-just-below-zero candidates, lowering the floor restores
   emits. A quick-cheap shot that's orthogonal to PV_ETA. If it works
   on seed=2 AND holds panel rate elsewhere, we have a free lift.

4. **Cross-process determinism audit.** Same-seed flip on seeds 0/2
   between single-opp and panel ProcessPool invocations is a separate
   recurring source of A/B noise. Triage RNG hot paths.

## Where this lives

- Sub 53111837 (PV_ETA=1, μ pending) is the current rolling-pair newer
  half. Settlement comes ~30 min after submit.
- The silent-turns investigation does NOT require re-iterating on
  PV_ETA. PV_ETA is settled (either positive or neutral on the panel).
- The next mechanism axis is "fix the rollout's opp-model" — which is
  a fundamentally different lever from any chooser/proposer surface
  we've previously touched.

## Connection to existing knowledge

- **Mechanism-ledger:** v22 (rollout-counter-recapture) tried to make the
  opp-model stronger in 2026-05-17, got 8/32 = 25% Wlo=0.13 FAIL. The
  failure mode there was "stronger opp → captures look fragile → chooser
  passive." The diagnostic above (silent turns) is the SAME shape.
  Both findings point at the rollout-opp-model as the load-bearing
  knob — but past attempts moved it the WRONG direction (stronger,
  not more representative). The mixture-sampling probe above is the
  unexplored direction.

- **Knowledge-base/concepts/lookahead-simulator-architecture.md:**
  documents the fast_sim substrate. The opp-model swap point is in
  `opp_actions_for_snap` (called per-rollout-tick). Single-point fix
  surface; minimal blast radius.

- **PEAK_BASELINE.md fragility risk #2** (`favor` leaf double-counts
  production via `pv_horizon`): orthogonal but adjacent. Both findings
  point at "the chooser's value calculation systematically misvalues
  in specific game phases." Same kind of bug, different dimension.
