# V2 is a tier above us; the recapture/holdability leaf-patch direction is exhausted

**Date:** 2026-06-17 (long session). **Trigger:** PI pushed to crack the
recapture/holdability gap that loses us games; "build it, tomorrow we rethink."

## The headline reckoning
- We beat the **producer-class panel**: 2P **4/4 vs Roman-1224** (the panel's
  highest labeled LB), ~50% 4P vs {V2, Roman, konbu} shuffled.
- We **lose to Producer V2**: 2P **1/5**, and V2 is **verified not-hobbled** (it
  beats weak v7_0 and launches 90–340×/game via `env.run`). So the losses are real.
- **V2 is simply a tier up.** Consistent with our mid-pack ladder μ (~1116); the
  unpublished top sits above us. "Do we beat the producer?" — the *panel* yes, **V2 no**.

## Why we lose to V2 (traced, seed 7)
Even — even slightly ahead — through ~step 50 (10 vs 8 planets). Then **V2
out-economies us mid-game** (step 80: 11 planets / 734 ships vs our 8 / 626) and
snowballs (step 100: 683 vs 461 ships). Mechanism = the **hold/retake battle**:
- **59% of our captures get retaken**; V2's stick.
- V2's fleets are bigger (median **36 vs our 24/~20**) — but that is **downstream
  of V2's richer economy**, not a knob: we send ~20 because we're **ship-constrained**
  (we send what we can afford); V2 sends 36 because it's richer because it holds.
- So: V2 holds → out-produces → bigger fleets → snowball. The fleet-size gap is a
  *symptom*, not a cause.

## Seven leaf/sizing/penalty levers — ALL null or worse
take-and-hold (2P win / 4P disaster on the ladder) · threat-size enemy · threat-size
all (worse vs strong field) · recap-opp (washed out n=20) · arrival-cap (null n=32,
far-metric unmoved) · neutral-hold (no fleet-size change, wins 3/6→1/6) ·
**holdability penalty** (tied, penalty barely fires). All gated **default-OFF**.

## The fundamental reason (why search/sim/analytic ALL fail on holdability)
Holdability = "will the opponent *choose* to retake this?" — an **opponent-future-
choice** quantity, not a board fact.
- **Simulation** is only as good as its opponent model; ours (base producer) is
  *weaker* than V2 (tuned), so it mis-predicts retakes, and **more depth amplifies
  the model error** (you search a wrong tree harder). The rollout went *more*
  defeatist with a stronger policy.
- **Analytic bound** (visible reachable force) **underestimates V2's reactive
  retakes** (launched after our capture, from anywhere) → the penalty almost never
  fires (captures 153→165, not down).
- We're also **ship-constrained**, so the sizing levers can't even move our fleet
  size (median stuck ~20 regardless of margin/threat/neutral-hold), and forcing it
  *starves* us (3/6→1/6).
You can't out-search a wrong opponent model, and you can't bound a reactive choice.

## What IS solid (kept)
- **Code-review fixes** (net-neutral on wins, but genuine): gang-up sized to the
  LATEST arrival (was under-strength); defense reinforces only vs CLOSING fleets;
  `_num_seats` slot-count; **whole-turn time bound (4P max 1219→451 ms)**;
  `_twoply`/rollout budget + exception guards.
- **Calibration submit LIVE** (sha `6d1d0365`): the 2P/4P gate — 2P hold-margin 0.5
  + defend (70%-class take-and-hold), 4P breadth-first min-force (lr-fixed-class
  ~31%) — + all fixes; every experimental lever default-OFF. Evicts the stuck
  PENDING 53768768; keeps the 1078 backstop.

## Tomorrow's rethink — directions (NOT another leaf-patch)
1. **Accept mid-pack & calibrate.** The gate agent is our best-of-both; use the
   remaining ~5 days of submits to ladder-calibrate it (does the gate beat the
   1078 ungated take-and-hold? — that IS a real, ladder-measurable question).
2. **Out-economy, not out-capture.** The loss is the snowball; the lever is keeping
   *tempo/economy* parity to step ~60, not per-capture holds. Unclear mechanism —
   needs its own replay study.
3. **A genuinely stronger search** (deeper/wider, or wrap the producer's *own*
   policy rather than the leaf + 2-ply). Big build, high-risk with the deadline.
4. **Re-examine the real target.** V2 may not even be on the live ladder; mine the
   actual ladder top's behavior (fingerprints) before building against V2.

## Repro
External strong agents in `audit/external/agents/` (gitignored; re-pull). V2 =
`slawekbiel_the-producer-v2`. Harnesses this session in `/tmp` (margin/threat/
neutral/holdability sweeps vs V2; paired fixed-vs-prefix; cap A/B). Load the
pre-fix agent from INSIDE `agents/least_resistance/` (not /tmp) or it loses orbit.
