# 2026-06-03 — region/chunk-aware MVP (parity) + the idle-source finding

Session built a region/chunk-aware agent MVP on `claude/region-mvp`
(off champion HEAD). Result: **parity with the champion**, durable
framework banked, two findings worth more than the agent.

## What shipped (default OFF, byte-identical floor)

`lib/region.py` + `BASELINE_REGION`: cluster planets by orbital parameters
(radius band × angular sector — stable under shared omega, deterministic),
score each region by the **production-time integral** (the one piece of the
falsified reach-frontier doctrine that survived as sound), compute a
per-region contest tick + predictability, then (a) **bias** the existing
launch candidates toward high-value predictable contested regions and
(b) **advance** idle rear mass toward the frontier region (generalizes
`drain_stagnant_rear`). GAIN(region) is a gated empty stub.

Separate `BASELINE_HORIZON_DECAY`: deepen the K-step rollout floor early,
scale to champion MIN_HORIZON over ~250 turns. Threaded as a `propose()`
param (bundle-safe — no module-global mutation, which would diverge
source-vs-bundle).

## The result, honestly

- region-only: 15/32 = 46.9% [0.31, 0.64] vs champion — parity.
- region+horizon: 7/16 = 43.8% [0.23, 0.67] — parity.
- Timing clean (region max 929ms < champion's own 1084ms).
- Off-is-identical proven twice (216-call replay, 80-state proposer parity).

**Neither lever beats the champion at default tuning.** The region layer
*does* change behavior (47/186 turns diverge, growing with game phase) but
nets neutral.

## Why it's neutral (the structural diagnosis)

The chooser selects by its **rollout score**, not by the candidate's
cheap-delta. So biasing cheap-delta only changes *which* candidates get
rolled out under the validation cap — it can never override the rollout's
verdict. The bias is gentle by construction. Most of the behavioral change
came from the **advance pass**, and redeploying idle ships toward the
frontier landed net-neutral here.

**The untried lever that follows from this:** put region value into the
chooser's *final score* as an additive term (prefer a high-value-region
capture at equal rollout delta), not candidate reordering. That's the
"feed the rollout" idea done at the scoring layer instead of the
enumeration layer. It's the natural next step and was out of MVP scope.

## The finding that may outlast the agent

`scripts/probe_idle_sources.py` (1922 champion-turn rows, vs v7_minimax +
mirror): the champion fires from **~1 of ~13 eligible planets per turn —
~90% sit idle**, even in close, contested mid-game. This **refutes the
empirical premise** ("source-saturated ⇒ no idle planets for teamwork")
that closed the joint-coordination axis on 2026-06-02 — that null was only
ever measured in blowout wins. The idle capacity is real; what's unclear is
whether deploying it is correct (hoarding wins) or wasted. The region
advance pass was the first attempt to deploy it; net-neutral so far.

## Open questions

- Does region-value-as-a-score-term convert the parity to lift?
- Is the ~90% idle correct (hoarding) or a real conversion gap? The
  advance pass being net-neutral leans toward "mostly correct," but only
  one redeploy heuristic was tried.
- Horizon decay alone was never A/B'd in isolation (only stacked) — likely
  also parity given the combined run, but unconfirmed.
