# 2026-05-10 — PI direction: physics correctness first, then look-ahead

> Per Rule 35: PI voice-dump, append-only. Captured during session
> `improve-strategy-ab-testing-jYA2R` after the lead_aim ETA-offset
> commit (`cbf142b`) landed and the broader deterministic-correctness
> A/B picture was discussed.

## Assumption baseline (PI ratified)

- **Take Kaggle's server simulation as correct and stable.** No version
  drift hedging. Our local `kaggle_environments==1.29.1` IS what Kaggle
  runs. If Kaggle changes versions we treat it as a separate event.
- We don't need to re-verify env semantics every session. The physics
  reading we did today (fleet spawn offset, swept-pair collision, sun
  geometry) is settled.

## Strategy from here (north star)

Two halves of the same plan.

### Half 1 — accuracy: don't lose ships to wrong physics

Our agents must execute the actions they intend. Ships sent must reach
their target (or fail for *strategic* reasons, never *modelling* reasons).
We measure this directly: capture-success rate per launched fleet.

Open items from today's punch list:

1. **Sun-avoid using lead-predicted arrival point** (punch #7). Current
   `sun_avoid` checks `path_clears_sun(src.center, target.xy_current)`.
   For orbiting targets the fleet flies to predicted arrival, not
   current xy. Same flaw in any strategy-side sun pivot. Fix: reuse
   the same `predict_relative(...)` call as `lead_aim` to compute the
   arrival point, then check `path_clears_sun(src.center, arrival_xy,
   safety=1.0)`. Once this is right, sun_avoid earns its DEFAULT slot.
2. **Re-test 3-iter lead_aim combined with the ETA fix** (punch #8).
   3-iter alone regressed; with the ETA correction the fixed point is
   different, so the earlier null isn't load-bearing.
3. **Capture-success probe** — instrument a roi run; count fleets that
   reach their declared target vs miss / die in sun. This is the
   diagnostic that tells us whether punch #7-#8 actually matter at
   the μ scale, or whether we should jump straight to Half 2.

### Half 2 — look-ahead and global decisions

The current strategies are **myopic per-source greedy**. Each owned
planet picks its own best target this turn. That's the floor; the
ceiling is much higher.

#### A. Fleets-in-flight awareness

The obs contains `fleets` (every ship in transit, ours and theirs).
None of our current strategies read this field. Implications:

- **Don't double-commit**: if our own fleet is already en route to a
  target with enough ships to capture it, the same source shouldn't
  send a second wave at the same target (over-commit dilution).
- **Anticipate enemy arrival**: predict where each enemy fleet will
  land and when. Defend or counter-attack accordingly.
- **Re-target on opponent's plan**: if the opponent sent a fleet to a
  high-yield neutral, racing them with our own fleet is a calculable
  decision (their ETA + ships vs our ETA + ships).

The obs already has the data. Just need to read it.

#### B. Joint global decisions (not planet-by-planet)

Current per-source loop is independent. A global solver:

- **Bipartite assignment**: think of `(sources × targets)` as a
  maximum-weight matching subject to garrison constraints. Hungarian
  algorithm or LP relaxation, both O(n³)-ish — fine at 24 planets.
  Per-source greedy is the lazy version; joint optimum is achievable.
- **Gang-up coordination**: H4 from `heuristics-research.md` (multi-
  source simultaneous-arrival timer) lives here. Two sources can
  combine ships on one valuable target by timing arrivals.
- **Defense / offense balance**: portfolio-level decision. Reserve N%
  of total garrison for defense against predicted enemy arrivals;
  deploy the rest offensively.

#### C. Look-ahead search (heavier compute)

Once Half 1 + 2A + 2B are in, action choice can become a search problem:

- **Beam search** over (target-set, allocation) for the next turn,
  scoring with a *simulated rollout* a few turns ahead under a
  fixed-policy opponent model.
- **Mini-MCTS** on a coarsened action space (top-K targets only —
  see §D below) with a 100-200 ms budget per turn.
- **One-turn lookahead with opponent model**: assume opponent uses a
  similar ROI strategy, predict their move, factor into our score.

We do NOT need a generic game-tree search to start. A beam search
with handful-of-targets is enough to break the myopic ceiling.

#### D. Simplification by ROI threshold

The action space explodes if we consider every (source, target) pair
and every fleet size. PI directive: **prune.**

- **Top-K filter** on the target list (e.g. keep only the K highest-
  ROI planets globally; ignore the rest). Reduces branching.
- **Absolute ROI threshold** (`score < τ`): some planets are always
  bad — outer-ring tiny-production, post-mid-game low-yield.
- **Owner-aware pruning**: late-game, neutral planets with low
  production are often dominated by enemy planets (denial bonus
  flips the math). Different threshold by owner class.

The pruning lets every later piece — assignment solver, search,
opponent modelling — operate on a tractable subset.

## What this means for next sessions

We're entering a **research-and-build** phase, not a submission rush.
Local A/B is the work; submission cadence stays cautious. Order:

1. Physics correctness: land punch #7 (sun-avoid arrival-aware), test
   #8 (3-iter), build the capture-success probe.
2. Fleets-in-flight awareness: read `obs.fleets`, build per-target
   "predicted-arrival accumulator" used by `arrival_size` and by
   target scoring.
3. ROI threshold pruning: cheap, isolates the action space.
4. Joint global decisions: bipartite assignment first (simpler), then
   gang-up timing.
5. Look-ahead search: only after the above are stable.

PI ratified deferrals:
- Replay-mining for "things I can't see directly" — deferred until
  obvious fixes land. We scan and plan first.
- ROI scoring variants (V1-V5 from earlier plan): folded into Half 2.
  `roi_margin`, `roi_horizon` are the right shape but must run on
  top of the corrected physics + the joint solver, not as standalone
  per-source variants.
