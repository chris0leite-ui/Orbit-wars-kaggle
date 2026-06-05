# Cycle stalemate + horizon scaling -- the unsolved structural defect

**Date:** 2026-06-05.
**Context:** observed in validation game 78807326 (sub 53384340 self-match),
diagnosed at session end; PI declined band-aids.

## The cycle

In a 4-way self-match the 4 agents converged into a perfect 50-turn
ownership exchange. Each agent owned exactly 2 planets at every sampled
tick (their starting planet + one of the high-prod planets P20-P23).
Ship counts on the high-prod planets cycled 3 -> 6 -> 9 -> launch -> 3.
Launches fired in **pairs every ~11 turns**: launch-the-attack one tick,
launch-the-tiny-follow-up the next, repeating across the entire 500-step
game. 4-way tie at the step cap.

## Why the agent keeps firing

The scorer (`competitive_score` over `sparse_launch_flow_delta` at
`config.horizon = 18`) sees:

- Tick 0-4: I send my fleet
- Tick 5-9: I own the contested planet (gain a few ticks of production)
- Tick 10-12: opp recaptures (loss starts)
- Tick 13-18: opp owns it (loss continues but TRUNCATED at H=18)

Net at H=18: roughly even or slightly positive over the truncated horizon,
hence above the renormalised firing threshold. The agent fires.

The PI's correct play: stockpile. The high-prod planet (prod=3) accumulates
3 ships per turn. After 30 turns of NOT firing, the agent has +90 ships,
enough to overpower the small planet's reactive defence and capture
permanently. But this play has horizon ~30+ ticks; the planner cannot see it.

## What the H bump experiment revealed

Empirically (n=16): H=36 regressed from ~11/16 to **2/16 (Wilson [3.5 %,
36 %])**. Mechanism: opp_proj only projects ONE TURN of opp launches
(etas land in [1, 8]). With H=36, the scorer simulates opp doing nothing
for ticks 9-36 -- 28 of 36 simulated ticks under the static-opp assumption
we were supposed to be fixing. Our candidates look artificially great
over the longer window; planner over-commits to attacks the real opponent
will counter outside the projection.

**The empirical truth: horizon and opp-model depth must scale together.**
You cannot bump one without the other.

## Why this matters strategically

Stuck-in-cycle is THE failure mode in symmetric self-match. Every game
that hits step 500 in our episode logs likely has this signature. The
ladder probably contains many opponents that are themselves Producer-like
(public agent + light tweaks), so we lose μ to draws + slow-attrition
losses against well-matched opponents who happen to have a tiny
geometric edge.

Quantitative cost (rough): if 30 % of our games stalemate or lose because
of this defect, that's a μ ceiling of ~ -50 to -100 below what we could
reach with proper long-horizon reasoning.

## The shape of the proper fix

Multi-tick opp projection (deferred). Concretely:

1. At our turn N, run opp's planner (Producer-mirror) -> opp's launches
   for turn N. (Already done.)
2. Roll the world forward one tick with both my hypothetical launches
   and opp's projected launches applied.
3. From that rolled state, run opp's planner again -> opp's launches for
   turn N+1.
4. Iterate K times.
5. Pack the union of all K turns of opp launches as the background
   LaunchSet handed to our scorer.

Cost: K * (current opp_proj cost) per turn. With K=4 the per-turn opp_proj
cost goes from ~20 ms to ~80 ms. Total per-turn budget stays under 500 ms
even with H bumped to ~30. Feasible.

Risk: the roll-forward state has compounding error -- opp's response at
N+1 may diverge from reality if my predicted N's launches differ from
what I actually do. Per-step opp_proj would be more accurate but recurses
on every candidate, which is the wallclock disaster we explicitly
deferred. The compromise: project once per turn with my "most likely"
launches (the best from the previous turn's plan), roll forward, project
again. Open-loop in our actions but multi-step in opp's.

## Knowledge to carry forward

- The 18-tick horizon is the structural ceiling for this codebase.
  Everything above ~μ1300 likely requires breaking it.
- H and opp-model depth are coupled. Bumping either without the other
  regresses.
- The PI rejected the band-aid (cycle-detection). The next session's
  agenda should be multi-tick opp_proj (proper fix) OR the value head
  (ML solution that sidesteps the horizon issue entirely).
- The H knob is in main.py (`PRODUCER_PLUS_HORIZON_2P / _4P`) defaulted
  OFF; can be flipped on once multi-tick opp_proj lands.

## Open questions (carry to next session)

1. Is multi-tick opp_proj cheaper to implement than the value head? Both
   are 1-2 days of work; the value head needs training data + integration.
   Multi-tick opp_proj is pure code, no training.
2. What's the right K (projection depth) for multi-tick opp_proj? Trade
   between accuracy at deeper ticks and compounding error. Plausibly
   K=2-3 is sweet spot; experiment.
3. Does multi-tick opp_proj also help in 2P games or only 4P stalemates?
   2P games may not show the static-rotation pattern as strongly because
   one player can decisively win the 2-player war of attrition without
   exchange cycles forming.
