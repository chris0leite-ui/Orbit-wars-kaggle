# 2026-06-04 — port pressure into chooser, or do goal-directed 2-hop?

**Open question.** The three failed circulation attempts this session
all collapse to the same root cause: our chooser doesn't think in
pressure, so pressure-routed ships go to destinations the chooser
ignores.

Two non-trivial paths to unblock ship-utilization as a mechanism:

1. **Port Biel's pressure-aware scoring into our chooser.** Replace or
   augment `cheap_marginal_value` so attack candidates include a
   pressure-weighted term. Then a pressure-gradient regroup naturally
   feeds the planner. Big project; touches the scoring core.

2. **Goal-directed 2-hop pre-positioning.** Instead of "send to
   pressure", identify a specific (rear → mid-friendly → opp) sequence
   where the chooser is one launch short of a positive-EV trade.
   Pre-position only for THAT play. Smaller; chooser-aware by
   construction. Requires a 2-hop search over candidate plays.

**Which one first?** Path 2 has lower risk and could be A/B'd in a
single session. Path 1 has higher potential but is a multi-session
build. Default to 2 if PI re-prioritises ship utilization, but only
after identifying a concrete failed-2-hop case in a live replay (per
`replay-mine-baseline-v15-fleet-waste` precedent).

**Pre-requisite for either**: a replay-mining pass that surfaces 3-5
concrete cases of "we could have captured X if Y had Z more ships, and
rear stockpile W had spare". Without that grounding, both paths risk
the same "feature without a failing case" failure mode that killed v15
overlays.
