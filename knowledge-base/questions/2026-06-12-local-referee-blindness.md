# Can ANY local referee reproduce live overextension conditions?

The shot-MLP session proved a sharp version of the long-suspected
local-vs-ladder calibration gap: our live agents make low-probability
attacks (49.5% of attack launches, 13.9% success) ONLY against the real
field — locally (vs producer, vs the old champion, in 4P producer
pools) the same planner emits essentially none (1 sub-threshold wave in
158 scored across 3 full games). Any mechanism whose value is
"stop doing the thing you only do against real opponents" is
unmeasurable with the current local zoo.

Candidate answers, untested:
- Vendor 1-2 MORE third-party public agents (like the Producer was) to
  diversify the local field.
- Replay-driven referee: replay a live loss turn-by-turn and check the
  mechanism's counterfactual decision at the recorded loss points
  (decision_trace.py does this shape of thing for other mechanisms).
- Accept that this mechanism class is live-probe-only and budget
  submission slots accordingly.
