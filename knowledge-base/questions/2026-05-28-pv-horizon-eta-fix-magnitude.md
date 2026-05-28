# Open question — how much μ does `pv_horizon(eta=fleet_eta)` recover?

The leaf scoring bug surfaced today (2026-05-28): `favor` calls
`pv_horizon(leaf_step, eta=0)` so production from a captured planet
is credited at the same weight regardless of how long the fleet took
to arrive. This is the structural source of the "small fleets on long
paths" scatter symptom that PI observed in the peak bundle.

The signature-clean fix is `pv_horizon(leaf_step, eta=fleet_eta)` —
existing function supports this; `favor` just doesn't pass `eta`.

**Open question:** what's the actual μ swing? Hypotheses:

- **+30 to +80μ** (optimistic): the discount kills the wasteful tail
  (2-ship 40-turn launches) without touching the productive head
  (cap-sized 10-15 turn captures), and the freed ships do something
  useful elsewhere.
- **±10μ** (null hypothesis): the scatter is net-neutral on this
  ladder's opp mix; suppressing it freed ships go idle and the
  effect cancels.
- **-30 to -50μ** (pessimistic): the chooser's leaf was implicitly
  calibrated around "production is worth ~99/prod regardless of
  eta" and discounting reveals that the OTHER scoring terms (ship
  margin, elim bonus) need recalibration to stay coherent. Same
  shape as the NEUTRAL_BONUS-into-v4 wiring regression.

**Required to answer:** isolated env-gated implementation +
instrumented trace measuring the Δ distribution shift + Phase 1
n=32 vs peak anchor + Rule 43 panel. The whole loop is a half-day
of compute on a 12-min full test baseline.

**Don't answer this question with a submit before running the loop.**
This is the same trap that produced today's μ=680 regression.

**Decision-blocker for the next session:** if peak-restore (sub
53099429) lands ≥ 1140, this is the highest-EV next iteration.
If peak-restore lands < 1100, fix the NEUTRAL_BONUS-plumbing
regression first (delete the 3 dead setdefaults from the wrapper,
re-bundle, submit) before adding any new behavior.
