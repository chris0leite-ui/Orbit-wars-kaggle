# 49% of our turns are no-action despite having launch-able ships

Date: 2026-05-29 (PM2 session, after Rule 47 trace).

## The finding

One-game trace of the surgical-revert bundle (commit `2224324`, sub
53163774) vs the frozen PV_ETA anchor (sub 53111837 μ=1163.5), seed=7,
captured every turn:

- **300 fleets launched.** Of those, **0** ran into the sun, off-board,
  or timed out. predict_fleet_fate is doing its job everywhere it's
  called. The Rule 47 substrate is clean on this bundle.
- **265 turns of P0 decision-time.** On **139 of them (52.5%)** we
  emitted nothing. Of those idle turns, **130 (49.1%)** happened
  while we had a launch-able surplus (≥ 8 ships on at least one
  planet over MIN_LAUNCH_GARRISON).

So the chooser is saying NO to half our turns even when we have
ships ready to go. That's not the "wasting fleets to physics"
pattern I had partly expected — it's the opposite: we have ships
sitting on planets while the chooser silently passes.

## What it means

PI listed five symptoms before the trace: wasting fleets, sending
slow fleets far, weak opening, no streamlined attacks, underutilized
fleets on planets. The trace doesn't speak to all five, but it nails
one of them and reframes the others:

- **Underutilized fleets on planets (symptom 5)** — directly confirmed.
  Half our turns leave ships sitting when something could have been
  launched.
- **Wasting fleets (symptom 1)** — reframed. Waste isn't fleets-sent-
  into-the-sun (zero of those). Waste is fleets-not-sent-at-all when
  they could have been useful.
- **Slow fleets far (symptom 2)** — not directly tested. The launches
  we DO make could still be slow-far. But the bigger lever is
  closer: we're missing half our opportunities to launch ANYTHING.
- **Weak opening (symptom 3)** — partially implicated. We don't know
  yet whether the 49% idle is opening-heavy. If it is, fixing the
  chooser fixes the opening directly. Phase-split is the next probe.
- **Streamlined attacks (symptom 4)** — orthogonal. Would need a
  separate trace that looks at fleet-chaining candidate emission.

## The mechanism (hypothesis)

The 2P leaf scores "ships sitting on a planet" as bare ship-delta.
PV_ETA discounts the candidate-launch side by `γ^(wait_N + eta)`
but does NOT discount the no-action side. So "do nothing" has a
flat present value while "launch this candidate" has a discounted
one. Any candidate close to zero Δ loses to "do nothing" — even
when the planet's production WILL be valuable later but ONLY if we
hold ground in the meantime.

This is a Rule 40 modeling-correctness shape: the fix is symmetric
PV on both sides, not a hand-coded "MIN_DELTA_TO_LAUNCH" threshold
bump. The wasting/underutilized symptoms should emerge naturally
from a correctly-priced no-action leaf.

## What this implies for the live ladder

The frozen PV_ETA anchor I A/B'd against settled at live μ=1163.5
when fresh. If both my revert and the anchor share the same
over-patience pathology — which they likely do, the chooser change
between them is only the dead-wait-grid strip and patience restore,
not the no-action leaf — then the local 50/50 A/B is non-predictive
for this axis. Both agents idle half the time; both lose the same
games. Live ladder includes opponents who DON'T over-patience, and
that's where the gap shows up.

This explains the at-parity-vs-anchor reading without contradicting
the symptoms PI saw on the ladder.

## What we DON'T yet know

- Distribution of idle-with-ships across the 265 turns. Is it
  opening-heavy (turns 1-60), mid-game-heavy, or uniform? Decides
  whether the fix is opening-specific or chooser-wide.
- Distribution of Δ values for the candidates the chooser DID
  accept. If they're tightly distributed near a threshold, threshold
  is doing real work; if they're broad, threshold is mostly
  irrelevant and we really do need the leaf fix.
- Whether the no-action-with-ships pattern repeats across seeds. We
  ran one game. n=1 is a strong hint, not a proof.

## Next-session first action

Phase-split the 49% idle by step quartile [0-60, 60-130, 130-200,
200+]. Same trace, three lines of code. Decides scope of the
chooser-fix.

If the trace tool (currently `/tmp/trace_rule47.py`) survived the
container reclaim, reuse it. If not, recreate from the postmortem
sketch — the harness is 130 lines and we have the design.

Then: prototype the symmetric-PV no-action leaf and A/B it vs the
revert bundle. We're comparing to the bundle we just shipped, not
to the anchor (which shares the bug).
