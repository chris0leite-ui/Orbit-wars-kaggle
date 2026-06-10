# 2026-06-10 — the brawl window, two more nulls, and the defense-sizing idea

## What the replay mining actually said

The PI asked for a close look at the agent's foundation. Instead of guessing,
we mined the 195 live episodes of our best submission (the corpus behind the
"carved by 2+ opponents" headline) for mechanics:

- The intuitive story — "we strip garrisons to attack, then get carved" — is
  FALSE as a separator: 60% of our planet losses follow a recent outbound
  launch in wins AND losses alike. That churn is just how the game is played.
- The real signature: 4-player losses are decided between steps 20 and 80.
  We are the ship-count leader at step 20 even in games we go on to lose.
  Our production peaks around step 40 and then declines — we are already
  net-losing planets — while the eventual winner's production doubles.
  Elimination at median step 120 is cleanup.
- Neutral expansion is identical in wins and losses (stalls at 3). The
  separator is who wins the mid-early fights over already-owned planets.

Method note for future sessions: every "obvious" mechanism question we asked
the corpus (drain-carve link, defensive-width starvation, expansion speed)
came back NO. The one axis that survived is brawl outcomes converting into
compounding production. Mine before building.

## Two more measured nulls (4P axis, 3×producer pool, seeds 0–31)

- 4P-only multi-tick opponent projection (K=3): 10/32 vs baseline 13/32.
  Plausible modeling cause: each projection round re-plans rivals from the
  same tick-0 board without debiting previously committed ships, so the
  planner sees rivals attack with the same ships up to three times over —
  phantom aggression. A budget-debited multi-tick is a different, untested
  mechanism.
- (pending in this session: reinforce_deficit pool result — see audit doc.)

## The defense-sizing gap (new mechanism, default OFF)

Found by reading the enumeration, then sanity-checked against the engine:
for an owned target the planner's floor is 1, so the three multi-size
candidates for defense are (1, 2, full-drain). The exact "hold the planet"
size — the attacker's projected post-capture survivor + 1, which the
garrison projection already computes — is never enumerated. Implemented as
PRODUCER_PLUS_REINFORCE_DEFICIT. Two false starts worth remembering:

- Fleet speed RISES with fleet size (log curve to 1000), so right-sizing
  defense is a budget-retention play, not a speed play. Big rescue fleets
  arrive FASTER, not slower.
- The exact scorer already prices mis-sized sends correctly; the gap is in
  what the chooser gets to choose from, not in the scoring. Candidate-space
  gaps and scoring gaps need different fixes.

## Open questions

- Live A/B verdict (ffa_uniform 53527125 vs multi_opp_def 53523036) reads
  after ~2026-06-11 07:00 UTC. Identical 2P play; the μ gap is pure 4P.
- Is the 3×producer pool predictive of ladder 4P at μ≈1260 at all? The live
  A/B doubles as the calibration check: ffa_uniform was local-parity but
  self-pool-dominant; if it lifts live, weight the self-pool more.
