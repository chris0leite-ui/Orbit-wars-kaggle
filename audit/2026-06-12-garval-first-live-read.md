# Garrison-value first live read (sub 53588922) — fires, but the alarm rings too late

Date: 2026-06-12 (~11 h after submit). Corpus: 47 live episodes pulled to
`audit/live-episodes/53588922/`.

## Headline numbers

| | garval (53588922) | coalition (53577315) |
|---|---|---|
| overall | 51.1% (24/47) | 48.7% (38/78) |
| 2-player | **70.4%** (19/27) | 65.9% (29/44) |
| 4-player | **25.0%** (5/20) | 26.5% (9/34) |
| rating | 1197.6 (warming) | 1241.6 |

No detectable 4-player lift. Sample is small (n=20) but the mechanism was
built specifically to move this number and it has not moved.

## Carve signatures are unchanged

`mine_4p_carving.py` on both corpora:

- median garrison the step before one of our planets falls: **10.0 in both**
- fraction of falls preceded by ≥5 ships launched OUT within 8 steps
  (self-drain signature): 74% garval vs 71% coalition
- median elimination step in losses: 123 garval vs 140 coalition

The garrison-value mechanism is not visibly changing live 4-player defense.

## Loss shape (`mine_4p_economy.py`, 15 losses)

We start ship-rank 1 at step 20, slip to rank 2 by step 40, rank 3 by 120.
The divergence axis is the **enemy-planet conquest race**: at step 60 the
eventual winner has taken 6 enemy planets to our 2; winner production
snowballs 12.5 → 38 while ours plateaus at ~8-12. Our total ships peak
near step 60 (~235) and then decline — we get carved while the winner
compounds.

## Probe: does the mechanism fire live? (episode 79615063→79622795, seat 1)

Replayed decision steps of a representative collapse (11 planets at step
100, dead by 160) through the planner with the exact live env config,
spying on the deficit-shortlist appender and the garrison-value bonus:

| step | appender | bonus credited |
|---|---|---|
| 60 | added 3 deficit planets | yes — 3 candidates, max 24.0 |
| 80 | added 2 | **zero** |
| 100 | added 2 | **zero** |
| 110 | added 2 | **zero** |
| 120 | added 2 | **zero** |
| 130 | added 1 | **zero** |

So the appender fires every step, and the bonus works early (step 60), but
during the entire collapse window the credit gate passes nothing.

## Why the bonus is zero during collapse

Extended probe at steps 80/110/120, printing each positive-deficit planet,
its deficit, and the candidates targeting it:

```
step  80  planet 23: deficit=46 g=2   candidates targeting it: 0
step  80  planet 26: deficit= 9 g=2   candidates targeting it: 0
step 110  planet 22: deficit=82 g=13  candidates targeting it: 0
step 120  planet 23: deficit=52 g=27  candidates: 33, valid: 0
```

- **Source-safety exonerated**: identical result with the drain cap off.
- Steps 80/110: **no source is within the planning horizon** of the
  threatened planet — the candidate builder produces zero (source, target)
  pairs for it.
- Step 120: pairs exist (a source one tick away) but none valid — nothing
  left to send.

## Diagnosis — live 4-player confirmation of the calibration frontier

This is the same geometry as the RYOTA 2-player loss
(`audit/2026-06-12-rotation-doctrine-and-massing.md`): at half threat
weight, the local balance-of-force deficit turns positive only when the
enemy mass is already close — at which point no friendly source can reach
the planet in time. The alarm and the impossibility of rescue arrive
together. The mechanism is built correctly; its detection threshold makes
it mute exactly when it is needed.

Full threat weight rings the alarm earlier but was locally refuted (mirror
−54%, panel share 14%). That is the frontier.

## A third resolution surfaced by this probe: extend the threat WINDOW, not the weight

The deficit at tick k counts enemy mass routable within k ticks, with k
capped at the same horizon K that caps our own reinforcement candidates.
An enemy stack 10 ticks away is invisible (threat within 8 = 0 → no
deficit), even though our reinforcement at 6 ticks could pre-garrison
before it lands. Defense must be planned on a horizon at least as long as
the enemy's reach, not equal to our own send horizon.

Extending the threat/deficit lookahead window beyond K gives lead time
WITHOUT inflating threat magnitude — it is the modeling-correctness fix
(Rule 40) where full weight was the constant-tuning fix. Candidate
mechanism, pending PI sign-off:

- compute the deficit over a window W > K (e.g. W = 2K) while
  reinforcement candidates stay at K;
- planets whose deficit goes positive anywhere in W become shortlist
  targets and earn credit while rescue is still feasible;
- mirror exposure plausibly small: magnitude is unchanged, only timing.

Alternatives already on the table: per-player-count weight, adaptive
weight, massing detector (enemy forward garrison growth rate).

## Status

- No new submission. Rolling pair: 53577315 (1241.6) + 53588922 (1197.6,
  warming). 4 of 5 daily slots unused.
- Next read: settled rating ~2026-06-13 04:24 UTC and a larger 4-player
  sample.
