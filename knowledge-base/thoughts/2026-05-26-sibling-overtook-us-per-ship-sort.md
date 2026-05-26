# 2026-05-26 — Sibling overtook us; per-ship-sort is the lift we predicted

## What I observed (live-state poll at session start)

Submitted K1+Z v2 yesterday (2026-05-25 11:54). It settled at **μ=1118.6**
— inside the predicted band 1100-1180, on the low end. That's the
"parity-band on the ladder" branch of yesterday's decision tree.

Meanwhile, sibling ESwSv shipped two more submissions:

- **53024913** at 15:44 — `baseline_ev_per_ship` with one env flag,
  `BASELINE_SORT_BY_EV_PER_SHIP=1`. Sorts positive-EV candidates by
  `score/ships` instead of `score`. Their local A/B used **our new
  5×250×no-swap standard**: 15/20 = 75% pooled, Wilson [0.541, 0.886].
  Settled μ=**1136.4**. Strong half of rolling pair.
- **53032723** at 23:02 — `baseline_unified` learning submit, mixed
  local, settled μ=1063.1. Weak half.

We're now 0 in the rolling pair.

## Why this is interesting

The sibling's win **is conceptually adjacent to our Layer V1 idea**.
On 2026-05-25 morning I explored `OPENING_VALUE_PER_SHIP=1` (per-ship
ranking in the BUILDUP MILP). Layer V1 stayed env-gated default-off
after the n=64 A/B showed it was redundant with K1's path. Sibling
applied the SAME concept to the **consolidation chooser** (not the
BUILDUP MILP) and got μ +20.

The difference:
- V1 (ours): per-ship sort in MILP objective during BUILDUP (~6% of
  decisions, opening turns).
- EV_per_ship (sibling): per-ship sort in chooser during CONSOLIDATION
  (~94% of decisions, midgame + late).

Same hypothesis, applied in a vastly more leveraged subsystem. The
lesson is brutal: when a hypothesis is right, **apply it everywhere
the math holds, not just in the smallest local subsystem.**

## Cross-branch coordination friction

We submitted K1+Z v2 → 4h later sibling submitted ev_per_ship → 8h
later sibling submitted unified → our K1+Z v2 settled (we were
asleep). By the time we look at live μ, we're already evicted.

The push-claim-board mechanism in `state/MULTI_BRANCH.md` is supposed
to prevent this but requires both branches to file claims pre-submit.
Sibling's commits show they're not using it consistently. Net: live
ladder is effectively first-come-first-served with retroactive
documentation. Future submits from our branch should assume the
rolling pair will turn over in ≤6h.

## Predicted vs actual calibration (Rule 26 snapshot, retro)

- Sub 53018599 K1+Z v2: predicted 1100-1180, actual 1118.6 — **WITHIN BAND**.
  Low end of band; n=64 vs phi1_only's 56.2% mapped to ladder roughly
  as expected. No PI override needed retrospectively — the parity-band
  prediction was honest.

Four-of-four recent predictions within or above band (sub 53000996 1115.2,
sub 52993021 1117.9, sub 52968889 1142.4 over, sub 53018599 1118.6). No
`pi-stamp-risk` trigger yet.

## What to do next

P1 in HANDOVER: cherry-pick sibling's `BASELINE_SORT_BY_EV_PER_SHIP=1`
onto our K1+Z v2 stack. Expected composite: their behavioral lift +
our wallclock fix.

The composite risk is real but specific: K1 is bit-parity, so it can't
mis-compose. Z v2 (proposer-side pre-filter) and the sibling's flag
(chooser-side post-filter) operate at different stages of the pipeline
— but the chooser receives the filtered candidate set from the proposer,
so Z v2's rejections do affect which candidates the per-ship sort sees.
The right validation is the 5×250×no-swap panel + a n=16-32 confirm vs
joint_aggr.

## Open questions

1. Did sibling's flag composes with K1 or did they ship without K1?
   (Their max=1018 ms suggests no K1, which is good news for us — K1
   absorbs that and gives us margin to push another behavioral change.)
2. Is Layer V1 (BUILDUP-side per-ship) STILL worth shipping if the
   chooser-side per-ship is in place? Plausibly yes for early turns
   where BUILDUP owns the decision.
3. The sibling's flag converts wait-N commits into fire-now (per
   their description). Is wait-N strategically wrong, or just a code
   artifact of insufficient discounting?
