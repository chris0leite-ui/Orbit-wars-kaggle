# state/ORACLE_STRATEGY.md — the oracle track (candidate new strategy)

> Status: IN DEVELOPMENT (2026-06-11, branch claude/elegant-dijkstra-uae6p0).
> This does NOT replace state/STRATEGY.md — promoting it is a PI decision.
> PI directive this session: "research and create an entirely new strategy
> on your own, that should be able to beat all our agents and perform well
> on the leaderboard. You may use machine learning… or a no-ML path.
> Think for yourself and work autonomously, but create something new."

## The strategy in one paragraph

Learn the *decisions* of the 1500-1750-rated ladder population directly
from their replays, and let our exact-physics engine guarantee execution.
Every turn the agent builds the exact future of every planet (the
parity-tested ledger), enumerates the decision surface a strong player
faces (attack waves, logistics transfers, just-in-time defenses, comet
evacuations), and asks a behavior-cloned policy net "which of these would
a top player fire right now, and how big?". Ship counts are then snapped
to exact-engine capture requirements at the true arrival tick, every shot
is verified by exact flight simulation, and a value net (trained on the
same replays) vetoes portfolios whose worst case against modelled replies
is worse than doing nothing.

Division of labor: **the policy says WHAT, the exact engine guarantees
HOW, the value net blocks blunders.**

## Why this design (evidence chain)

1. The prize zone starts ~1505 and the top is ~1713; our best live agent
   (producer_plus vetorf) sits ~1290. Hand-tuned pricing has been the
   recurring failure mode across 30+ audited sessions (mis-sized waves,
   churn, passivity bugs). Rule 40 says fix the model, not the constants —
   the strongest available model of "what is right" is the top of the
   ladder itself.
2. Kaggle exposes every episode's full replay (both seats' actions) and
   the episode service lets us walk the match graph from our submissions
   to the top teams without Meta Kaggle. We crawled 66,860 episodes /
   10,579 submissions / 2,445 teams and downloaded the highest-rated
   ~1,000 games (data/external/, gitignored).
3. Measured on those replays, the top agents' signature matches the
   audits' loss analyses: bimodal launch sizes (median ~11 skirmish +
   massive hammers, mean ~150), own->own logistics as the single most
   common move type, 98-99% landed-tonnage stick rate.
4. A pure VALUE-function planner was falsified first (audit-grade probe,
   scripts/oracle_value_probe.py): a net with test AUC 0.999 for "who
   wins from here" still prefers null over the expert's actual action 75%
   of the time — outcome prediction reads who is ahead, not which action
   helps. Hence policy-primary, value-as-veto.

## Components (all under agents/oracle/)

| File | Role |
|---|---|
| engine.py | Exact World: positions (orbits/comets), swept-disk flight, ledger walk, combat — ported from the ledger agent, parity-gated by tests/test_oracle_engine.py. Shared PLAN_HORIZON=90. |
| planner.py | source_states -> shortlist_pairs (the decision surface; ALSO used to label training data) -> policy scoring -> exact sizing/verification -> value veto -> emit. |
| policy_features.py | 42 per-(source,target) features: pair geometry/tempo, ledger-priced economics (required ships at true arrival, race margins), source/target state, global context. Shared train/serve. |
| features.py | 103 state features for the value net: per-player forecast trajectories probed at t+{0,4,8,16,32}, posture, frontline geometry, neutral-pool economics, phase. Incremental leaf() for search. |
| policy.py / value.py | numpy inference; weights embedded base64 (policy_weights.py / value_weights.py, generated). |
| main.py | Entry; loader hardened against the silent dead-agent mode (audit 2026-06-12). |

## Data pipeline (scripts/)

```
scrape_episodes.py crawl      # match-graph crawl, 429-backoff
scrape_episodes.py download   # replays, strongest episodes first
oracle_policy_dataset.py      # replays -> per-(state,pair) expert labels
oracle_dataset.py             # replays -> (state features, outcome)
oracle_policy_train.py        # fire+size heads -> policy_weights.py
oracle_train.py               # win+share heads -> value_weights.py
oracle_battery.py             # liveness-guarded sequential eval battery
oracle_value_probe.py         # the V-can't-rank-actions falsification probe
```

Dataset v1: 5.88M rows, 73.8k expert launches, shortlist coverage 73.6%
of expert moves (the uncovered tail is far attacks / long transfers).

## Gates before any submission (Rule 46 adapted)

1. tests/test_oracle_engine.py GREEN (forecast parity).
2. Battery vs panel, n>=32 live games each, liveness-asserted, solo:
   v7_0_drop_one, ledger_v1_4, producer (vanilla), live_vetorf_1291
   rebuild (data/external/live_vetorf_1291.py, sha 354001a8cabe).
3. 4P spot check (binary-reward parity baseline 4/16).
4. Bundle build + cold-load smoke + per-turn p99 < 1000 ms.
5. Rule 42 push-claim row + explicit PI sign-off (Rule 1).

## Open risks

- Behavior cloning inherits the experts' blind spots and the clone is
  usually a bit weaker than the original; the exact-sizing layer and the
  veto are the mitigations, self-play fine-tuning the upside path.
- 26% of expert moves are outside the candidate shortlist (far plays).
- The opponent-reply model for the veto is shallow (reinforce / counter-
  snipe / best base wave).
- 4P behavior rides on the same policy (trained on both 2P and 4P
  states); no dedicated 4P objective yet.

---

## Build inventory (updated 2026-06-12)

Pinned single-file bundles (weights + feature code consistent by
construction; the live agents/oracle package changes during training):

| bundle | profile (n=16 panels) | provenance |
|---|---|---|
| submissions/oracle_round1.py | v7_0 **14/16**, ledger 10/16, Producer 6/16, champion 4/16 | commit 1e59da8 |
| submissions/oracle_threat.py | v7_0 11/16, Producer **9/16**, champion 4/16 | commit a229168 |

Falsified generations (full evidence in audit/2026-06-11-oracle-track-day1.md
and the A/B logs): chained conditioning (11/6/3/0 — runtime commitment
feedback amplifies aggression), defense-dense data unweighted (3/16 both
with and without chaining — oversampling broke cadence calibration; calm
top-1 0.677 -> 0.583).

In flight: natural-frequency reweighted training on the defense-dense
data (no-chain) — the principled version of the defense pass.

## Uniform loss law (holds across ALL generations)

Every loss to the Producer family is an elimination between t99 and t189;
every win is a 500-step economy game. The oracle wins the long game —
including against the champion — and the open question is only surviving
concentrated mid-game pressure.

## Submission posture

The two ladder slots hold the champion family (~1244 settling) and
ledger_v1_2 (~855). The oracle's true live rating is bracketed, not
known: local panels cover four opponents; the ladder has 4,000+ teams.
Next decision for the PI: a probe submission of the best oracle bundle
(evicting the ledger_v1_2 slot, Rule 42 trivially green) to measure live
mu and harvest our own loss replays — the highest-value training data
that exists for the next cycle.
