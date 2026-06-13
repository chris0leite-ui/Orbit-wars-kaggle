# 2026-06-11 — Oracle track day 1: data engine, two falsifications, the BC deadlock

PI directive: build an entirely new strategy able to beat all our agents
and perform on the leaderboard; ML/RL explicitly allowed; work autonomously.

## What was built (chronological)

1. **Ladder data engine.** The Kaggle episode service lists any
   submission's episodes (each carries every seat's submissionId + rating
   at game time, and every referenced team's current best submission), and
   any episode's replay is downloadable. Crawled the match graph from our
   submissions: 66,860 episodes / 10,579 submissions / 2,445 teams
   catalogued; ~1,000 replays of the highest-rated games downloaded
   (scripts/scrape_episodes.py; data/external/, gitignored). The crawl
   reached the #1 team (1752) in 9 hops.
2. **Oracle agent scaffold** (agents/oracle/): exact-physics engine ported
   from the parity-tested ledger World (own parity gate
   tests/test_oracle_engine.py, green), shared-horizon feature extraction,
   candidate shortlist, numpy nets, single-file bundler.
3. **Value net** (state -> P(win) + final share): test AUC 0.999.
4. **Policy net** (per (source,target) pair -> P(expert fires) + size
   fraction): PR-AUC 0.757 at base rate 1.3%; the expert's actual pair is
   our top-ranked candidate in 83.4% of states (top-3: 94.6%); size-frac
   MAE 0.054 around a 0.945 mean — data-confirmed full-drain doctrine.

## Falsification 1 — a value net cannot rank actions here

Probe (scripts/oracle_value_probe.py): inject the expert's actual launches
vs null vs random into the exact ledger and compare V on the planner's own
leaf path. Result: **V(expert action) > V(null) in only 15/60 states**
(median gap −0.0007). An outcome model with near-perfect AUC reads WHO IS
AHEAD, not WHICH ACTION HELPS — launching always looks like spending
inside a 32-tick feature window. Consequence: policy-primary architecture;
the value net is only a portfolio-level blunder veto.

## Falsification 2 — absolute-threshold behavior cloning deadlocks

With FIRE_THETA=0.25 the agent never launched a single fleet in full
games, despite healthy P values on replay states. Diagnosis chain:
- max P over candidates in OUR games: ~0.001 for 200+ turns;
- on TRUE expert states the same code gives sharp P (0.71 exactly when
  the expert fires at t=2);
- feature diff of the two states shows the pair head's signal rides on
  follow-up conditioning: `tgt_flips_to_me` (target already falling to a
  fleet in flight) and `my_inflight_frac` dominate. Initiation-from-cold
  states are one per game vs hundreds of continuation states, so a cold
  board scores ~0 everywhere -> the agent never initiates -> its states
  stay cold-board-like forever. A self-reinforcing off-manifold loop.

Fix shipped: dedicated **state-level initiation head** P(expert launches
anything | state-global features, immune to pair conditioning and to the
builder's negative subsampling because the global slice is constant within
a state) + rank-based pair selection (relative floor at 25% of the top
pair's P) + absolute override at P>=0.5. Trainer reports the calibration
table; planner knobs: ORACLE_THETA_STATE / ORACLE_PAIR_MIN / ORACLE_REL_KEEP.

## Measurement-integrity notes (per the 2026-06-12 corrections)

- Battery harness (scripts/oracle_battery.py) is sequential-solo with
  liveness asserts and per-game JSONL flushes.
- The dead-agent loader mode hit US once already (exec-without-__file__ in
  kaggle_environments path loading): agents/oracle/main.py now locates the
  package defensively and RAISES on import failure instead of quietly
  playing [].
- A first 12-seed battery vs v7_0 measured a dead (zero-launch) focal — 
  caught by the launches_focal column, root-caused to the deadlock above,
  battery aborted instead of recorded.

## Open at end of day

- Retrain with state head in flight; planner v3 (state-gated rank firing,
  same-turn coalition co-arrival accounting) untested in games.
- Value net rebuild on the full replay set pending (veto quality).
- 4P spot-check pending; bundle parity gate pending.

## Addendum — evening session (same day)

### Measured panel evolution (n=16 each, same seeds, liveness-asserted)

| build | v7_0 | ledger | Producer | champion |
|---|---|---|---|---|
| round-1 (corrected pairing) | **14/16** | 10/16 | 6/16 | 4/16 |
| + donation gate/forced defense/burst-4 (3 changes at once) | — | — | 0/10 ABORTED+REVERTED | — |
| + opponent-threat globals | 11/16 | — | **9/16** | 4/16 |

The threat build trades economist strength for rush defense; net zero
overall but profile-shifted. The round-1 build is pinned as
submissions/oracle_round1.py (single file, weights+features consistent
by construction, reproducible from commit 1e59da8).

Uniform loss anatomy vs the Producer family persists: all losses are
eliminations t99-189; ALL wins are 500-step economy games — the oracle
wins every game it survives.

### Router: three trigger designs falsified

1. Opening in-flight fraction (t<=14): all styles open identically
   (0.7-0.93 for everyone).
2. Fist concentration + booked-threat peaks (t8-36): degenerate while the
   opponent holds 1-2 planets (fist = 1.0 trivially).
3. Cumulative damage under champion-default play (t<=72): the champion's
   own aggression makes every matchup look like a brawl (economists show
   100-2000 ships landed).

Binding lesson: behavioral opponent classification from inside a game our
own agent shapes needs more than hand-picked thresholds; the router code
stays parked (agents/oracle_router/) without a validated trigger.

### Method violation logged

The donation-gate experiment changed three things in one battery (gate,
forced defense, MAX_WAVES). The 0/10 verdict could not be attributed.
Reverted wholesale instead of bisected — three battery-hours lost.

### In flight at write time

Chained same-turn conditioning (commitment features + per-pick re-scoring,
the PI's "combined actions" direction) — dataset rebuild + retrain
running; re-panel next.

## Day-2 addendum (2026-06-12) — the live probe verdict + cycle 1

**Live probe (sub 53594710, oracle_rw):** settled ~1004 (25W-35L over 60
episodes) — below the 1150-1300 prediction band. Loss autopsy on 20
downloaded replays: **15/20 are early eliminations (t65-190)**, in both
2P (9) and 4P (6) — the uniform loss law confirmed in production. The
900-1100 band is Producer-saturated; the agent is gated exactly where
the population is thickest. The 4P strength (42/64 first-place locally)
did not outweigh it.

**Checkmate finisher shipped** after the PI's live observation (won game
dragged 428 turns at 5x superiority): exact-engine forced kills when
provably won (one opponent, 6x score, <=3 planets). Validated on the
exact live failing state (fires the kill at t=231 of game 79631502) and
on the v7 spot battery (7/8, eliminations at t130-168 replacing 500-step
drags). In submissions/oracle_rw2.py.

**Self-play cycle 1, first attempt flawed and caught:** generation ran
with exploration temperature 1.0 — Gumbel noise swamped the policy's
log-probabilities and effectively randomized play (2/30 vs Producer vs
the sober 8/16). Lesson: noise scale must be validated against sober
win rates before trusting generated data. Regenerated deterministically
(board variety supplies diversity); the 20 live-loss replays (real
ladder winners beating the actual build) joined the lesson pool.

**Bundle versioning hole found:** submissions/* was gitignored; every
"pinned" bundle existed only on the ephemeral disk while commit messages
claimed otherwise. Whitelisted and committed (round1, threat, rw, rw2).

Rolling pair after the parallel branch's latest: their shotml 1286 +
our oracle_rw 1004 (older half -> next submission evicts the cheap probe).

### Cycle-1 self-play fine-tune: REJECTED by the frozen-panel gate

Lesson pool: 150 sober sparring games + 20 live-loss replays; winner-side
extraction (98.5% coverage, 1.24M rows); 70/30 elite/lesson mixture.
Result: Producer 8/16 -> 4/16, v7 14/16 -> 13/16. Diagnosis: winner-side
extraction imports the RUSHER'S STYLE when the rusher wins — its
fire-every-turn cadence shifted our calibrated rhythm into the opponent's
strength. Naive "learn from whoever won" is style-contaminated; cycle 2
needs phase- or role-filtered lessons (e.g. winners' decisions only while
the game was still even, or own-wins-only for consistency). Weights
restored to the rw2 set (committed state).

## Day-3 (2026-06-13) — RL lane: offline self-play fine-tuning is STRUCTURALLY CAPPED

PI directive: enhance IL with RL; sister branch runs rl_v7_selfplay_league
(live 947). Approved "both lanes" — our PPO/RL + IL handed to the sibling.

**Three offline fine-tunes have now all regressed the frozen gate:**

| attempt | Producer | v7_0 | vs rw2 baseline (8/16, 14/16) |
|---|---|---|---|
| chained same-turn conditioning | 3/16 | 11/16 | worse |
| cycle-1 winner-copy (BC on winners) | 4/16 | 13/16 | worse |
| AWR advantage-weighted (both seats) | 2/16 | 10/16 | worse |

AWR row-weight distribution was the tell: mean 1.13 but **p90 only 1.013**
— the value net almost never under-predicts a win, so AWR overwhelmingly
DOWN-weights (losing play) and barely UP-weights (winning play). Net: a
muddy drift off the IL optimum, no clear improvement direction.

**The structural conclusion (clean, and it unifies all three):** the
self-play pool is generated BY our own ~1018 policy. Imitating your own
games — under ANY weighting (copy-winner, advantage, chained) — cannot
exceed your own level; it pulls the 1500-1750 elite prior toward 1018
mediocrity. Only TWO things move the policy above IL level:
  1. **Better information** — the threat-feature representation fix is the
     ONLY change that improved a matchup (Producer 6->9/16), because it
     gave the policy a new signal, not a reweighting of old data.
  2. **True online RL** — fresh on-policy rollouts with a win/loss reward
     forcing improvement. That is the sister branch's online league; our
     offline-from-own-games approach is capped at IL by construction.

**Decision:** stop offline self-play fine-tuning on this branch (3 dead).
Our contribution to the IL+RL goal is now (a) ship the strongest IL build,
(b) hand it to the sister branch as RL initialization + a 1018 league
opponent — where above-IL improvement actually happens. Weights restored
to rw2 (committed). AWR pipeline kept (scripts) — it becomes useful once
an online-RL policy generates ABOVE-IL games to learn from.
