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
