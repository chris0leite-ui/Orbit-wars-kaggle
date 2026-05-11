# HANDOVER.md — next-session brief

> Last written: 2026-05-11 (Day 2 wrap) by the bootstrap-agentic-systems-lqnm6
> branch. Format budget ≤150 lines. Prior bootstrap-day-1 + competition-
> strategy-brainstorm sections archived to
> `audit/archive-2026-05-11-handover-day1-bootstrap.md`.

## Where we are

- **Comp:** Orbit Wars (slug `orbit-wars`). Deadline 2026-06-23 23:59 UTC →
  **43 days remaining.**
- **Submitted agent:** v3_snipe — `submissions/v3_snipe.py` (61.7 KB).
  Submission **ID 52544634**, pushed 2026-05-11 12:16:01 UTC, status
  **PENDING** at wrap time (kaggle CLI 503 on re-poll; will check next
  session). Rolling-last-2: `[v2 (μ=974.3), v3_snipe (PENDING)]`.
  v1.2/roi (μ=1006.9) evicted by the v3 push.
- **Gap to top-10 prize:** v2 at 974.3 → +473μ to cliff at 1447.6
  (ShunkiKyoya). v3 expected to lift +30-80μ; not enough alone — see
  next-session #5 below.
- **Daily submission budget:** 2/5 used today (v2 04:04 UTC + v3_snipe
  12:16 UTC). 3 slots remain.
- **Test suite:** 228/228 green. Bundle E.2 self-play 0/10 crashes.

## Today's progress

14 commits on this branch (657cddd .. f43e23f). Load-bearing only:

1. **Bootstrap infra hardening:** workers default = `cpu_count()`,
   `scripts/_agent_paths.py` (shared resolver), `scripts/ffa_tournament.py`
   + `scripts/ffa_panel.py` (4P FFA fixture — 33% of ladder games are 4P
   per live-replay analysis), `scripts/live_episode_summary.py` (post-
   submit diagnostic).
2. **Block E mission framework lands:** `lib/{mission,planner,missions/
   snipe,missions/reinforce}.py` + `agents/v3_snipe/`. Initial v3.0 was
   bit-for-bit parity with v2 (32/32 draws at step 500); v3.1 fills out
   the portfolio with reinforce + same-turn ledger.
3. **Lookahead probes (Phase 1a/1b/2):** static WorldModel captures
   only 14% of available predictive signal at step 50; one-turn action
   injection adds nothing (falsified); `env.clone()` + K-step forward
   sim closes the gap completely — **Sim<K=50> AUC = 0.952, matches
   perfect oracle.** Cheapest cost: 280 ms per evaluation. Audit:
   `audit/2026-05-11-lookahead-phase{1a,1b,2}-*.md`.
4. **v3.1 v3_lookahead MVP:** drop-one candidate enumerator + Sim<K=10>
   scorer. 8-seed showed 68.8% lift; 32-seed retest = 50/50 parity.
   Framework works; candidate set is too narrow. Audit:
   `audit/2026-05-11-v3-lookahead-mvp-parity.md`.
5. **Live fleet-loss fix:** `lib/trajectory.predict_fleet_fate` ray-casts
   the FULL fleet trajectory until first collision. Replaces endpoint-only
   sun_avoid / oob_guard / path_clears_other_planets. **Capture probe:
   reached 77.2% → 93.0% → 97.2%; OOB 7.5% → 2.6% → 0.3%; sun 3.2% →
   0.1% → 0.0%.**
6. **The four ROI-doc shortcomings:** all fixed.
   - Comet lifetime: `comet_remaining_lifetime` helper; time_to_hold caps
     by `len(path) - path_index` for comet targets.
   - Defence: new `lib/missions/reinforce.py` builds reinforcement
     missions when WorldModel timeline predicts our planet to flip.
   - Same-turn ledger: `settle_plan` tracks per-target pending-arrivals
     from earlier this-turn picks; gang-up allowed, overcommit skipped.
   - Mission classification: v3_snipe now calls snipe + reinforce
     builders, settle_plan arbitrates across classes.
7. **v3_snipe verification:** 32-seed 2P vs v2 = 37/64 = **57.8%**
   (Wilson [45.6, 69.2]); 16-seed 4P FFA = 58/64 = 90.6% (parity within
   Wilson overlap vs v2 92.2% / roi_baseline 93.8%). E.2 gate 0/10.
8. **Bundler fix:** twice in one day, new lib modules silently broke
   bundles. First bundle attempt crashed 10/10 (`NameError: 'propose_
   snipe_missions' is not defined`); `DEFAULT_LIB_ORDER` now includes
   `mission`, `missions/snipe`, `missions/reinforce`, `planner`.

## Falsified-or-dead

- **One-turn action injection** as a cheap lookahead extension (Phase
  1b). Hours50 ≈ Hall50 ≈ H50 within 0.005 AUC across the whole probe
  table. The 32pp oracle gap lives in the SEQUENCE of K future turns,
  not the boundary turn.
- **Drop-one candidate enumerator** as a Sim<K> lift mechanism (v3.1
  lookahead MVP). 32-seed 50/50 parity vs v2. Drop-one is purely
  subtractive — never proposes candidates v3_snipe didn't already
  consider; v2's launches are mostly individually positive-EV so the
  scorer rarely disagrees enough to flip a decision.
- **No-double-commit as a planner-level filter** (first settle_plan
  attempt at Block E). Hurt parity with v2 on dense boards where
  legitimate gang-up was being prevented. Replaced with same-turn
  arrival ledger (gang-up allowed when defender exceeds one source's
  fleet).

## Next-session first-action

Ranked. EV is from local-panel + live-data evidence; cost is wallclock.

1. **Pull v3 live μ + episode summary** (cost: 5 min, EV: high).
   `KAGGLE_API_TOKEN="$KAGGLE_KEY" kaggle competitions submissions
   orbit-wars` confirm v3 reached COMPLETE. Then
   `python -m scripts.live_episode_summary 52544634 --pull` for the
   2P/4P split + who-beats-us breakdown. **Calibration:** does the
   32-seed 2P 57.8% point estimate translate to a live μ lift, or
   does the Wilson lo at 45.6% land us at parity with v2 (974)?
2. **Comet_aim re-enablement test** (cost: ~1h, EV: medium). The
   mechanism was excluded for a 22.5% ablation regression; the
   regression was likely caused by the same endpoint-only path-check
   bug we fixed today. Add to DEFAULT_MECHANISMS in a v3.2 build;
   local A/B vs v3_snipe; ship if lift.
3. **Recapture mission class** (cost: ~2-3h, EV: medium). Sibling
   to reinforce — for planets we recently lost (or are about to lose
   despite reinforce being too late), score "can we re-take this
   before the enemy fortifies?" Roman has this; known-value mission
   class.
4. **(THE CEILING-RAISER) Richer Sim<K> candidate enumerator**
   (cost: 1-2 days, EV: high but uncertain). Two principled options:
   (a) sibling-strategy candidate — score v3_snipe's choice AND a
   different strategy's choice via Sim<K>, pick best. (b) per-source
   top-2 swap — for sources where the top mission has the smallest
   margin over its second-best, propose swapping. Phase 2 oracle gap
   was 32pp; this is mathematically the only path with enough headroom
   to close the +473μ gap to top-10.
5. **(Defer)** RL training (B.4), 4P-specific tuning (premature until
   we see actual 4P ladder behaviour of v3), full Roman mission portfolio
   (gang_up / elimination — diminishing returns past recapture without
   lookahead in the loop).

## Pointers (added today)

- `audit/2026-05-11-lookahead-phase1a-substrate-fitness.md` — oracle
  gap measurement: static WorldModel captures 14% of available signal.
- `audit/2026-05-11-lookahead-phase1b-action-injection.md` — one-turn
  action injection falsified; signal lives in K-turn sequence.
- `audit/2026-05-11-lookahead-phase2-forward-sim.md` — env.clone() +
  step() Sim<K> matches perfect oracle (AUC 0.952).
- `audit/2026-05-11-v3-lookahead-mvp-parity.md` — drop-one candidates
  yield parity; framework needs richer enumerator.
- `audit/2026-05-11-block-e-snipe-mvp.md` — Block E v3.0 refactor parity
  vs v2; no-double-commit planner rule falsified.
- `audit/2026-05-11-postmortem-bootstrap-agentic-systems-lqnm6.md` —
  this session's decision-quality review. Three promotion candidates
  awaiting PI ratification.
- `audit/tournaments/20260511T{052542,055428,055652,064949,070745,
  102311,112936}Z.json` + `ffa-panel-*` — all today's tournament JSONs.
- `audit/lookahead/20260511T0{61813,62658,63556}Z.json` — Phase 1a/1b/2
  probe artifacts.
- `audit/2026-05-11-capture-success-probe.json` — capture probe
  after-the-fix (97.2% reached).
- `audit/live-episodes/52532938/summary.json` — v2 live-episode
  aggregator output (15/21 wins, 2P 64% / 4P 86% split).
- `submissions/v3_snipe.py` — 61.7 KB bundle, gitignored; rebuild via
  `python -m scripts.bundle_agent agents/v3_snipe`.
