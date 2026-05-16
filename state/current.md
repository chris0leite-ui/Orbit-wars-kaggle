# state/current.md — current submitted agent + tournament rank

> Updated 2026-05-16 by `claude/recover-main-foundations-MV0e2` (v15 submit).
> Previous update 2026-05-16 (v13 submit). Live μ has shifted since:
> v12 1142.3 → 1095.4 (more games settled lower); v13 793.2 → 1063.8 (more games settled).
> All Score values pulled live from `kaggle competitions submissions orbit-wars`.

```yaml
date: 2026-05-16
days_to_deadline: 37                     # 2026-06-23 23:59 UTC minus today
current_submitted_agent: v15_banded      # PENDING; multi-wait + banded dedup
last_kernel_push: 2026-05-16 13:43:40 UTC
last_submission_id: 52710995
last_submission_status: PENDING
last_submission_file: submissions/v15.py  # 309 KB bundle; parity OK 458 turns
last_submission_message: |
  v15: multi-wait grid + wait-N for feasible-now pairs + banded
  (src, tgt, wait_band) dedup. Addresses PI directive on opening-game
  premature convergence: "chooser converges too early; we do not wait
  long enough; consider more actions."
  Three coordinated changes (agents/v15/main.py):
  (1) _wait_then_fire_candidate returns a list of variants (was single
      tuple or None). _WAIT_EXTRA_SURPLUS = (0, 5, 12) — three fleet
      sizes: just-enough, +5 surplus, +12 surplus. Larger fleet means
      longer wait but more robust capture less prone to opp counter-
      recapture in reactive-opp rollouts.
  (2) Generate wait-N candidates for FEASIBLE-now pairs too (the
      previous infeasible-only guard is gone). Feasible pairs now also
      get the multi-wait grid, so the chooser sees "wait longer to
      accumulate extra surplus" on planets it could otherwise fire-now
      at.
  (3) Stage-2 dedup changed from per-(src, tgt) → per-(src, tgt,
      wait_band) where wait_band buckets wait_N: 0 (fire-now), 1..7
      (short wait), >=8 (long wait). Load-bearing change: the rollout
      validator now compares fire-now vs short-wait vs long-wait for
      the same target. Previously cheap-Δ pre-selected wait_min (since
      cheap-Δ strictly decreases in wait_N for the same tgt) and longer
      waits never validated.
  Funnel A/B Forrest seed step 80: 27 wait variants survived dedup
  vs 10 in v13. 213tubo seed P1 flipped LOSS→WIN.
  Panel n=32: v7_0 84.4% Wlo=0.682, v4_planner 90.6% Wlo=0.758,
  v3.5.1 87.5% Wlo=0.719; worst Wlo 0.682 (v13's 0.666). PASS.
  Head-to-head v15 vs v12 n=32: 68.8% Wlo=0.514 INCONCLUSIVE.
  Head-to-head v15 vs v13 n=32: 68.8% Wlo=0.514 INCONCLUSIVE.
  Bench 3 games: p50=90 p95=258 max=480; over_1000ms=0.
  Panel max=906ms; head-to-head max=1245ms (occasional 4P outlier
  above 1000ms ceiling — PI accepted risk).
  Audit: f315dc7 commit + this session's diagnostic of Forrest 2P loss.

# Rolling-last-2 (Kaggle auto-keeps these two for final evaluation;
# the third push auto-evicts the previous oldest).
rolling_last_2:
  - {agent: v15_banded,   sub_id: 52710995, score: PENDING, status: PENDING,
     episodes: 0, note: 'just submitted; initial μ in 5-10 min, σ settles 5-6h'}
  - {agent: v13_reactive, sub_id: 52704189, score: 1063.8, status: COMPLETE,
     episodes: '~tens', note: 'current floor; v15 builds on v13 base'}
evicted_recent:
  - {agent: v12_principled, sub_id: 52699232, score: 1095.4, reason: evicted by v15 push}
  - {agent: v9_scavenge,    sub_id: 52687411, score: 1119.9, reason: evicted by v13 push}
  - {agent: v8_scavenge,    sub_id: 52684059, score: 1065.8, reason: evicted by v12 push}

# Team leaderboard score = max(rolling_last_2) = max(v15, v13=1063.8)
# Until v15 settles: team floor = 1063.8 (v13_reactive)
# Floor-protected: v15 cannot bring team below v13's 1063.8.
# v13 → v15 prediction: median 1100-1150, range 1050-1200.
tournament_rank_today: TBD / 2779 (await v15 settle)
our_best_rank: v13_reactive μ=1063.8 (#52704189 COMPLETE, ROLLING)
lb_top10_cliff: 1430                     # refresh on next pull

# CALIBRATION WARNING (3 consecutive submissions over-predicted live):
# - v3.5.1 (5/12): local +56.6% Wlo vs v3_snipe → live μ=945.6 (-150μ vs expected)
# - geo v3.1 (5/14): local +7pp / +31pp vs v7_0 → live μ=1004.9 (settled at floor)
# - iter v1 (5/14): local +10pp panel over v7_pv → live μ=1034.7 (-18μ vs v7_pv 1053.5)
# - iter v2 (5/15): structural 4P fix on iter_v1 → live μ=1028.2 (modest decline)
# Local-vs-live mapping has been roughly -20 to -30pp on every recent submission.
# v8_scavenge local panel +5-10pp over iter_v1; with same calibration shift,
# live could land 1010-1030 (sideways), 1030-1050 (modest gain), or
# 1050+ (recover v7_pv level). Floor: 1028.2 (iter_v2 stays).

# σ-awareness. Kaggle's published Score = μ − κσ already discounts uncertain submissions.
sigma_proxy:
  geo: 80-130              # ~5h since submit; σ band ~10-12 Score points
  v7_pv: ~80               # ~13h since submit; tightening
  # v7_0 family fully tight at ~64 episodes (σ band ~6 Score points).

submissions_used_today: 1                # geo #52643676 (09:10 UTC)
submissions_used_total: 16               # full live-submission ladder
plateau_days: 0
saturation_count: 0

# v3.2 (geo + gang_up only on top of v3.1) is built + bundled locally but NOT submitted.
# Eviction math: next push removes v7_pv (1064.4); only submit if new agent
# is decisively above 1064.4. Currently NOT proven.

# Full live ladder (most recent first; from `kaggle competitions submissions`).
live_submissions:
  - {agent: geo,                    sub_id: 52643676, submitted: 2026-05-14T09:10, score: 984.0}  # σ-floor
  - {agent: v7_pv,                  sub_id: 52630118, submitted: 2026-05-13T23:31, score: 1064.4}
  - {agent: v7_0_drop_one_rebuilt,  sub_id: 52607699, submitted: 2026-05-13T08:33, score: 1056.6}
  - {agent: v7_0_drop_one_original, sub_id: 52588156, submitted: 2026-05-12T17:36, score: 1081.5}
  - {agent: v4_planner,             sub_id: 52579863, submitted: 2026-05-12T14:25, score: 1038.6}
  - {agent: v7_minimax,             sub_id: 52568317, submitted: 2026-05-12T06:50, score: 1040.8}
  - {agent: v3.5.1,                 sub_id: 52565976, submitted: 2026-05-12T05:20, score: 945.6}
  - {agent: sigma_equiv_v1,         sub_id: 52565034, submitted: 2026-05-12T04:39, score: 1041.4}
  - {agent: v3.4_spoiler,           sub_id: 52556866, submitted: 2026-05-11T21:19, score: 995.4}
  - {agent: precision_v3,           sub_id: 52552139, submitted: 2026-05-11T17:00, score: 1011.4}
  - {agent: v3_snipe,               sub_id: 52544634, submitted: 2026-05-11T12:16, score: 1005.7}
  - {agent: v2,                     sub_id: 52532938, submitted: 2026-05-11T04:04, score: 966.1}
  - {agent: v1.2_roi,               sub_id: 52518060, submitted: 2026-05-10T14:59, score: 1006.9}
  - {agent: v1.1_arrival,           sub_id: 52509319, submitted: 2026-05-10T09:28, score: 565.7}
  - {agent: v1_orbitfix,            sub_id: 52507539, submitted: 2026-05-10T08:11, score: 568.0}
  - {agent: day1_baseline,          sub_id: 52497828, submitted: 2026-05-10T00:09, score: 303.2}

session_log:
  - 2026-05-14 — game-strategy-eda-roatN (this branch). Pulled v7_pv's
    own 30W + 42L corpus (sub 52630118) and re-ran Mine 4 on both
    buckets. Headline: median episode in our ladder is 180 turns, so
    the "last-100-turn endgame" Mine 4 was framed against doesn't
    occur — W/L split is decided by turn 100 (+30pp ship-share gap).
    v7_pv launches 0.44/turn in wins, 0.29 in losses, vs top-10's
    0.70. Wired --panel hardened preset (v7_0_drop_one + v3.5.1 +
    roi + baseline); 32-seed calibration: v7_0 mean_wr 78.6%,
    worst-Wilson-lo 53.4% (vs v3.5.1). Built cluster-conditional
    opening overlay (lib/opening_overlay.py + agents/v7_opening) —
    FALSIFIED: 17W/15L = 53% vs v7_0 on n=32, overlay-active games
    46%, pure-v7 fallback 80%. v2 sweep's apparent 67% was a
    broken-orbital_frac proxy forcing every board into the
    high-cadence cluster 3 — fixed but variant doesn't survive
    correct classification. 3 frictions logged (helper-reimplemented-
    inline-silently-wrong; broken-mechanism-yields-fake-positive-
    signal; soft-clusters-need-confidence-fallback). Findings:
    audit/2026-05-14-loss-mode-mine.md. Postmortem:
    knowledge-base/thoughts/2026-05-14-overlay-postmortem.md.
  - 2026-05-14 — simplify-fast-setup-azW8T (this branch's most recent session).
    Built fast.py (single-file iteration entry point, validated bit-identical
    vs audit-logged v7_1 result) + lib/geo/{sense,posture,allocator}.py
    (geometric primitives: clustering, Voronoi, front, threat, comet claims,
    posture arbiter, LP/greedy-multi allocators) + agents/geo/main.py v3.2
    (K=10 lookahead + 4 sense tilts + 2 archetypes + gang_up + 4P branch
    + SIGALRM timeout). 29 commits, 17 unit tests. Local A/B: vs v7_0
    n=192 = 57.3% (~+7pp), vs v3.5.1 n=128 = 57.0%, 4P vs 3×v7_0 n=128
    = 56.3% first-place (+31pp). Submitted as geo #52643676 → live
    μ=984.0 σ-discounted floor (not settled). Team score unchanged at
    1064.4 (v7_pv carries us). Five "fixes" REGRESSED: v2.4 lite_greedy
    (-17pp), v2.5 WALLCLOCK 350 (-20pp), v2.7 K=8 (-20pp), v3.0
    composite head (-19pp), v3.2 empty_out+tap_capture cumulative (-4pp).
    SIGALRM-based per-score timeout (v2.9) bounded wallclock max
    1500-2900ms → 1100-1200ms with no strategic cost.
    Postmortem: audit/2026-05-14-postmortem-geo-session.md.
    Knowledge: knowledge-base/thoughts/2026-05-14-geo-v2-iteration-results.md,
    knowledge-base/thoughts/2026-05-13-geo-v1-bisect-lessons.md.
    Frictions: 5 entries under `## 2026-05-14` in audit/friction.md.
  - 2026-05-13/14 — research-competition-analysis-2R8I3 (parallel branch,
    merged into main first). PV target valuation passes 32-seed local A/B
    68.8% vs v7_0 (Wilson [56.6, 78.8]); LIVE submitted as v7_pv #52630118
    at μ=1061.8 climbing → 1064.4 settled. Eight other interventions
    FALSIFIED monotonically. Architectural finding: v7 + PV is a tight
    local optimum for the K=10 drop-one chooser. Plumbing landed:
    3-anchor Wilson gate, PV_GAMMA in JAX, HAV helpers, Mission
    Renaissance flags. Postmortem:
    audit/2026-05-14-postmortem-research-competition-analysis-2R8I3.md.
  - 2026-05-14 — read-handover-iLWTq (parallel branch, merged into main).
    Seven v7_X variants (chooser-axis sweep) all FALSIFIED at 32-game
    scalar A/B vs v7_0. Best v7_4 = v7_6 = 40.6%. Chooser-axis design
    space EXHAUSTED — further refinements have negative marginal EV.
    Side wins: bundler parity-gate non-determinism fixed (env-var
    override); composite_capture_value value head; 3 new
    action-primitive enumerators; opp-archetype set. JAX depth-2
    parked (GPU compile too slow). Postmortem:
    audit/2026-05-14-postmortem-read-handover-iLWTq.md.

mechanism_families_explored:
  - heuristic-greedy-nearest-target          # comp-shipped baseline; μ=303
  - heuristic-orbit-aware-greedy             # v1: lead-prediction; μ=568
  - heuristic-orbit-aware-greedy-with-shared-mechanism-layer  # v1.1 arrival_size
  - simple-greedy-target-selection-variants  # nearest/production/roi/weakest/enemy_first
  - heuristic-physics-upgrade                # 5-iter aim + safe-intercept + sun-safe
  - heuristic-worldmodel-aware               # v2: WorldModel.owner_at dedup; μ=966
  - mission-framework-snipe-only             # v3.0 Mission dataclass + settle_plan
  - env-clone-forward-sim-scorer             # Sim<K> AUC=0.952 at K=50
  - lookahead-drop-one-candidates            # v3.1 v3_lookahead: 50/50 vs v2
  - full-trajectory-predict-fleet-fate       # lib/trajectory.py ray-cast
  - cost-aware-roi-additive-denominator      # additive cost in roi
  - comet-lifetime-correction                # time_to_hold caps on comet path
  - mission-framework-snipe-plus-reinforce   # v3_snipe submitted; μ=1005.7
  - aggressive-snipe-ship-sizing-v3.5.1      # v3.5.1 submitted; live μ=945.6 (REGRESSED)
  - sigma-equivariance-patches               # planner score-round + sym_hypot;
                                              # 16/16 v3-vs-v3 self-play draws; live μ=1041.4
  - v7-maximin-search                         # v7_minimax: live μ=1040.8
  - v4-receding-horizon-mission-portfolio    # v4_planner: live μ=1038.6
  - v7-drop-one-fast-brain                    # v7_0_drop_one: live μ=1081.5
  - v7-sweep-variants-failed                  # v7_1..v7_4 — all FAIL
  - v7-iteration-variants-failed              # v7.1..v7.6 super-versions — all FAIL
  - v8-psro-self-play-pool                    # parked
  - v9-super-version-failed                   # all FAIL
  - v10-evaluate-value-head                   # FAIL
  - pv-target-valuation                       # H16: live v7_pv μ=1064.4 (only winner since v7_0)
  - danger3-allegiance-multiplier             # H17: FALSIFIED
  - fleet-overcommit                          # H19: FALSIFIED
  - pre-reinforce-window                      # H21: FALSIFIED
  - drop-comet-targets                        # H15/H18: INCONCLUSIVE
  - mission-renaissance-trio                  # H30: FALSIFIED
  - hav-hold-aware-value                      # FALSIFIED
  - hav-holding-tier                          # FALSIFIED
  - chooser-axis-sweep-7-variants             # claude/read-handover-iLWTq: v7_1..v7_7 all FAIL
  - geometric-strategy-with-lookahead         # claude/simplify-fast-setup-azW8T:
                                              # geo v3.1 local +7pp 2P / +31pp 4P first-place;
                                              # live μ=984.0 σ-floor (TBD if settles higher)
  - composite-capture-value-head              # lib/value_heads.py composite_capture_value;
                                              # +9pp local in v7_4 vs v7_2 (32-game); but
                                              # v7_4 vs v7_0 = 40.6% FAIL. Reusable head.
  - cluster-conditional-opening-overlay       # claude/game-strategy-eda-roatN: KMeans(k=4)
                                              # on 60 top-10 boards + ROI-style proposer for
                                              # turns 0-30. FALSIFIED: 17W/15L = 53% vs v7_0
                                              # on n=32 (Wilson-lo 36%), overlay-active 46%,
                                              # pure-v7-fallback 80%. v2 sweep's 67% was a
                                              # broken-orbital_frac proxy. Code stays on
                                              # branch; learnings ported to main.

gate_status: cleared                        # full pytest passes;
                                            # geo's 17 tests + parallel branches' tests all green
```
