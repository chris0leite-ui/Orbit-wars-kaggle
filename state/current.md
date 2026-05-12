# state/current.md — current submitted agent + tournament rank

> Updated 2026-05-12 21:40 UTC on the consolidation branch. All Score
> values pulled live from `kaggle competitions submissions orbit-wars`
> and `kaggle competitions leaderboard orbit-wars -d`.

```yaml
date: 2026-05-12
days_to_deadline: 42                     # 2026-06-23 23:59 UTC minus today
current_submitted_agent: v7_0_drop_one
last_kernel_push: 2026-05-12 17:36:01 UTC
last_submission_id: 52588156
last_submission_status: COMPLETE
last_submission_file: submissions/v7_0_drop_one.py  # 121 KB bundle; sha256:bb7ab23a75bc5865
last_submission_message: |
  v7_0_drop_one: fast_sim (183× faster than env.clone+step, bit-exact)
  + drop-one chooser (incumbent + each-launch-dropped) + K=10 forward
  sim + Tier-1 (v3.5.1) opp mirror + 4P → v3.5.1 fallback. Local A/B:
  79.2% vs v7_minimax (Wilson lo 59.5%) PASS, 75.0% vs v4_planner
  (Wilson lo 55.1%) PASS. Live: μ=1094.9 — TEAM PEAK.

# Rolling-last-2 (Kaggle auto-keeps these two for final evaluation;
# the third push auto-evicts the previous oldest).
rolling_last_2:
  - {agent: v4_planner,      sub_id: 52579863, score: 1038.6, episodes: 58}
  - {agent: v7_0_drop_one,   sub_id: 52588156, score: 1094.9, episodes: 64}  # team peak
evicted_recent:
  - {agent: v3.5.1,          sub_id: 52565976, score: 945.6,  reason: regression vs prediction}
  - {agent: v7_minimax,      sub_id: 52568317, score: 1040.8, reason: evicted by v4_planner push}
  - {agent: sigma_equiv_v1,  sub_id: 52565034, score: 1041.4, reason: evicted earlier}

# Public ladder snapshot (2026-05-12 21:40 UTC).
tournament_rank_today: 109 / 2587 teams (top 4.2%)
our_best_rank: v7_0_drop_one μ=1094.9 (#52588156 COMPLETE)
lb_top10_cliff: 1430.9                   # 3Comets, #10. #1 = bowwowforeach 1675.9
headroom_to_top10_prize: +336 μ          # v7_0_drop_one at 1094.9 → +336μ to 3Comets cliff

# σ-awareness. Kaggle's published Score = μ − κσ already discounts
# uncertain submissions. Episode counts at this LB snapshot:
sigma_proxy:
  v7_0_drop_one: 64                      # ~4h since submit; σ band ~6 Score points
  v4_planner:    58                      # ~7h
  v7_minimax:    63                      # ~15h, fully tight
  v3.5.1:        53                      # ~17h
  sigma_equiv:   36                      # ~17h
  v3_snipe:      65                      # ~33h
# v7_0's +56 lead over v4_planner is outside any plausible σ band → real.
# v3.5.1's −149 vs v7_0 is well outside σ → regression is real, not noise.

submissions_used_today: 2                # v3.5.1 #52565976 (05:20 UTC) + v7_0_drop_one #52588156 (17:36 UTC) from this branch's lineage
submissions_used_total: 13               # full live-submission ladder; 10 in the table below
plateau_days: 0
saturation_count: 0

# Full live ladder (most recent first; from `kaggle competitions submissions`).
live_submissions:
  - {agent: v7_0_drop_one,  sub_id: 52588156, submitted: 2026-05-12T17:36, score: 1094.9}
  - {agent: v4_planner,     sub_id: 52579863, submitted: 2026-05-12T14:25, score: 1038.6}
  - {agent: v7_minimax,     sub_id: 52568317, submitted: 2026-05-12T06:50, score: 1040.8}
  - {agent: v3.5.1,         sub_id: 52565976, submitted: 2026-05-12T05:20, score: 945.6}
  - {agent: sigma_equiv_v1, sub_id: 52565034, submitted: 2026-05-12T04:39, score: 1041.4}
  - {agent: v3.4_spoiler,   sub_id: 52556866, submitted: 2026-05-11T21:19, score: 995.4}
  - {agent: precision_v3,   sub_id: 52552139, submitted: 2026-05-11T17:00, score: 1011.4}
  - {agent: v3_snipe,       sub_id: 52544634, submitted: 2026-05-11T12:16, score: 1005.7}
  - {agent: v2,             sub_id: 52532938, submitted: 2026-05-11T04:04, score: 966.1}
  - {agent: v1.2_roi,       sub_id: 52518060, submitted: 2026-05-10T14:59, score: 1006.9}
  - {agent: v1.1_arrival,   sub_id: 52509319, submitted: 2026-05-10T09:28, score: 565.7}
  - {agent: v1_orbitfix,    sub_id: 52507539, submitted: 2026-05-10T08:11, score: 568.0}
  - {agent: day1_baseline,  sub_id: 52497828, submitted: 2026-05-10T00:09, score: 303.2}

session_log:
  - 2026-05-12 EVE — consolidate-fast-simulation-ysd9M (this branch).
    Merged origin/claude/game-theory-strategy-analysis-0oH4N which itself
    merged claude/game-ai-lookahead-3ucqH. Result: one branch carrying
    the full fast-brain stack (lib/fast_sim.py, lib/opp_model.py,
    lib/v7_search.py, lib/lookahead_planner.py, lib/value_heads.py,
    lib/candidate_portfolios.py, lib/mirror.py) plus σ-equivariance
    patches in lib/planner.py + lib/orbit.py plus v7_0_drop_one as the
    live anchor agent (μ=1094.9). Pruned: failed v7_1..v7_6 sweeps,
    v7_combined/v7_ablations cruft, parked v8_* PSRO stack, failed v9_*
    and v10_* variants, intermediate v3.5/v35_ablations/v35_iter2
    scaffolding, intermediate v4_endgame/hybrid/mirror sweeps, dead
    A/B harness scripts (psro_*, run_v7_ablation, run_iter2_ablation,
    run_v35_ab, run_sizing_sweep, run_aggressive_sizing_32, etc.).
    Test gates: 373 passed / 2 skipped / 1 xfailed (v3_snipe replay
    parity drift from σ-equiv patches — documented, not a regression
    of live behaviour). State files rewritten to true live scores.
    Next phase: research a 100%-accurate pure-Python rebuild of
    `kaggle_environments.envs.orbit_wars.orbit_wars.interpreter()` so
    we can iterate on the fast brain without the Environment overhead
    or kaggle_environments dependency.
  - 2026-05-12 EVE — game-ai-lookahead-3ucqH (parent, archived in this
    branch): full v7 + v8 + v9 + v10 super-version iteration. No
    super-version cleared Wilson 55% gate vs v7_0; ship target =
    v7_0_drop_one. Architecture: fast_sim simulator (183× faster than
    env.clone+step, bit-exact) + opp_model (Tier-0 v3_snipe, Tier-1
    v3.5.1 mirror). v7_0 beats v7_minimax 19/24 = 79.2% Wilson lo 59.5%
    PASS; v7_0 beats v4_planner 18/24 = 75.0% Wilson lo 55.1% PASS;
    nothing else cleared 55%. Diagnosed v4_planner's receding-horizon
    pathology (audit/2026-05-12-v4-planner-receding-horizon-pathology.md):
    K-step rollout with noop in portfolios prefers "wait" when target
    eta>K; drop-one architecture structurally sidesteps this. σ-equiv
    REVERTED from drop-one regime (v7.6 bisect: −54pp). v10_evaluate
    62.5% Wilson lo 42.7% FAIL by 2.3pp. Knowledge anchor:
    knowledge-base/concepts/lookahead-simulator-architecture.md
    (permanent reference).
  - 2026-05-12 — research-lookahead-strategy-kfRsy: v4_planner
    receding-horizon mission-portfolio search, 5 portfolios per turn,
    Sim<K=6-10> with production-share+denial value head. Submitted
    #52579863 (live score 1038.6). σ-equivariance lib patches cherry-
    picked from game-theory-strategy-analysis-0oH4N (3 commits):
    lib/planner score rounding + symmetric tie-break, lib/orbit
    sym_hypot for bit-exact paired distances.
  - 2026-05-12 — analyze-leaderboard-strategies-sdZlE: v3.5.1
    aggressive snipe sizing (32-seed 2P vs v3_snipe 68.8% Wilson lo
    56.6% PASS local). Submitted #52565976 — **live regression
    μ=945.6**. Local A/B over-predicted by ~150 points; the local
    panel didn't capture whatever the live ladder rewards. Lesson:
    σ-equiv-base agents play tighter draws against the broader ladder
    than against v3_snipe alone — aggressive sizing helps locally
    but evidently doesn't generalise.
  - 2026-05-11 PM — analyze-submission-logs-dFHeS: live↔local parity
    100% (postmortem off-by-one fix in scripts/episode_postmortem.py).
    v3.2 / v3.4 internal builds (not submitted as standalone scores).
  - 2026-05-11 — bootstrap-agentic-systems-lqnm6: Block E mission
    framework MVP; lookahead probes (env.clone() Sim<K=50> AUC=0.952);
    v3_snipe submitted #52544634 (live μ=1005.7).
  - 2026-05-10 — earlier sessions archived under
    audit/archive-2026-05-10-handover-prior-pm-sessions.md.

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
  - v7-maximin-search                         # v7_minimax: N=2 × M=2 × Sim<K=3>;
                                              # live μ=1040.8
  - v4-receding-horizon-mission-portfolio    # v4_planner: 5 portfolios × Sim<K=6-10>;
                                              # live μ=1038.6
  - v7-drop-one-fast-brain                    # v7_0_drop_one: lib/fast_sim (183× faster)
                                              # + drop-one chooser + Tier-1 opp mirror;
                                              # live μ=1094.9 TEAM PEAK
  - v7-sweep-variants-failed                  # v7_1_target_swap, v7_2_ship_sweep,
                                              # v7_3_archetype, v7_4_hungarian — ALL FAIL
                                              # at Wilson 55%; pruned from this branch
  - v7-iteration-variants-failed              # v7_1_minimax through v7_6_no_recapture —
                                              # ALL FAIL or PARITY; pruned
  - v8-psro-self-play-pool                    # PSRO infrastructure built; degenerate
                                              # Nash with pure v7. Parked; pruned
  - v9-super-version-failed                   # v9_inflight, v9_k15, v9_combined,
                                              # v9_opening — none clear Wilson 55%; pruned
  - v10-evaluate-value-head                   # drop-one + K=10 + evaluate_value head;
                                              # 62.5% Wilson lo 42.7% FAIL; pruned

gate_status: cleared                      # 373 pytest pass / 2 skipped / 1 xfailed
                                            # (v3_snipe replay-parity drift from σ-equiv
                                            # is documented, not a regression)
```
