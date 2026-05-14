# state/current.md — current submitted agent + tournament rank

> Updated 2026-05-14 02:00 UTC on `claude/research-competition-analysis-2R8I3`.
> All Score values pulled live from `kaggle competitions submissions orbit-wars`.

```yaml
date: 2026-05-14
days_to_deadline: 40                     # 2026-06-23 23:59 UTC minus today
current_submitted_agent: v7_pv
last_kernel_push: 2026-05-13 23:31:38 UTC
last_submission_id: 52630118
last_submission_status: COMPLETE
last_submission_file: submissions/v7_pv.py  # 205 KB bundle; v7_0_drop_one + PV_GAMMA=0.99
last_submission_message: |
  v7_pv: v7_0_drop_one + PV_GAMMA=0.99 (geometric production horizon).
  Replaces linear `(500 - step - eta)` with γ^eta · (1−γ^h)/(1−γ),
  γ=0.99. 32-seed local A/B vs v7_0_drop_one: 44/64 = 68.8% Wilson
  [56.6, 78.8] PASS. H16 / TID 699003. Live: μ=1061.8 and climbing
  (still settling ~3h post-submit).

# Rolling-last-2 (Kaggle auto-keeps these two for final evaluation;
# the third push auto-evicts the previous oldest).
rolling_last_2:
  - {agent: v7_0_drop_one_rebuilt, sub_id: 52607699, score: 1043.4, episodes: ~150}
  - {agent: v7_pv,                 sub_id: 52630118, score: 1061.8, episodes: ~50}  # climbing
evicted_recent:
  - {agent: v7_0_drop_one_original, sub_id: 52588156, score: 1081.5, reason: evicted by v7_0_drop_one rebuild push (2026-05-13 morning)}
  - {agent: v4_planner,             sub_id: 52579863, score: 1038.6, reason: evicted by v7_pv push}

# Public ladder snapshot (2026-05-14 02:00 UTC).
tournament_rank_today: ~100 / 2600 (top ~4%; PV still tightening σ)
our_best_rank: v7_pv μ=1061.8 (#52630118 COMPLETE, still climbing)
lb_top10_cliff: ~1430                   # estimate; refresh next session
headroom_to_top10_prize: ~+370 μ        # PV climbing toward v7_0 rebuilt; long way to cliff still

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

submissions_used_today: 1                # v7_pv #52630118 (23:31 UTC 2026-05-13; counts toward 2026-05-14 budget per Kaggle UTC day)
submissions_used_total: 15               # full live-submission ladder
plateau_days: 0
saturation_count: 0

# Full live ladder (most recent first; from `kaggle competitions submissions`).
live_submissions:
  - {agent: v7_pv,                  sub_id: 52630118, submitted: 2026-05-13T23:31, score: 1061.8}
  - {agent: v7_0_drop_one_rebuilt,  sub_id: 52607699, submitted: 2026-05-13T08:33, score: 1043.4}
  - {agent: v7_0_drop_one,  sub_id: 52588156, submitted: 2026-05-12T17:36, score: 1081.5}
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
  - 2026-05-13/14 — research-competition-analysis-2R8I3 (this branch).
    PV target valuation passes 32-seed local A/B 68.8 % vs v7_0
    (Wilson [56.6, 78.8]); LIVE submitted as v7_pv #52630118 at
    μ=1061.8 and climbing. Eight other interventions FALSIFIED
    monotonically (danger3 ×3 κ, FLEET_OVERCOMMIT ×3 mults,
    PRE_REINFORCE ×3 windows, Renaissance Open+Drain+Gang-up all-on
    plus per-mission ablation, HAV-1 binary + soft-floor ×2,
    Holding-tier alone). Architectural finding: v7 + PV is a tight
    local optimum for the K=10 drop-one chooser; pre-discounting
    scoring/proposer signals the rollout already evaluates →
    monotonic regression. Productive next move is architectural —
    portfolio search, deeper K via JAX, opponent ensemble, or
    Bovard IL. Plumbing landed: 3-anchor Wilson gate
    (`scripts/ab_variants.py --candidate --gate-threshold`),
    `PV_GAMMA` ported to JAX (`scripts/kaggle_ab_kernel/run_jax_ab.py`,
    87/87 parity), HAV helpers (`time_to_enemy_threat`,
    `expected_hold`), Mission Renaissance wired into `v7_search`
    behind default-off flags, snipe tier emission framework in
    place, mission proposer chain extended in DEFAULT_LIB_ORDER.
    Postmortem: `audit/2026-05-14-postmortem-research-competition-analysis-2R8I3.md`.
    Frictions: 8 entries under `## 2026-05-13` in audit/friction.md.
    Full pytest 500 passed / 4 skipped at default flags.
    No PI promotions ratified this session.
  - 2026-05-13 — consolidate-fast-simulation-ysd9M (parent, archived
    in this branch's lineage), Phase 3a — scalar-rollout speedups.
    Profile (cProfile, 2000 steps): swept_pair_hit (28%) +
    spawn-step cliff (~20%). Five waves landed: comet-path cache,
    AABB prune in swept_pair_hit, local hoists + module-level math
    aliases, set-based fleet removal. Per-step microbench
    1224→983 µs (1.24×, +20%); fast_sim agent-rollout per-step in
    mid-game ~190 µs. Spawn-crossing turn (step 49, K=10 × 50
    candidates): 5+ s → 155 ms (~32× faster). 62/62 game-parity
    tests green. Knowledge ref:
    `knowledge-base/concepts/pure-python-game-rebuild.md`.


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
  - pv-target-valuation                       # H16: PV_GAMMA=0.99 geometric horizon;
                                              # 32-seed PASS 68.8% Wilson [56.6, 78.8];
                                              # LIVE as v7_pv #52630118 (μ=1061.8 climbing)
  - danger3-allegiance-multiplier             # H17: 3-NN signed count; FALSIFIED at v7
                                              # (κ ∈ {0, 0.1, 0.3} → 50/43.8/37.5%, monotonic)
  - fleet-overcommit                          # H19: ×1.05 / ×1.10 ship inflation; FALSIFIED
                                              # (50/43.8/37.5%, monotonic)
  - pre-reinforce-window                      # H21: ledger-scan follow-up bump; FALSIFIED
                                              # (windows 0/1/3 → 50/43.8/25%, monotonic)
  - drop-comet-targets                        # H15/H18: prune comet targets; 32-seed
                                              # 51.6% Wilson_lo 39.6% INCONCLUSIVE
                                              # (point estimate +18.8pp; gate-just-misses)
  - mission-renaissance-trio                  # H30: opening + drain + gang-up together;
                                              # FALSIFIED 9.4%. Per-mission: gang-up
                                              # 12.5% (catastrophic), drain 41.7%,
                                              # opening 62.5% (borderline)
  - hav-hold-aware-value                      # binary drop 0/16, soft-floor MIN_HOLD=5
                                              # 6.2%, MIN_HOLD=50 18.8%; FALSIFIED — the
                                              # K-rollout already absorbs hold-time signal
  - hav-holding-tier                          # +Mission per (src, target) sized to absorb
                                              # counter-attack; FALSIFIED 12.5% with 11/16
                                              # draws (bigger-incumbent drop-one timed out)

gate_status: cleared                      # 500 pytest pass / 4 skipped at default flags
                                            # (HAV / Renaissance gate flags all default 0;
                                            # bundles byte-equivalent to pre-session)
```
