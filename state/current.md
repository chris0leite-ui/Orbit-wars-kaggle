# state/current.md — current submitted agent + tournament rank

> Day-1 agent: leave this empty until the first kernel push. Once
> populated it is updated at every wrap-up (WRAPUP.md step 3).

```yaml
date: 2026-05-11
days_to_deadline: 43                     # 2026-06-23 23:59 UTC minus today
current_submitted_agent: v3_snipe        # rolling-last-2: [v2 (974.3), v3_snipe (PENDING)]; v1.2/roi (1006.9) evicted by v3 push
last_kernel_push: 2026-05-11 12:16:01 UTC
last_submission_id: 52544634
last_submission_status: PENDING          # validation episode running; kaggle CLI 503 at wrap time, will re-poll
last_submission_file: submissions/v3_snipe.py  # 61.7 KB bundle of agents/v3_snipe/main.py + lib/{geometry,fleet,orbit,aim,combat,world_model,intent,trajectory,mechanism,mission,missions/snipe,missions/reinforce,planner}. Strategy = snipe + reinforce mission classes through settle_plan (same-turn arrival ledger). Mechanism stack = DEFAULT_MECHANISMS with full-trajectory predict_fleet_fate guards.
last_submission_message: |
  v3_snipe: Block E missions (snipe + reinforce) + cost-aware ROI +
  comet-lifetime + same-turn ledger + full-trajectory ray-cast guards.
  Capture-probe reached 77.2% (pre-fix) → 93.0% (trajectory) → 97.2%
  (this build). 32-seed 2P vs v2 = 57.8% (Wilson [45.6, 69.2]);
  16-seed 4P FFA parity. audit/2026-05-11-v3-lookahead-mvp-parity.md
  + tournaments/20260511T112936Z.json
tournament_rank_today: v2=974.3, v3_snipe=PENDING   # v1.2/roi (1006.9) evicted by v3 push; v2 dropping from 1025.5 → 974.3 over day
our_best_rank: μ=974.3 (#52532938, v2) — pending v3 validation
lb_top10_cliff: 1447.6                   # ShunkiKyoya, 2026-05-11. #1 = bowwowforeach 1697.7
submissions_used_today: 2                # v2 (04:04 UTC) + v3_snipe (12:16 UTC)
submissions_used_total: 6
plateau_days: 0
saturation_count: 0
session_log:
  - 2026-05-11 — bootstrap-agentic-systems-lqnm6 (THIS SESSION). Major work waves: (a) bootstrap infra — workers default = cpu_count(), 4P FFA local fixture (scripts/{_agent_paths,ffa_tournament,ffa_panel}.py), scripts/live_episode_summary.py; (b) Block E mission framework MVP (lib/{mission,planner,missions/snipe}.py + agents/v3_snipe — refactor parity with v2 at 32-seed); (c) lookahead probes — Phase 1a oracle gap = 32pp at step 50, Phase 1b one-turn action injection FALSIFIED, Phase 2 env.clone() Sim<K=50> AUC = 0.952 (≈ oracle); (d) v3.1 lookahead MVP with drop-one candidates — 50/50 parity vs v2 at 32 seeds (8-seed 68.8% was upward noise); (e) live fleet-loss fix — lib/trajectory.predict_fleet_fate ray-casts the FULL flight, replaces endpoint-only checks; capture probe reached 77.2% → 93.0% → 97.2% across 3 fix waves; bundler bug fix (DEFAULT_LIB_ORDER missing trajectory + missions); (f) ROI cost-awareness across snipe/simple-roi/v2 (additive cost in denominator, not pure value/cost); (g) the four remaining ROI-doc shortcomings: comet lifetime, reinforce mission class, same-turn ledger in settle_plan, mission classification used in v3_snipe; (h) 32-seed 2P vs v2 verification = 57.8% (Wilson lo 45.6%), 16-seed 4P FFA parity; (i) PI-approved v3_snipe submit as #52544634. Total: 14 commits on this branch.
  - 2026-05-10 PM — competition-strategy-brainstorm-ZK6XT (prior): strategic-direction plan + Block A physics + Blocks C+D arrival-ledger + v2 strategy. v1.2/roi μ settled at 978.7. Top-10 cliff = 1460. (Older entries archived to audit/archive-2026-05-10-handover-prior-pm-sessions.md.)
mechanism_families_explored:
  - heuristic-greedy-nearest-target       # comp-shipped Nearest Planet Sniper; calibration anchor only
  - heuristic-orbit-aware-greedy          # v1: lead-prediction for orbiting non-comet targets + tie-break randomisation (closes A.6)
  - heuristic-orbit-aware-greedy-with-shared-mechanism-layer  # v1.1: above + Strategy/Mechanism split + arrival_size (production-aware sizing)
  - simple-greedy-target-selection-variants  # nearest, production, roi, weakest, enemy_first — ROI dominates panel at 32 seeds; v1.2 = roi submitted as #52518060
  - meta-strategy-framework-infra-only       # replay capture + behavioural fingerprint + manifold diagnostic; gate not cleared with v1 features
  - heuristic-physics-upgrade               # v1.3: 5-iter aim_with_prediction + search_safe_intercept fallback + sun-safe arrival-aware + path-clears-other-planets + oob_guard
  - heuristic-worldmodel-aware              # v2: roi target selection + WorldModel.owner_at predictive dedup; submitted as #52532938
  - mission-framework-snipe-only            # v3.0 (Block E MVP): Mission dataclass + propose_snipe_missions + settle_plan; bit-for-bit parity with v2 (32/32 draws at step 500)
  - env-clone-forward-sim-scorer            # Sim<K> scoring head via env.clone() + step; AUC 0.952 at K=50 from probe step 50 ≈ perfect oracle (audit/2026-05-11-lookahead-phase2-forward-sim.md)
  - lookahead-drop-one-candidates           # v3.1 v3_lookahead: env.clone() forward sim over drop-one candidate set; 32-seed 50/50 vs v2 — framework works, drop-one too narrow
  - full-trajectory-predict-fleet-fate      # lib/trajectory.py replaces endpoint-only guards with full-flight ray-cast; capture probe reached 77.2% → 97.2% across the 3 fix waves of 2026-05-11
  - cost-aware-roi-additive-denominator     # score = (production × time_to_hold) / (ships_to_send + distance + 1). Additive cost avoids pure-value/cost over-correction toward 1-ship targets
  - comet-lifetime-correction                # comet_remaining_lifetime helper; time_to_hold caps at len(path) - path_index for comet targets
  - mission-framework-snipe-plus-reinforce  # v3.1 v3_snipe: snipe + reinforce mission classes through settle_plan (with same-turn arrival ledger). 32-seed 2P 57.8% vs v2 (Wilson lo 45.6%); 16-seed 4P FFA parity. Submitted as #52544634.
gate_status: cleared                      # 228/228 tests green; bundle E.2 self-play 0/10 crashes; bundle-vs-unbundled parity 10/10; 32-seed 2P + 16-seed 4P panels run
headroom_to_top5pct: deprecated            # no longer the binding target — top-10 prize cliff is
headroom_to_top10_prize: +473 μ           # top-10 cliff at 1447.6 (ShunkiKyoya, 2026-05-11). v2 at 974.3 → +473μ. v3 PENDING.
headroom_to_roman_public: +250 μ          # Roman published 1224; we are 250 μ below his ceiling. Block E missions narrow this; v3.1 candidate enumerator for lookahead is the next ceiling-raiser.
```
