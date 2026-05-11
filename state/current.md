# state/current.md — current submitted agent + tournament rank

> Day-1 agent: leave this empty until the first kernel push. Once
> populated it is updated at every wrap-up (WRAPUP.md step 3).

```yaml
date: 2026-05-11
days_to_deadline: 43                     # 2026-06-23 23:59 UTC minus today
current_submitted_agent: v2              # rolling-last-2: [v1.2/roi (996.5), v2 (PENDING)]; v1.1 EVICTED
last_kernel_push: 2026-05-11 04:04:07 UTC
last_submission_id: 52532938             # `kaggle competitions submissions orbit-wars`
last_submission_status: PENDING          # validation episode running
last_submission_file: submissions/v2.py  # bundle of agents/v2/main.py + lib/{geometry,fleet,orbit,aim,combat,world_model,intent,mechanism}; targeting = roi + WorldModel.owner_at predictive dedup; mechanism set = [validate, arrival_size, lead_aim_v2, sun_avoid, path_clears_other_planets, oob_guard]
last_submission_message: |
  v2: WorldModel-aware roi (skip targets predicted ours at arrival; per-source
  re-pick). Block A physics upgrade (5-iter aim+search_safe_intercept,
  sun-safe arrival-aware, path_clears_other_planets, oob_guard) + Block D
  worldmodel dedup. 64% mean panel WR, 69% h2h vs frozen v1.2-equiv,
  86% vs broader panel. audit/2026-05-10-block-c-d-arrival-ledger-and-v2.md
tournament_rank_today: v1.2/roi=996.5, v2=PENDING   # rolling-last-2 active; v1.1 (565.7) evicted by v2 submit
our_best_rank: μ=996.5 (#52518060, v1.2/roi) — pending v2 validation
lb_top10_cliff: 1460.0                   # sash, 2026-05-10 PM. #1 = bowwowforeach 1663.4
submissions_used_today: 1                # v2 (04:04 UTC, 2026-05-11)
submissions_used_total: 5
plateau_days: 0
saturation_count: 0
session_log:
  - 2026-05-10 — review-competition-handover-0pGNc (cont.): Step 3.5 strategy/mechanism architecture — refactored v1 into Strategy → Intent → realize(mechanisms) pipeline; added 4 mechanisms (validate, arrival_size, lead_aim ALL in DEFAULT; comet_aim + sun_avoid implemented but EXCLUDED based on negative ablation tournaments); third submission v1.1 as ID 52509319 (PI-approved). 111 tests green.
  - 2026-05-10 — simple-trading-strategies-QS0xV: simple-strategy panel (5 target-selection ablations) + Phase 1 meta-strategy infra (replay capture, lib/fingerprint, scripts/manifold_check). 32-seed: roi 97.1% mean WR, 100% (64/64) vs v1_orbitfix. Manifold gate ❌ (RF 80.5% K=100; ROI-family is one basin). v1.1 settled at μ=597.4. Fourth submission roi as ID 52518060 (PI-approved on retry after Kaggle 503).
  - 2026-05-10 PM — competition-strategy-brainstorm-ZK6XT: strategic-direction plan written + PI-ratified (top-10 prize target, adapt-Roman architecture). Capture-success probe shows reached 77.2% / collided_other 10.7% / oob 7.6% / sun 2.1% — punch #7 demoted, path-clears-other-planets + OOB guard promoted. Public-kernel teardown (Roman 1224 + Pilkwang structured-baseline + sigmaborov physics-accurate-planner + sun-dodging) confirms architecture decision; 4P matters (Roman's FOUR_PLAYER_ROTATING_* constants). v1.2/roi μ settled at 978.7 (below the evening 1105 read). Top-10 cliff = 1460. Audits day-1 PM phase 1 deliverables - audit/2026-05-10-capture-success-probe.{json,md}, audit/2026-05-10-public-kernel-teardown.md.
  - 2026-05-10 PM Block A — physics-module upgrade landed: lib/aim.py (5-iter aim + search_safe_intercept), upgraded sun_avoid (arrival-aware), added path_clears_other_planets + oob_guard. Capture probe reached 77.2% → 83.9% (+6.7pp); collided_other 10.7% → 5.2%. A/B: roi (new) vs roi_baseline (frozen pre-physics) = 56% (36/64) — clears h2h gate, marginal on panel gate. Bundle staged at submissions/v1_3_roi_physics.py (held). Audit: audit/2026-05-10-block-a-physics-upgrade.md.
  - 2026-05-10 PM Blocks C+D — arrival-ledger substrate + v2 strategy: lib/combat.py, lib/world_model.py, agents/v2/main.py (roi + WorldModel-aware dedup). 21 new tests pass (181 total). A/B v2 vs (roi, roi_baseline) = 64.1% mean panel WR, 69% h2h vs roi_baseline (44/64). Both gates clear. Bundle staged at submissions/v2.py (50 KB, self-play 5/5 DONE, p95 turn 3.7 ms). Audit: audit/2026-05-10-block-c-d-arrival-ledger-and-v2.md.
mechanism_families_explored:
  - heuristic-greedy-nearest-target       # comp-shipped Nearest Planet Sniper; calibration anchor only
  - heuristic-orbit-aware-greedy          # v1: lead-prediction for orbiting non-comet targets + tie-break randomisation (closes A.6)
  - heuristic-orbit-aware-greedy-with-shared-mechanism-layer  # v1.1: above + Strategy/Mechanism split + arrival_size (production-aware sizing)
  - simple-greedy-target-selection-variants  # nearest, production, roi, weakest, enemy_first — ROI dominates panel at 32 seeds; v1.2 = roi submitted as #52518060
  - meta-strategy-framework-infra-only       # replay capture + behavioural fingerprint + manifold diagnostic; gate not cleared with v1 features (Phase 2 paused on PI choice between coarsen / extend / learned-embedding)
  - heuristic-physics-upgrade               # v1.3: 5-iter aim_with_prediction + search_safe_intercept fallback + sun-safe arrival-aware + path-clears-other-planets + oob_guard. Probe reached 77.2% → 83.9%. 56% h2h vs frozen pre-physics ROI. Bundle: submissions/v1_3_roi_physics.py. NOT YET SUBMITTED.
  - heuristic-worldmodel-aware              # v2: roi target selection + WorldModel.owner_at predictive dedup (don't double-commit). 64.1% mean panel WR, 69% h2h vs pre-physics ROI. Bundle: submissions/v2.py. NOT YET SUBMITTED. PRIMARY CANDIDATE for next submit.
gate_status: cleared                      # E.2 self-play 10/10 for v1.2/roi; bundled-vs-unbundled parity 4/4; 151 tests green
headroom_to_top5pct: -120 μ               # top-5% threshold ≈ 1100; us at 978.7. **No longer the binding target.**
headroom_to_top10_prize: +481 μ           # top-10 cliff at 1460 (sash). PRIZE TARGET. v1.2/roi at 978.7 → +481 μ to close.
headroom_to_roman_public: +246 μ          # Roman published 1224; we are 246 μ below his ceiling. Block A-D goal: close this. Block F goal: exceed it.
```
