# state/current.md — current submitted agent + tournament rank

> Day-1 agent: leave this empty until the first kernel push. Once
> populated it is updated at every wrap-up (WRAPUP.md step 3).

```yaml
date: 2026-05-10
days_to_deadline: 44                     # 2026-06-23 23:59 UTC minus today
current_submitted_agent: v1.2_simple_roi # rolling-last-2: [v1.1 (597.4), v1.2/roi (PENDING)]; v1 EVICTED
last_kernel_push: 2026-05-10 14:59:32 UTC
last_submission_id: 52518060             # `kaggle competitions submissions orbit-wars`
last_submission_status: PENDING          # validation episode running
last_submission_file: submissions/roi.py # bundle of agents/simple/roi.py + lib/{geometry,fleet,orbit,intent,mechanism}; targeting = argmax production/distance, mechanism set = [validate, arrival_size, lead_aim]
last_submission_message: |
  v1.2 simple/roi: production/distance ROI targeting; same DEFAULT_MECHANISMS
  as v1.1; 32-seed local 100% (64/64) vs v1_orbitfix; 97.1% mean panel WR.
  audit/2026-05-10-phase1-manifold-verdict.md
tournament_rank_today: v1.1=597.4, v1.2/roi=PENDING   # rolling-last-2 active; v1 (568.0) evicted by this submit
our_best_rank: μ=597.4 (#52509319, v1.1)
submissions_used_today: 4                # baseline (00:09) + v1 (08:11) + v1.1 (09:28) + roi (14:59); 1 slot left
submissions_used_total: 4
plateau_days: 0
saturation_count: 0
session_log:
  - 2026-05-10 — review-competition-handover-0pGNc (cont.): Step 3.5 strategy/mechanism architecture — refactored v1 into Strategy → Intent → realize(mechanisms) pipeline; added 4 mechanisms (validate, arrival_size, lead_aim ALL in DEFAULT; comet_aim + sun_avoid implemented but EXCLUDED based on negative ablation tournaments); third submission v1.1 as ID 52509319 (PI-approved). 111 tests green.
  - 2026-05-10 — simple-trading-strategies-QS0xV: simple-strategy panel (5 target-selection ablations) + Phase 1 meta-strategy infra (replay capture, lib/fingerprint, scripts/manifold_check). 32-seed: roi 97.1% mean WR, 100% (64/64) vs v1_orbitfix. Manifold gate ❌ (RF 80.5% K=100; ROI-family is one basin). v1.1 settled at μ=597.4. Fourth submission roi as ID 52518060 (PI-approved on retry after Kaggle 503).
mechanism_families_explored:
  - heuristic-greedy-nearest-target       # comp-shipped Nearest Planet Sniper; calibration anchor only
  - heuristic-orbit-aware-greedy          # v1: lead-prediction for orbiting non-comet targets + tie-break randomisation (closes A.6)
  - heuristic-orbit-aware-greedy-with-shared-mechanism-layer  # v1.1: above + Strategy/Mechanism split + arrival_size (production-aware sizing)
  - simple-greedy-target-selection-variants  # nearest, production, roi, weakest, enemy_first — ROI dominates panel at 32 seeds; v1.2 = roi submitted as #52518060
  - meta-strategy-framework-infra-only       # replay capture + behavioural fingerprint + manifold diagnostic; gate not cleared with v1 features (Phase 2 paused on PI choice between coarsen / extend / learned-embedding)
gate_status: cleared                      # E.2 self-play 10/10 for v1.2/roi; bundled-vs-unbundled parity 4/4; 151 tests green
headroom_to_top5pct: ~500 μ               # public top μ=1224, top-5% threshold ≈1100. v1.1 at μ=597.4 → ~500 μ from top-5%. ROI predicted +200-500 μ vs v1 — landing μ TBD post-validation; could close ~half the gap.
```
