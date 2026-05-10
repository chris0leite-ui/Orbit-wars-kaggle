# state/current.md — current submitted agent + tournament rank

> Day-1 agent: leave this empty until the first kernel push. Once
> populated it is updated at every wrap-up (WRAPUP.md step 3).

```yaml
date: 2026-05-10
days_to_deadline: 44                     # 2026-06-23 23:59 UTC minus today
current_submitted_agent: v1.1_orbitfix_arrival_size  # rolling-last-2: [v1 (508.1), v1.1 (597.4)]; staged-for-submit: roi
last_kernel_push: 2026-05-10 09:28:38 UTC
last_submission_id: 52509319             # `kaggle competitions submissions orbit-wars`
last_submission_status: COMPLETE         # validated, μ=597.4
last_submission_file: submissions/v1_orbitfix.py  # bundle of agents/v1_orbitfix/main.py + lib/{geometry,fleet,orbit,intent,mechanism}; mechanism set = [validate, arrival_size, lead_aim]
last_submission_message: |
  v1.1 orbitfix + arrival_size: production-aware sizing for enemy targets
  (lib/mechanism.arrival_size); local 17/20 = 85% vs submitted v1, 100%
  vs shipped baseline.
staged_for_submit: submissions/roi.py    # bundled simple/roi (PI approval pending). E.2 10/10 DONE; bundled-vs-unbundled parity 4/4; 32-seed local 100% (64/64) vs v1_orbitfix.
tournament_rank_today: v1=508.1, v1.1=597.4   # rolling-last-2 active
our_best_rank: μ=597.4 (#52509319, v1.1)
submissions_used_today: 3                # baseline (00:09) + v1 (08:11) + v1.1 (09:28); 2 left
submissions_used_total: 3
plateau_days: 0
saturation_count: 0
session_log:
  - 2026-05-10 — review-competition-handover-0pGNc (cont.): Step 3.5 strategy/mechanism architecture — refactored v1 into Strategy → Intent → realize(mechanisms) pipeline; added 4 mechanisms (validate, arrival_size, lead_aim ALL in DEFAULT; comet_aim + sun_avoid implemented but EXCLUDED based on negative ablation tournaments); third submission v1.1 as ID 52509319 (PI-approved). 111 tests green.
  - 2026-05-10 — simple-trading-strategies-QS0xV: simple-strategy panel (5 target-selection ablations under agents/simple/) + scripts/strategy_panel.py runner + per-strategy markdown docs. 8-seed: roi 96.9% mean panel WR, 100% vs v1. Then Phase 1 meta-strategy infra: replay capture (--capture-replays), lib/fingerprint.py (15 features), scripts/manifold_check.py. 32-seed capture (1568 games, 404 MB): roi 97.1% mean WR, 100% (64/64) vs v1_orbitfix. Manifold gate: RF 80.5% at K=100 (target 90% — NOT cleared; ROI-family is one basin). v1.1 settled at μ=597.4. Bundled submissions/roi.py (E.2 10/10, parity 4/4) staged for PI submission approval.
mechanism_families_explored:
  - heuristic-greedy-nearest-target       # comp-shipped Nearest Planet Sniper; calibration anchor only
  - heuristic-orbit-aware-greedy          # v1: lead-prediction for orbiting non-comet targets + tie-break randomisation (closes A.6)
  - heuristic-orbit-aware-greedy-with-shared-mechanism-layer  # v1.1: above + Strategy/Mechanism split + arrival_size (production-aware sizing)
  - simple-greedy-target-selection-variants  # nearest, production, roi, weakest, enemy_first — ROI dominates panel at 32 seeds
  - meta-strategy-framework-infra-only       # replay capture + behavioural fingerprint + manifold diagnostic; gate not cleared with v1 features (Phase 2 paused on PI choice between coarsen / extend / learned-embedding)
gate_status: cleared                      # E.2 self-play 10/10 for both v1.1 and roi; bundled-vs-unbundled parity 4/4 for roi; 151 tests green
headroom_to_top5pct: ~500-600 μ           # public top μ=1224, top-5% threshold ≈1100. v1.1 at μ=597.4 → ~500 μ from top-5%. ROI predicted +200-500 μ vs v1 (so live μ probably 700-1000) — could close ~half the gap in one submit.
```
