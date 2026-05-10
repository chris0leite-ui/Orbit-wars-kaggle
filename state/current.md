# state/current.md — current submitted agent + tournament rank

> Day-1 agent: leave this empty until the first kernel push. Once
> populated it is updated at every wrap-up (WRAPUP.md step 3).

```yaml
date: 2026-05-10
days_to_deadline: 44                     # 2026-06-23 23:59 UTC minus today
current_submitted_agent: v1.1_orbitfix_arrival_size  # rolling-last-2: [v1 (508.1), v1.1 (PENDING)]; baseline EVICTED
last_kernel_push: 2026-05-10 09:28:38 UTC
last_submission_id: 52509319             # `kaggle competitions submissions orbit-wars`
last_submission_status: PENDING          # validation episode running
last_submission_file: submissions/v1_orbitfix.py  # bundle of agents/v1_orbitfix/main.py + lib/{geometry,fleet,orbit,intent,mechanism}; mechanism set = [validate, arrival_size, lead_aim]
last_submission_message: |
  v1.1 orbitfix + arrival_size: production-aware sizing for enemy targets
  (lib/mechanism.arrival_size); local 17/20 = 85% vs submitted v1, 100%
  vs shipped baseline.
tournament_rank_today: v1=508.1, v1.1=PENDING   # rolling-last-2 active
our_best_rank: μ=508.1 (#52507539, v1)
submissions_used_today: 3                # baseline (00:09) + v1 (08:11) + v1.1 (09:28)
submissions_used_total: 3
plateau_days: 0
saturation_count: 0
session_log:
  - 2026-05-10 — bootstrap branch (claude/orbit-wars-bootstrap-irewT): seed import + day-1 discovery (a)–(g) + first submission (shipped baseline as calibration probe).
  - 2026-05-10 — review-competition-handover-0pGNc: public-research phase (10 top kernels into external/kernels/), built D.1 tournament fixture (54 tests), lib/{orbit,fleet,geometry} primitives with TDD, agents/v1_orbitfix + bundler + strategy docs, second submission v1_orbitfix as ID 52507539 (μ=508.1).
  - 2026-05-10 — review-competition-handover-0pGNc (cont.): Step 3.5 strategy/mechanism architecture — refactored v1 into Strategy → Intent → realize(mechanisms) pipeline; added 4 mechanisms (validate, arrival_size, lead_aim ALL in DEFAULT; comet_aim + sun_avoid implemented but EXCLUDED based on negative ablation tournaments); third submission v1.1 as ID 52509319 (PI-approved). 111 tests green.
mechanism_families_explored:
  - heuristic-greedy-nearest-target       # comp-shipped Nearest Planet Sniper; calibration anchor only
  - heuristic-orbit-aware-greedy          # v1: lead-prediction for orbiting non-comet targets + tie-break randomisation (closes A.6)
  - heuristic-orbit-aware-greedy-with-shared-mechanism-layer  # v1.1: above + Strategy/Mechanism split + arrival_size (production-aware sizing)
gate_status: cleared                      # pre-baseline gate artifacts in audit/2026-05-10-day-1-data-inventory.md; D.1 fixture green; bundler E.2 self-vs-self 10/10 DONE; v1.1 17/20 = 85% vs v1
headroom_to_top5pct: TBD                  # public top μ=1224 (Roman); top-5% threshold ≈ μ 1100-1200 (≈ top-120 of 2413). v1 at μ=508 sits roughly mid-pack pending v1.1 settle.
```
