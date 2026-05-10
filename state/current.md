# state/current.md — current submitted agent + tournament rank

> Day-1 agent: leave this empty until the first kernel push. Once
> populated it is updated at every wrap-up (WRAPUP.md step 3).

```yaml
date: 2026-05-10
days_to_deadline: 44                     # 2026-06-23 23:59 UTC minus today
current_submitted_agent: v1_orbitfix     # rolling-last-2 also retains shipped baseline (52497828)
last_kernel_push: 2026-05-10 08:11:27 UTC
last_submission_id: 52507539             # `kaggle competitions submissions orbit-wars`
last_submission_status: PENDING          # validation episode running
last_submission_file: submissions/v1_orbitfix.py  # bundle of agents/v1_orbitfix/main.py + lib/{geometry,fleet,orbit}
last_submission_message: |
  v1 orbitfix: orbit-aware aim (lead-prediction for orbiting non-comet
  targets) + tie-break randomisation; local 40/40 vs shipped baseline
tournament_rank_today: TBD               # read after validation passes via `kaggle competitions leaderboard orbit-wars -s`
our_best_rank: TBD
submissions_used_today: 2                # baseline (00:09 UTC) + v1 (08:11 UTC)
submissions_used_total: 2
plateau_days: 0
saturation_count: 0
session_log:
  - 2026-05-10 — bootstrap branch (claude/orbit-wars-bootstrap-irewT): seed import + day-1 discovery (a)–(g) + first submission (shipped baseline as calibration probe).
  - 2026-05-10 — review-competition-handover-0pGNc: public-research phase (10 top kernels into external/kernels/), built D.1 tournament fixture (54 tests), lib/{orbit,fleet,geometry} primitives with TDD, agents/v1_orbitfix + bundler + strategy docs, second submission v1_orbitfix as ID 52507539 (PI-approved).
mechanism_families_explored:
  - heuristic-greedy-nearest-target       # comp-shipped Nearest Planet Sniper; calibration anchor only
  - heuristic-orbit-aware-greedy          # v1: lead-prediction for orbiting non-comet targets + tie-break randomisation (closes A.6)
gate_status: cleared                      # pre-baseline gate artifacts in audit/2026-05-10-day-1-data-inventory.md; D.1 fixture green; bundler E.2 self-vs-self 10/10 DONE
headroom_to_top5pct: TBD                  # public top μ=1224 (Roman); top-5% threshold ≈ μ 1100-1200 (≈ top-120 of 2413)
```
