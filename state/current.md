# state/current.md — current submitted agent + tournament rank

> Day-1 agent: leave this empty until the first kernel push. Once
> populated it is updated at every wrap-up (WRAPUP.md step 3).

```yaml
date: 2026-05-10
days_to_deadline: 44                     # 2026-06-23 23:59 UTC minus today
current_submitted_agent: shipped-baseline-nearest-planet-sniper
last_kernel_push: 2026-05-10 00:09:54 UTC
last_submission_id: 52497828             # `kaggle competitions submissions orbit-wars`
last_submission_status: PENDING          # validation episode (self-vs-self) running
last_submission_file: data/main.py       # comp-shipped, unmodified
last_submission_message: |
  Day-1 calibration: comp-shipped Nearest Planet Sniper baseline
  (unmodified) — used to anchor mu-rating before any agent variant
tournament_rank_today: TBD               # read after validation passes via `kaggle competitions leaderboard orbit-wars -s`
our_best_rank: TBD
submissions_used_today: 1                # first submit on 2026-05-10 UTC
submissions_used_total: 1
plateau_days: 0
saturation_count: 0
session_log:
  - 2026-05-10 — bootstrap branch (claude/orbit-wars-bootstrap-irewT): seed import + day-1 discovery (a)–(g) + first submission (shipped baseline as calibration probe).
mechanism_families_explored:
  - heuristic-greedy-nearest-target       # comp-shipped Nearest Planet Sniper; calibration anchor only
gate_status: pending_pi_signoff           # pre-baseline gate artifacts present in audit/2026-05-10-day-1-data-inventory.md
headroom_to_top5pct: TBD
```
