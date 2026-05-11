# audit/INDEX.md

> One line per dated audit. Updated by WRAPUP step 4b.

- `audit/2026-05-10-day-1-data-inventory.md` — Day-1 bootstrap on branch `claude/orbit-wars-bootstrap-irewT`: comp registration verified; comp-context TBDs filled from rules+evaluation pages; data/ inventory; shipped baseline rolled out 6/6 vs random and 6 self-play (P1 wins 4/6 — asymmetry logged as ISSUES.md A.6); orbit-prediction math verified with off-by-one finding on the absolute formula; first submission shipped (ID 52497828, calibration probe = shipped baseline).
- `audit/2026-05-10-day-1-rollouts.json` — raw rewards/ship-counts JSON dumped by `scripts/run_day1_rollouts.py`.
- `audit/2026-05-10-postmortem-orbit-wars-bootstrap-irewT.md` — Day-1 postmortem on the bootstrap branch: one bad-decision flag (audit-date drift), one promotion to `improvements.md` (`audit-date-must-track-system-currentdate`, PI-ratified), no PI additions.
- `audit/2026-05-11-v3-snipe-critical-review.md` — Critical review of submission 52544634 (branch `claude/analyze-submission-logs-dFHeS`): live μ=1055.5 (+90.2 over v2) but absolute winrate dropped 50.9% → 41.2% (TrueSkill matched against stronger opponents). Bounce rate doubled (7.6% → 14.7%) — under-sized fleets are the dominant error mode, not physics. Parallel branch `claude/precision-physics-engine-ymJkA` adds a deterministic intercept solver (`agents/precision/`) that beats v3_snipe on seed 42. Diagnostic: `scripts/episode_postmortem.py` (new).
