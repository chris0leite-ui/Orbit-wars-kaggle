# 2026-05-20 — flag: rolling-pair floor is 320 μ below team peak

Live rolling-last-2 (Kaggle auto-keeps these two):
- 52857903 — analytical_wait_N_traj_plus_endgame_play — μ 806.5
- 52854094 — analytical (earlier) — μ 829.1

Team peak EVICTED:
- 52744856 — composite_a2_hybrid (composite head 2P + A2 4P) — μ 1149.2

**Floor lost in 24 h:** ~320 μ, due to five sequential pushes from
`claude/strategy-framework-design-OyoYR-rebased` (the analytical
track) that evicted strong agents from sibling branches without
coordination. This is the friction `cross-agent-push-coordination-gap`
(logged on btjeK 2026-05-21), now addressed by new Rule 42 + the
push claim board in `state/MULTI_BRANCH.md`.

**Recovery options surfaced for PI in HANDOVER.md:**
- Rebundle composite_a2_hybrid lineage (was peak; bundler had silent
  import bug at #52744234 ERROR, needs re-test under Rule 46)
- Rebundle trajectory v4 + wait_N + wallclock (52754310, μ 1143.7)
- Rebundle hold-feasibility solo (52811320, μ 1135.1) — note its
  solo-validation A/B is still pending on btjeK Phase B

**Time pressure:** rolling window cycles every push; weak slot
unrecoverable for ~24h once a 3rd push lands. 34 days to deadline
(2026-06-23). 3 submission slots remaining on 5/20; 5/day budget
resets 00:00 UTC.

PI sign-off required per Rule 42 (any push where evicted-μ >
candidate-μ is BLOCKED).
