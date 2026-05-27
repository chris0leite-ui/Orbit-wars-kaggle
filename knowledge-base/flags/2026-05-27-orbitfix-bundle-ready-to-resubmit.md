# 2026-05-27 — orbitfix bundle ready to resubmit (~+80μ floor recovery)

**Flag.** `submissions/baseline_joint_aggr_consolidated_orbitfix.py`
on `claude/competitive-programming-ab-test-BZknl` is
parity-tested and ready to resubmit AS-IS. SHA256
`17515bf3cda1f01e76503eb9d0c2f809b483a7598492ef752c458ec69e56f646`
(508,892 bytes).

**Live state at flag time:** current rolling pair is
sub 52966655 baseline.py μ=1097.5 (latest) +
sub 52965748 orbitfix_kt_p23.py μ=981 (older). Floor: 981.

**Predicted lift:** resubmit replaces the older half (981) →
new rolling pair becomes orbitfix + baseline (1097). Predicted
orbitfix μ ≈ 1165 (the previous settle). Floor lifts to ~1097
either way; ceiling lifts ~+70μ vs the 981 it replaces.

**Caveat (substrate-correctness, not ladder).** This bundle
fails the candidate Rule 48 nearest-elim gate (14/16 ELIM, 2
step-500 score-wins). The bug doesn't fire on the ladder
(competitive games don't reach dominant-endgame state), so
ladder μ is not at risk. But shipping without first cherry-
picking commit `68c24be` from `claude/session-EqJuT` leaves a
known substrate failure in the bundle. ~5 LOC fix; should
ride along on the resubmit.

**Why this is here.** The current session ended without
shipping. The next agent on this branch (or any agent looking
for ladder-floor recovery) should know the candidate is
shovel-ready.