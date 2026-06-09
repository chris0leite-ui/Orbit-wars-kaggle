# 2026-05-29 — flag: perf-chain may carry a ~12pp regression

**Status:** suspect, unverified.

The five perf commits on `claude/game-theory-winning-strategy-SEU7P`
(`b4f885d`, `0f1da5b`, `8c6f47c`, `357b52d`, `bdfe9c7`) plus the
H41 floor `9ebd311` lost 5/8 sequential vs the pre-perf bundle
`/tmp/baseline_pv_eta.py` (bundled 2026-05-28 14:17). Three separate
A/B configurations landed at exactly 37.5% — strong indicator the
perf chain itself is the binder, not any one knob.

**What to do before merging any of these commits to main:**

Run a sequential n=16 A/B for each commit individually vs its
immediate predecessor, with bundle provenance stamped in the output.
If any commit drops Wilson-lo below 0.45 vs its predecessor, revert
or surgically reproduce the speedup without the behavioral delta.

Specific candidates to bisect (most likely → least likely):
1. `0f1da5b` KT singleton wiring (state-leak across games is
   easy to miss; the singleton is module-scoped)
2. `bdfe9c7` agent_deadline (truncates rollout queue too aggressively
   under tight timing; verified seed 2 dropped from 831 → 536 ms
   focal_max with floor, indicating chooser termination)
3. `b4f885d` vec (FP rounding on borderline `predict_fleet_fate`
   decisions)
4. `8c6f47c`/`357b52d` WC bump (more candidates validated = more
   variance, not always strictly more lift)

**Watcher:** any future agent on this branch. Until cleared, do
not stack new commits on top of this chain — the next session
should switch to a sibling branch (`btjeK` for Track B work).
