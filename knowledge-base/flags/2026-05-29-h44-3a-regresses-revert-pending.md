# Flag — 2026-05-29 — H44 Phase 3a wait_N filter cherry-pick regresses; revert pending

**Tag:** `h44-phase-3a-wait-n-filter-regresses-load-bearing-bypass`

**Commit:** `8b20b6d` on `claude/game-theory-winning-strategy-SEU7P`
("fix(proposer): close H44 wait_N filter gap (Phase 3a)" —
cherry-picked from `extract-physics-trajectory-Vjaz9`).

**Evidence:** subprocess-isolated A/B at n=32 vs same-bundle no-filter:
13/32 = 40.6%, Wilson [0.255, 0.577]. Per-seat both regress (P0=43.8%,
P1=37.5%). The "wait_N candidates would mis-classify" bypass at
`agents/baseline/proposer.py:993-1012` was load-bearing — closing it
cuts useful proposals without commensurate physics-waste-prevention
gain. Underlying H44 "physics-waste" premise was retracted by PI on
2026-05-29 (`92371dc`: fleets cannot be destroyed in flight).

**Persistent risk if not addressed:**

- This commit was pushed (PI's stop-hook flagged it earlier). Anyone
  bundling baseline from this branch's HEAD will ship the regression.
- The commit is on this branch only; sibling branches (`btjeK`,
  `hqNVM`, `extract-physics-trajectory-Vjaz9`) are not contaminated
  by it, but if anyone rebases from this branch they pick it up.

**Recommended action next session:** `git revert 8b20b6d` on this
branch. Do not cherry-pick Phase 3b (`25589ad`); same axis just
falsified.

**Why this is a flag and not just friction:** the regression survives
in the repo. Friction documents what happened; the flag is the
durable warning to future sessions.
