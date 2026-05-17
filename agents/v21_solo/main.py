"""v21_solo — Fix 1: cap commits at 1 per turn (no dogpile).

Keeps v21's full filtering stack (A + E1 + E2) but forces single-commit
emit. Diagnostic per audit/2026-05-17-v21-diag-roll-up: v21 over-commits
2-4 captures per turn whose joint Δ looks positive in rollout but
realised value is negative under live opp counter-recapture (lost-back
+62 vs v15-self on seed 1002). Single-commit = v15's emit policy with
v21's better candidate filtering on top.

If this clears the h2h gate vs v15 (Wlo > 0.55), Patch A's joint-emit
was the regression. If not, the issue is elsewhere (E1/E2 or interaction).
"""
from __future__ import annotations

import agents.v21.main as _v21

# MAX_COMMIT_ROUNDS = 0 → exactly 1 commit per turn. The loop's
# `len(committed) >= MAX_COMMIT_ROUNDS + 1` check breaks after the
# first append. The joint-rescore branch is skipped entirely
# (no remaining commits to score against). E1 prefilter and E2
# hold-check still apply.
_v21.MAX_COMMIT_ROUNDS = 0

agent = _v21.agent
_INSTRUMENT_COUNTERS = _v21._INSTRUMENT_COUNTERS
