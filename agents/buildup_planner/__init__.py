"""buildup_planner — phased agent: BUILDUP → CONSOLIDATION → STRIKE.

Step 1 (this commit) wires BUILDUP + CONSOLIDATION. Predicate and STRIKE
are stubbed; see /root/.claude/plans/yes-start-with-the-lexical-codd.md.
"""
from agents.buildup_planner.main import agent

__all__ = ["agent"]
