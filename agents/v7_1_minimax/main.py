"""v7.1 — σ-equiv + symmetric scoring + maximin overlay.

Stacks the σ-equiv layer (sym_hypot + planner _tb + score rounding —
ported from `origin/claude/game-theory-strategy-analysis-0oH4N`,
proved load-bearing for v7_minimax's μ=1063 result) on top of v7_0's
fast_sim-based candidate evaluation, plus a real maximin overlay over
an M=2 opp class.

Per turn:
1. Build v3.5.1 incumbent (with σ-equiv tie-break already in planner).
2. Enumerate N+1 our candidates: incumbent + each drop-one variant.
3. Enumerate M=2 opp candidates: opp's v3.5.1 + drop-smallest.
4. Score every (our_i, opp_j) cell via `score_joint_symmetric`
   (seat-flipped average; cancels env P1-bias).
5. Pick i* = argmax_i min_j P[i,j]; tie → row 0 (incumbent).

4P → falls back to v3.5.1 (maximin doesn't extend cleanly to n>2).
"""

from __future__ import annotations

from lib.v7_search import choose_maximin


def agent(obs, configuration=None):
    return choose_maximin(
        obs, configuration,
        K=10,
        wallclock_ms=700.0,
        use_symmetric=True,
    )
