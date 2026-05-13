"""v7.0 — drop-one enumerator under the new fast_sim scorer.

Re-runs the v3_lookahead MVP architecture (drop-one candidate set)
but with `lib/fast_sim` (183× faster than env.clone) and
`lib/opp_model.top_tier_mirror_policy` (v3.5.1 mirror, better
opponent proxy than v3_snipe). Measures the FRAMEWORK's contribution
alone; lift here would mean the previous Phase 2 result was budget-
limited, not enumeration-limited.

Fallback: if no candidate scores strictly above the incumbent (or the
watchdog trips), returns v3.5.1's action — parity floor preserved.
"""

from __future__ import annotations

from lib.v7_search import choose


def agent(obs, configuration=None):
    return choose(
        obs, configuration,
        enumerator_mode="drop_one",
        K=10,
        wallclock_ms=700.0,
    )
