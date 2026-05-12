"""v7.4 — v7.2 + 4P-aware drop-one rollout.

In 2P, behaves as v7.2 (σ-equiv + symmetric + maximin + recapture).
In 4P, runs a drop-one candidate set against a 4-seat Snapshot
rollout where the three opponents play `top_tier_mirror_policy`.
Scoring head: `ours − max(other seat ships)` — rewards keeping the
lead vs the best-remaining-opponent (better 4P first-place proxy
than ships-sum-delta).

Previous versions of v7 returned the v3.5.1 incumbent verbatim in
4P (because the 2-seat Snapshot can't represent 4P state). This
agent uses `lib/fast_sim`'s native num_seats=4 support to extend
the lookahead into the missing half of the ladder.
"""

from __future__ import annotations

from lib.v7_search import choose_with_4p


def agent(obs, configuration=None):
    return choose_with_4p(
        obs, configuration,
        K_2p=10,
        K_4p=8,
        wallclock_ms=700.0,
        use_symmetric=True,
        include_recapture=True,
    )
