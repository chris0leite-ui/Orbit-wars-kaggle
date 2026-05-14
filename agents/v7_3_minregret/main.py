"""v7.3 — min-regret depth-2 chooser over hand-crafted opp archetypes.

The depth-2 work (v7_2) and the H11 opening grab (v7_1) both failed
in scalar A/B against v7_0_drop_one. Both shared the assumption that
in the rollout the opponent plays v3.5.1's pipeline. Hypothesis: the
live ladder is heterogeneous, so v3.5.1-mirror maximin is biased
toward exploiting v3.5.1 specifically.

v7.3 replaces the v3.5.1 drop-one opp candidate set with a fixed
list of 5 hand-crafted opp threat archetypes
(`lib.missions.opp_archetypes`): no-launch, v3.5.1, counter-reinforce,
counter-snipe, cross-attack. Aggregation is **min-regret** by default:
pick the our-action whose worst-case gap from the best response (over
any of the 5 opp archetypes) is smallest. Equivalent maximin variant
is a single-flag switch.

4P games fall back to `choose_4p` (depth-2 game theory is 2P-only).
"""

from __future__ import annotations

from lib.v7_search import choose_archetype_minregret_with_4p


def agent(obs, configuration=None):
    return choose_archetype_minregret_with_4p(
        obs, configuration,
        K=6,
        K_4p=8,
        wallclock_ms=700.0,
        use_min_regret=True,
    )
