"""v7.2 — depth-2 maximin chooser on top of v7_1's incumbent.

Outer enumerate: our drop-one candidate set (≤ 8 candidates).
For each our candidate i:
  1. Step the snapshot one turn with [our_i, opp_initial_incumbent].
  2. From the post-step state, enumerate opp's drop-one set (≤ 4).
  3. For each opp candidate j, force it on turn 2 (we pass), then
     rollout K-2 mirror-mirror steps. Score with delta_us_minus_them.
Maximin: argmax_i min_j payoff[i][j]. Tie → row 0 (incumbent).

Built on top of the H11+H15 wire — the incumbent already includes the
opening proposer + comet reject (audit/2026-05-13-v7-0-loss-modes.md
showed 68% opening-bucket dominance). The depth-2 oracle showed 14.8%
move disagreement with v7_0's single-ply chooser — moderate but real
head-room from explicit opp-response anticipation.

4P → choose_4p fallback (no Nash maximin at n > 2).
"""

from __future__ import annotations

from lib.v7_search import choose_depth2_with_4p


def agent(obs, configuration=None):
    return choose_depth2_with_4p(
        obs, configuration,
        K_2p=6,
        K_4p=8,
        wallclock_ms=700.0,
    )
