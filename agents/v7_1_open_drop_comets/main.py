"""v7.1 — opening-grab + comet-reject (H11 + H15) on top of v7_0_drop_one.

Identical agent code to v7_0_drop_one. The improvements are wired at
library level:

- H11 (`lib/v7_search.py:_build_incumbent_intents`): `propose_opening_missions`
  joins snipe + reinforce, firing on every owned planet with ships > 8
  during steps 0-5. Front-loaded value `(remaining)^1.5 / (d+1)` directs
  the source's per-source slot to the best neutral capture.
- H15 (`lib/missions/snipe.py`): comet targets where `remaining_lifetime
  <= eta` are rejected at proposer level. The source's runner-up target
  wins the slot instead of consuming it with a score≈0 candidate.

The diagnostic (audit/2026-05-13-v7-0-loss-modes.md) showed 68% of v7_0
losses are opening-stage failures (90% in 4P). H11 directly targets that
gap. H15 is orthogonal but cheap.
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
