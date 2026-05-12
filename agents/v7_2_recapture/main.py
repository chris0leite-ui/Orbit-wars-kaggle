"""v7.2 — v7.1 + recapture mission class.

Adds `propose_recapture_missions` to the incumbent's mission set
alongside snipe + reinforce. Recapture targets planets we lost in
the last 50 turns with a time-decaying RECAPTURE_BONUS.

This was previously regressed at -14pp when wired into v3_snipe
(audit/2026-05-12-recapture-wireup-ab.md); the regression was traced
to (1) score-scale mismatch and (2) per-turn proposal-volume dilution.
Both fixed via the constants `RECAPTURE_SCORE_DENOM_MATCHES_SNIPE=1`
and `RECAPTURE_TOPK_PER_TURN=5` in lib/missions/recapture.py.

The maximin layer is unchanged from v7.1.
"""

from __future__ import annotations

from lib.v7_search import choose_maximin


def agent(obs, configuration=None):
    return choose_maximin(
        obs, configuration,
        K=10,
        wallclock_ms=700.0,
        use_symmetric=True,
        include_recapture=True,
    )
