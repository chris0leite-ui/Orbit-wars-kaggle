"""v7.6 — v7.5 minus recapture (σ-equiv + 4P-aware only).

v7.5 (σ-equiv + recapture + 4P-aware) regressed -8.3pp vs v7_0 in
2P A/B. The 4P-aware path doesn't activate in 2P, so the regression
is either σ-equiv (library-level) or recapture (mission class).
v7.6 isolates: keeps σ-equiv + 4P-aware, drops recapture.

If v7.6 PASS vs v7_0 → recapture was the bug (even after calibration).
If v7.6 FAIL  → σ-equiv interaction is regressive; drop everything.
If v7.6 NEUTRAL → both contributions are zero or wash out.
"""

from __future__ import annotations

from lib.v7_search import choose_simple_with_4p


def agent(obs, configuration=None):
    return choose_simple_with_4p(
        obs, configuration,
        K_2p=10,
        K_4p=8,
        wallclock_ms=700.0,
        include_recapture=False,   # the bisect: recapture OFF
        value_fn=None,
    )
