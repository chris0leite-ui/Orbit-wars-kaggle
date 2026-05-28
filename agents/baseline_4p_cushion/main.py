"""baseline_4p_cushion — doctrine-anchored 4P delayed-launch wrapper.

What it does: in 4P/3P FFA games, returns no actions for the first
`CUSHION_STEPS` ticks; otherwise delegates to `agents.baseline.main.agent`
unchanged. 2P games are passed through identically — no behaviour change.

Why: `audit/2026-05-28-peak-53088099-share.md` shows the live μ=1125
peak loses 0/2 4P games in the sample, launching at step ~20. The
doctrine's n=92 empirical (evaluation-metrics §5 / doctrine §8.3)
puts 4P winners' first-capture median at step 137 and 4P losers' at
72. Baseline is launching 6× earlier than 4P-winner-median; the
cushion gate makes 4P first-launch happen later, after production
has accumulated, matching the empirical winning fingerprint.

A/B protocol: run this focal against three current-baseline opponents
in `scripts/ffa_tournament.run_ffa_tournament`, n ≥ 32 seeds × 4 seat
rotations. Compare focal first-place rate to baseline first-place rate
on the same background. Rule 45 floor: n ≥ 32 with Wilson-lo ≥ 0.30
(4P first-place is 0.25 at parity — gate is lift above parity).
"""

from __future__ import annotations

import os

from agents.baseline.main import agent as baseline_agent
from agents.baseline.main import _as_dict
from agents.baseline.main import _num_seats

# Default 60 ticks. Doctrine empirics put 4P-winner median first-
# capture at 137; 4P-loser median at 72; baseline currently launches at
# step ~20. 60 is the midpoint between current (~20) and 4P-loser
# median (72) — conservative first-cut. Env-var overridable so the A/B
# can sweep 30 / 60 / 90 / 120 before locking.
DEFAULT_CUSHION_STEPS: int = 60


def _cushion_steps() -> int:
    raw = os.environ.get("BASELINE_4P_CUSHION_STEPS")
    if raw is None or raw.strip() == "":
        return DEFAULT_CUSHION_STEPS
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_CUSHION_STEPS


def agent(obs, configuration=None):
    obs_d = _as_dict(obs)
    step = int(obs_d.get("step", 0))
    planets = obs_d.get("planets", []) or []
    fleets = obs_d.get("fleets", []) or []

    # Count seats from the obs (mirrors baseline's _num_seats but inlined
    # so we don't need the full Planet/Fleet construction for the gate).
    max_owner = -1
    for p in planets:
        if int(p[1]) > max_owner:
            max_owner = int(p[1])
    for f in fleets:
        if int(f[1]) > max_owner:
            max_owner = int(f[1])
    num_seats = max(2, max_owner + 1)

    if num_seats >= 3 and step < _cushion_steps():
        # 4P/3P cushion phase: no actions. Production accrues at the
        # home planet; opponents fight each other in the early game
        # (4P kingmaker, doctrine §8.3).
        return []
    return baseline_agent(obs, configuration)
