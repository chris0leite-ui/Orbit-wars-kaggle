"""v4_endgame — v3_snipe + end-scenario routing.

After the mirror tier ladder (Tier 0-2 + hybrid) was empirically
falsified vs v3_snipe, this iteration pivots to building ON TOP of
v3 rather than against it: keep v3's strategy unchanged on most
turns, only override when a closed-form end-scenario check shows
the optimal play is something other than v3's default.

End-scenario triggers implemented (from the plan):

  W1 Production-rate lockout. If my_total + my_prod * remaining_steps
     strictly exceeds opp_total + opp_prod * remaining_steps, we WIN
     by arithmetic — switch to defensive (return no attacks). Mirror
     lock implicit: opp can't catch up; trading ships is risk-up
     reward-zero.

  W4 Ship-count lockout. Same idea but ignoring our future production
     to check a tighter bound: my_total > opp_total + MAX_PROD * remaining.

  D1 Step-480 parity freeze. Last 20 turns + ship parity within ε.
     Fleets launched now can't land before step 500 — just defend.

Otherwise pass through to v3_snipe. This is purely additive; if no
trigger fires, behavior is identical to v3.

Plan reference: /root/.claude/plans/you-are-a-top-parallel-swan.md
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_V3_AGENT = None
MAX_PROD = 5
EPISODE_STEPS = 500
PARITY_EPS_RATIO = 0.05  # within 5% ship count = "parity"


def _load_v3():
    global _V3_AGENT
    if _V3_AGENT is not None:
        return _V3_AGENT
    path = Path(__file__).resolve().parents[2] / "agents" / "v3_snipe" / "main.py"
    spec = importlib.util.spec_from_file_location("_endgame_v3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _V3_AGENT = mod.agent
    return _V3_AGENT


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _totals(planets, fleets, my_id: int, opp_id: int):
    """Return (my_ships, opp_ships, my_prod, opp_prod)."""
    my_s = my_p = opp_s = opp_p = 0
    for p in planets:
        if p[1] == my_id:
            my_s += int(p[5])
            my_p += int(p[6])
        elif p[1] == opp_id:
            opp_s += int(p[5])
            opp_p += int(p[6])
    for f in fleets:
        if f[1] == my_id:
            my_s += int(f[6])
        elif f[1] == opp_id:
            opp_s += int(f[6])
    return my_s, opp_s, my_p, opp_p


def _opp_id_2p(planets, my_id: int):
    """The (single) other non-neutral owner if this is a 2P board, else None."""
    owners = {p[1] for p in planets if p[1] != -1}
    if len(owners) != 2:
        return None
    return next(iter(owners - {my_id}))


def _end_scenario(obs, my_id: int):
    """Return one of {"coast", "freeze", "default"}.

    Conservative — only fires when the math is strictly in our favor.
    Boundary cases (equal, +/-1) stay on default so v3 keeps planning.
    """
    planets = _obs_get(obs, "planets", []) or []
    fleets = _obs_get(obs, "fleets", []) or []
    step = int(_obs_get(obs, "step", 0))
    opp_id = _opp_id_2p(planets, my_id)
    if opp_id is None:
        return "default"

    my_s, opp_s, my_p, opp_p = _totals(planets, fleets, my_id, opp_id)
    remaining = max(0, EPISODE_STEPS - step)

    # W1: production-rate lockout — we strictly win by arithmetic.
    if my_s + my_p * remaining > opp_s + opp_p * remaining + 1:
        return "coast"

    # W4: ship-count lockout — we win even if opp produces at full rate.
    if my_s > opp_s + MAX_PROD * remaining + 1:
        return "coast"

    # D1: terminal-window parity freeze.
    if remaining <= 20:
        total = max(1, my_s + opp_s)
        diff_ratio = abs(my_s - opp_s) / total
        if diff_ratio <= PARITY_EPS_RATIO and my_s >= opp_s:
            return "freeze"

    return "default"


def agent(obs):
    my_id = int(_obs_get(obs, "player", 0))

    scenario = _end_scenario(obs, my_id)
    if scenario in ("coast", "freeze"):
        # Won-or-tied subgame: don't risk ships. Stop attacking.
        return []

    return _load_v3()(obs)
