"""v6_steady — v3_snipe with a correctly-tightened end-scenario layer.

Iter 4 (v4_endgame) added W1/W4/D1 end-scenario routing on top of v3_snipe.
It produced the only iteration with wins vs v3 (4W) but also 10 losses
(37.5% W/D over 16 games). Diagnosis (plan, "Deep reframe" section):

  v4_endgame's W1 trigger fires on `my_s + my_p * remaining > opp_s
  + opp_p * remaining + 1`. The +1 margin only blocks exact equality.
  ANY 1-unit production lead with remaining=300 makes the inequality
  fire; we return [] (coast); opp keeps attacking; opp captures our
  planets → our production drops → arithmetic flips → we lose.

  W1 is over-eager and breaks the v3-vs-v3 natural draw lock (94%
  empirical per v2-vs-v2 audit; v3 inherits this).

v6_steady fixes the bug by:

  1. Dropping W1 entirely — the simple production-lockout arithmetic
     ignores opp's potential captures, so it's wrong-way biased.
  2. Replacing it with W1_SAFE: only coast when we win even in the
     worst case where opp captures EVERY planet of ours. Conservative;
     rarely fires; when it does the lockout is provable.
  3. Keeping W4 (ship-count lockout, 1500+ ship lead required —
     extremely rare, but real when it fires).
  4. Keeping D1 (step ≥ 480 + ship parity + we ≥ opp). Last-window
     freeze; only a few turns, low risk of breaking the lock.

The thesis (plan, "What we were missing"): the cannot-lose property
at our μ-bracket is INTRINSIC to v3 (via the v3-vs-v3 draw lock).
End-scenarios are useful ONLY when they preserve that lock. v6_steady
adds overlays that fire so rarely they can't damage it.

To climb ABOVE the lock (cannot-lose at a higher μ-bracket) we need
a strictly stronger base than v3 — recapture missions, gang-up,
or self-play RL. Those are separate iterations.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_V3_AGENT = None
MAX_PROD = 5
EPISODE_STEPS = 500
PARITY_EPS_RATIO = 0.05


def _load_v3():
    global _V3_AGENT
    if _V3_AGENT is not None:
        return _V3_AGENT
    path = Path(__file__).resolve().parents[2] / "agents" / "v3_snipe" / "main.py"
    spec = importlib.util.spec_from_file_location("_v6_v3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _V3_AGENT = mod.agent
    return _V3_AGENT


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _totals(planets, fleets, my_id: int, opp_id: int):
    """Return (my_ships, opp_ships, my_prod, opp_prod, my_planet_ships)."""
    my_s = my_p = opp_s = opp_p = 0
    my_planet_ships = 0
    for p in planets:
        if p[1] == my_id:
            my_s += int(p[5])
            my_p += int(p[6])
            my_planet_ships += int(p[5])
        elif p[1] == opp_id:
            opp_s += int(p[5])
            opp_p += int(p[6])
    for f in fleets:
        if f[1] == my_id:
            my_s += int(f[6])
        elif f[1] == opp_id:
            opp_s += int(f[6])
    return my_s, opp_s, my_p, opp_p, my_planet_ships


def _opp_id_2p(planets, my_id: int):
    owners = {p[1] for p in planets if p[1] != -1}
    if len(owners) != 2:
        return None
    return next(iter(owners - {my_id}))


def _end_scenario(obs, my_id: int):
    """Return one of {"coast", "freeze", "default"}.

    Conservative — only fires when the math is robust to opp's
    plausible attacks. v4_endgame's loose W1 is replaced by a
    worst-case-bounded W1_SAFE; W4 and D1 are kept as-is.
    """
    planets = _obs_get(obs, "planets", []) or []
    fleets = _obs_get(obs, "fleets", []) or []
    step = int(_obs_get(obs, "step", 0))
    opp_id = _opp_id_2p(planets, my_id)
    if opp_id is None:
        return "default"

    my_s, opp_s, my_p, opp_p, my_planet_ships = _totals(planets, fleets, my_id, opp_id)
    remaining = max(0, EPISODE_STEPS - step)

    # W1_SAFE: production-rate lockout that accounts for opp's worst-case
    # capture of all our planets. The pessimistic scenario:
    #   - We lose ALL planet garrison (my_s → my_s - my_planet_ships)
    #   - We lose ALL our production (my_p → 0)
    #   - Opp gains our garrison (opp_s → opp_s + my_planet_ships)
    #   - Opp gains our production (opp_p → opp_p + my_p)
    # If we still win even then, the lockout is provable.
    my_worst = my_s - my_planet_ships  # only fleets-in-flight survive
    opp_best_s = opp_s + my_planet_ships
    opp_best_p = opp_p + my_p
    if my_worst > opp_best_s + opp_best_p * remaining + 1:
        return "coast"

    # W4: ship-count lockout — we win even if opp produces at full rate
    # AND we get zero more production. Extremely strict; rarely fires.
    if my_s > opp_s + MAX_PROD * remaining + 1:
        return "coast"

    # D1: terminal-window parity freeze. Last 20 steps, near parity,
    # we are weakly ahead. No fleet launched now can land before step 500
    # (fleet speed ≥ 1 unit/turn; min fleet flight time ≥ ~10 turns).
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
        return []

    return _load_v3()(obs)
