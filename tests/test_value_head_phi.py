"""Phi-1 leaf value oracle tests.

Verifies favor_phi:
  - matches favor when 2P-elim trigger doesn't fire (no behavior change).
  - fires the 2P elimination bonus when opp is weak and we dominate.
  - 4P path retains the original ELIMINATION_BONUS=55 (no regression).
  - select_favor_fn() routes BASELINE_VALUE_HEAD=phi -> favor_phi.
  - uses PHI_HORIZON for the pv_horizon t_total (not EPISODE_STEPS).
"""
import importlib
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _reload_value_with_env(env: dict):
    saved = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import agents.baseline.value as v
        importlib.reload(v)
        return v
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _obs(my_ships, my_prod, opp_ships, opp_prod, *, step=0,
         num_opps=1, weakest_share=1.0):
    """Build a minimal obs dict with planets/fleets that aggregate correctly.

    `num_opps`: 1 for 2P scenarios, 3 for 4P.
    `weakest_share`: fraction of opp ships/prod assigned to the weakest in 4P
        (only relevant when num_opps > 1).
    """
    # Planets: [id, owner, x, y, radius, ships, production]
    planets = [
        [0, 0, 50, 50, 2.0, my_ships, my_prod],
    ]
    if num_opps == 1:
        planets.append([1, 1, 30, 30, 2.0, opp_ships, opp_prod])
    else:
        # 4P: distribute opp_ships/opp_prod across opps, weakest gets `weakest_share`.
        weakest_ships = opp_ships * weakest_share
        weakest_prod = opp_prod * weakest_share
        other = (1.0 - weakest_share) / (num_opps - 1)
        planets.append([1, 1, 30, 30, 2.0, weakest_ships, weakest_prod])
        for k in range(2, num_opps + 1):
            planets.append([k, k, 30 + k, 30 + k, 2.0,
                            opp_ships * other, opp_prod * other])
    return {"planets": planets, "fleets": [], "step": step}


def test_phi_matches_favor_when_no_elim_in_2p():
    """When opp is strong (no 2P-elim trigger), favor_phi differs from
    favor only in the pv_horizon t_total parameter (250 vs 500).
    Bracket: both produce values of similar sign and magnitude."""
    v = _reload_value_with_env({
        "PHI_HORIZON": None, "PHI_GAMMA": None, "PHI_ELIM_BONUS": None,
    })
    obs = _obs(my_ships=50, my_prod=5, opp_ships=80, opp_prod=8, step=50)
    f = v.favor(obs, me=0, num_seats=2, gamma=0.99)
    p = v.favor_phi(obs, me=0, num_seats=2, gamma=0.99)
    # Both are negative (opp dominates production); favor uses t_total=500,
    # favor_phi uses t_total=250 -> smaller pv -> less negative.
    assert f < 0 and p < 0
    assert p > f, f"phi (t_total=250) should be less negative than favor (t_total=500); got phi={p}, favor={f}"


def test_phi_2p_elim_bonus_fires():
    """Opp very weak (5 ships, 1 prod -> strength = 5 + 15 = 20), we
    dominate (200 ships, 10 prod -> strength = 200 + 150 = 350).
    20 <= 110 AND 350 >= 0.9 * 20 = 18 -> elim trigger.

    favor_phi vs favor: phi adds +PHI_ELIM_BONUS (default 300) but also
    uses a smaller pv_horizon (t_total=250 vs 500), which shrinks the
    production-difference term by ~110 in this scenario. Net delta is
    ~189 (300 - 110). Assert the elim bonus is positive-net."""
    v = _reload_value_with_env({
        "PHI_HORIZON": None, "PHI_GAMMA": None, "PHI_ELIM_BONUS": None,
    })
    obs = _obs(my_ships=200, my_prod=10, opp_ships=5, opp_prod=1, step=50)
    base_favor = v.favor(obs, me=0, num_seats=2, gamma=0.99)
    phi = v.favor_phi(obs, me=0, num_seats=2, gamma=0.99)
    # Sanity: phi strictly greater than favor (elim_bonus net of pv shrink).
    delta = phi - base_favor
    assert delta > 100.0, (
        f"expected phi - favor > 100 (elim bonus net of pv shrink); "
        f"got phi={phi}, favor={base_favor}, delta={delta}"
    )


def test_phi_2p_elim_bonus_isolated():
    """Compare phi(elim_bonus=300) against phi(elim_bonus=0) on the same
    state. Delta should be exactly 300.0 (the bonus, no pv compounding)."""
    v_with = _reload_value_with_env({
        "PHI_ELIM_BONUS": "300.0",
    })
    obs = _obs(my_ships=200, my_prod=10, opp_ships=5, opp_prod=1, step=50)
    phi_with = v_with.favor_phi(obs, me=0, num_seats=2, gamma=0.99)
    v_without = _reload_value_with_env({
        "PHI_ELIM_BONUS": "0.0",
    })
    phi_without = v_without.favor_phi(obs, me=0, num_seats=2, gamma=0.99)
    assert abs((phi_with - phi_without) - 300.0) < 1e-6, (
        f"isolated elim bonus delta should be 300; got {phi_with - phi_without}"
    )


def test_phi_no_elim_when_we_dont_dominate():
    """Opp weak (5 ships, 1 prod, strength=20) but we don't dominate
    (10 ships, 0 prod, strength=10). 10 < 0.9 * 20 = 18 -> no trigger."""
    v = _reload_value_with_env({
        "PHI_HORIZON": None, "PHI_GAMMA": None, "PHI_ELIM_BONUS": None,
    })
    obs = _obs(my_ships=10, my_prod=0, opp_ships=5, opp_prod=1, step=50)
    phi = v.favor_phi(obs, me=0, num_seats=2, gamma=0.99)
    # No elim bonus. phi ≈ base value only.
    # base = (10 - 5) + (0 - 1) * pv ≈ 5 - pv. For pv ≈ 90 (gamma=0.99,
    # t_total=250, step=50, eta=0): 5 - 90 ≈ -85. Definitely below 200.
    assert phi < 100.0, (
        f"no elim trigger expected (we don't dominate); got phi={phi}"
    )


def test_phi_4p_retains_original_elim_bonus():
    """4P path uses ELIMINATION_BONUS=55, NOT PHI_ELIM_BONUS. So
    favor_phi in 4P is structurally identical to favor in 4P modulo
    the t_total horizon."""
    v = _reload_value_with_env({
        "PHI_HORIZON": None, "PHI_GAMMA": None, "PHI_ELIM_BONUS": None,
    })
    # 4P: 3 opps, weakest gets 70% of opp_total -> weakest_ships=7, weakest_prod=0.7
    obs = _obs(my_ships=200, my_prod=10, opp_ships=10, opp_prod=1,
               step=50, num_opps=3, weakest_share=0.7)
    f = v.favor(obs, me=0, num_seats=4, gamma=0.99)
    p = v.favor_phi(obs, me=0, num_seats=4, gamma=0.99)
    # Both should fire 4P elim bonus (=55). Delta between phi and favor
    # is just the t_total difference; should be relatively small.
    # If phi were using PHI_ELIM_BONUS=300 in 4P, delta would be huge.
    delta = abs(p - f)
    assert delta < 200.0, (
        f"4P elim bonus must remain 55, not PHI_ELIM_BONUS=300; "
        f"phi={p}, favor={f}, delta={delta}"
    )


def test_select_favor_fn_routes_phi(monkeypatch):
    """BASELINE_VALUE_HEAD=phi resolves to favor_phi.

    select_favor_fn reads os.environ at call time, so we set the env
    via monkeypatch (NOT the reload helper which restores in finally
    before the test reads the result)."""
    import agents.baseline.value as v
    monkeypatch.setenv("BASELINE_VALUE_HEAD", "phi")
    fn = v.select_favor_fn()
    assert fn is v.favor_phi


def test_select_favor_fn_default_unchanged(monkeypatch):
    """Default (no env) routes to favor (not favor_phi)."""
    import agents.baseline.value as v
    monkeypatch.delenv("BASELINE_VALUE_HEAD", raising=False)
    fn = v.select_favor_fn()
    assert fn is v.favor
