"""Sanity tests for `nearest_opp_policy` in `lib/opp_model.py`.

Coverage:
  - Target selection picks closest non-our planet even when a farther
    one has higher production (the lite_greedy ROI tie-break would
    flip this; nearest should not).
  - Affordability skip: a target whose defender prediction exceeds
    src.ships at eta returns empty moves for that src.
  - Owner filter: only player-owned planets generate launches.
  - Neutral targets use static defender count (no production accrual
    during fleet flight; mirror lite_greedy's neutral handling).
  - Env-var routing: `BASELINE_OPP_MODEL=nearest` makes
    `_select_opp_policy` return `nearest_opp_policy`.
"""

from __future__ import annotations

import os


# Planet tuple shape used throughout: [id, owner, x, y, radius, ships, production]


def _obs(player: int, planets: list) -> dict:
    return {"player": player, "planets": planets, "fleets": []}


def test_nearest_picks_closest_over_higher_prod_farther():
    """Two targets visible from one source.
       Near target: low production, close. Far target: high production, far.
       lite_greedy would prefer far/high-prod (ROI), nearest must pick near.
    """
    from lib.opp_model import nearest_opp_policy

    # src at origin, 100 ships, owner=0.
    # tgt_A at (10, 0): close, low production (1).
    # tgt_B at (60, 0): far, high production (5).
    src = [0, 0, 0.0, 0.0, 1.0, 100.0, 0.0]
    tgt_near = [1, 1, 10.0, 0.0, 1.0, 5.0, 1.0]
    tgt_far = [2, 1, 60.0, 0.0, 1.0, 5.0, 5.0]
    obs = _obs(player=0, planets=[src, tgt_near, tgt_far])

    moves = nearest_opp_policy(obs)
    assert len(moves) == 1, f"expected 1 launch from src; got {moves}"
    src_id, angle, ships = moves[0]
    assert src_id == 0
    # Angle should aim at tgt_near (atan2(0, 10) = 0.0), not tgt_far.
    assert abs(angle) < 0.01, (
        f"expected angle ~0 (aiming at tgt_near at +x); got {angle}"
    )


def test_nearest_skips_unaffordable_capture():
    """Defender prediction > src.ships → no launch."""
    from lib.opp_model import nearest_opp_policy

    # src has 12 ships. Single target: enemy planet at d=10 with 50 defenders.
    src = [0, 0, 0.0, 0.0, 1.0, 12.0, 0.0]
    tgt = [1, 1, 10.0, 0.0, 1.0, 50.0, 0.0]
    obs = _obs(player=0, planets=[src, tgt])

    moves = nearest_opp_policy(obs)
    assert moves == [], f"expected no launch (unaffordable); got {moves}"


def test_nearest_owner_filter_only_player_planets():
    """Only the player's own planets contribute launches."""
    from lib.opp_model import nearest_opp_policy

    # planet 0 owned by p1 (NOT us), planet 1 owned by us (p0).
    # planet 2 is the only valid target.
    p1_planet = [0, 1, 0.0, 0.0, 1.0, 50.0, 1.0]
    my_planet = [1, 0, 20.0, 0.0, 1.0, 50.0, 1.0]
    tgt = [2, -1, 25.0, 0.0, 1.0, 3.0, 1.0]
    obs = _obs(player=0, planets=[p1_planet, my_planet, tgt])

    moves = nearest_opp_policy(obs)
    # Exactly one launch should fire — from my_planet (id=1).
    assert len(moves) == 1
    assert moves[0][0] == 1, (
        f"expected launch from my_planet (id=1); got src={moves[0][0]}"
    )


def test_nearest_neutral_no_production_accrual():
    """Neutral target: defender count must NOT accrue production during flight.
    Mirror lite_greedy's bugfix (lib/opp_model.py:253-256).
    """
    from lib.opp_model import nearest_opp_policy

    # src has 50 ships at origin. neutral target at d=20 with 13 ships, prod=10.
    # If production accrued during eta (~3 turns), defenders_at_eta would be
    # 13 + 30 = 43, needed = 44; launch would skip.
    # With neutral handling, defenders_at_eta = 13, needed = 14; launch fires.
    src = [0, 0, 0.0, 0.0, 1.0, 50.0, 0.0]
    tgt = [1, -1, 20.0, 0.0, 1.0, 13.0, 10.0]
    obs = _obs(player=0, planets=[src, tgt])

    moves = nearest_opp_policy(obs)
    assert len(moves) == 1, (
        f"expected 1 launch (neutral, defenders=13, budget=50); got {moves}"
    )
    src_id, angle, ships = moves[0]
    # Ship sizing is max(aggressive=35, needed=14) → 35, clamped to budget=50.
    assert ships == 35, f"expected ships=35 (aggressive sizing); got {ships}"


def test_nearest_env_routes_through_chooser(monkeypatch):
    """BASELINE_OPP_MODEL=nearest → _select_opp_policy returns nearest_opp_policy."""
    from agents.baseline.chooser import _select_opp_policy
    from lib.opp_model import lite_greedy_policy, nearest_opp_policy

    # Default routing → lite_greedy.
    monkeypatch.delenv("BASELINE_OPP_MODEL", raising=False)
    monkeypatch.delenv("BASELINE_OPP_TIER", raising=False)
    assert _select_opp_policy() is lite_greedy_policy

    # nearest routing.
    monkeypatch.setenv("BASELINE_OPP_MODEL", "nearest")
    assert _select_opp_policy() is nearest_opp_policy

    # Unknown value → fall through to lite_greedy (preserves default).
    monkeypatch.setenv("BASELINE_OPP_MODEL", "xyzzy")
    assert _select_opp_policy() is lite_greedy_policy
