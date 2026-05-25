"""lite_greedy_policy 5-ship floor oracle tests (2026-05-25).

Verifies the source-ship floor change from 10 → 5 in `lib.opp_model.lite_greedy_policy`.
Pre-fix, a 7-ship source idled; post-fix it launches. The 4-ship case
remains a no-op (the floor still binds at 5).

Background: the rollout's opp model was systematically blind to 5-9 ship
opp planets — but live opps DO launch small recapture fleets from those.
The chooser's leaf value scored captures as "safe to hold" when in fact
opp would recapture in 5-8 turns with the 5-9 ship fleet our model
refused to simulate.
"""
from __future__ import annotations

import math

from lib.opp_model import lite_greedy_policy


def _planet(pid: int, owner: int, x: float, y: float, ships: int,
            production: int, radius: float = 1.0):
    """env planet tuple: (id, owner, x, y, radius, ships, production)."""
    return (pid, owner, float(x), float(y), float(radius),
            int(ships), int(production))


def _obs(player: int, planets: list):
    return {"player": player, "planets": planets}


def test_lite_greedy_launches_at_5_ships():
    """5-ship source SHOULD launch under the new floor (was the bug: pre-fix
    needed ≥ 10 ships). Targets a weak neutral nearby so the affordability
    + ROI checks pass."""
    obs = _obs(
        player=1,
        planets=[
            # opp source: 5 ships, prod=2, at origin.
            _planet(pid=0, owner=1, x=10.0, y=10.0, ships=5, production=2),
            # neutral target: 2 defenders, prod=3 (high ROI), close.
            _planet(pid=1, owner=-1, x=20.0, y=10.0, ships=2, production=3),
        ],
    )
    moves = lite_greedy_policy(obs)
    assert len(moves) == 1, f"expected one launch, got {moves}"
    src_id, angle, ships = moves[0]
    assert src_id == 0
    assert ships >= 3  # at least enough to capture 2 + production-during-flight
    # Angle points roughly along +x.
    assert abs(angle) < math.pi / 4, f"unexpected angle {angle}"


def test_lite_greedy_launches_at_exactly_5_ships():
    """Boundary case: exactly 5 ships. Floor is `< 5`, so 5 should launch."""
    obs = _obs(
        player=1,
        planets=[
            _planet(pid=0, owner=1, x=10.0, y=10.0, ships=5, production=2),
            _planet(pid=1, owner=-1, x=20.0, y=10.0, ships=1, production=3),
        ],
    )
    moves = lite_greedy_policy(obs)
    assert len(moves) == 1


def test_lite_greedy_skips_at_4_ships():
    """4-ship source still no-ops — the 5-ship floor binds at the boundary."""
    obs = _obs(
        player=1,
        planets=[
            _planet(pid=0, owner=1, x=10.0, y=10.0, ships=4, production=2),
            _planet(pid=1, owner=-1, x=20.0, y=10.0, ships=1, production=3),
        ],
    )
    moves = lite_greedy_policy(obs)
    assert moves == [], f"expected no launches, got {moves}"


def test_lite_greedy_still_skips_unaffordable_capture():
    """Affordability gate (lines 224-225) still binds: if defenders + prod
    accrual exceed source budget, skip. 5-ship source vs heavy neutral
    should not launch a bouncing fleet."""
    obs = _obs(
        player=1,
        planets=[
            _planet(pid=0, owner=1, x=10.0, y=10.0, ships=5, production=2),
            # 50-defender neutral way more than 5 ships can capture.
            _planet(pid=1, owner=-1, x=20.0, y=10.0, ships=50, production=1),
        ],
    )
    moves = lite_greedy_policy(obs)
    assert moves == [], f"5 ships should not launch at 50-defender target, got {moves}"
