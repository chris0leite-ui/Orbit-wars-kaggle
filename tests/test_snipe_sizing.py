"""Tests for snipe.py's pred_ships-aware launch sizing (2026-05-15).

Two bugs from real-game observation:
- Over-commit: snipe sent 16 ships at a 15-ship neutral planet when an
  enemy 1-ship attack was already inbound; 15 ships would have sufficed.
  Root cause: target_min was computed from CURRENT garrison, ignoring
  inbound enemy fleets that would reduce the garrison before arrival.
- The fix is a DOWNSIZE-ONLY override: when pred_ships at our predicted
  arrival is LOWER than current garrison, use pred_ships + 1 for sizing.

PI test case: "neutral planet close to us that gets attacked should
almost certainly be captured by a 2-ship fleet or a bit higher."
"""

from __future__ import annotations

from types import SimpleNamespace

from lib.intent import World
from lib.missions.snipe import propose_snipe_missions
from lib.world_model import WorldModel


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    return SimpleNamespace(
        id=pid, owner=owner, x=x, y=y, radius=radius,
        ships=ships, production=production,
    )


def _world(my_id, planets, fleets=None, *, step=0):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": fleets or [],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": step,
    }
    return World.from_obs(obs)


def test_pred_ships_unused_when_no_inbound_fleets():
    """No in-flight fleets → pred_ships == t.ships → target_min unchanged.

    Snipe should size at int(t.ships) + 1 = 16 for a 15-ship enemy.
    """
    me = _planet(0, owner=0, x=0.0, y=0.0, ships=50, production=2)
    enemy = _planet(1, owner=1, x=20.0, y=0.0, ships=15, production=2)
    world = _world(my_id=0, planets=[me, enemy])
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    assert missions, "should propose at least one mission"
    # Enemy target → arrival_size will bump for prod growth; proposal stage
    # is target_min = current + 1 = 16 (no inbound friendly/enemy here).
    m = next(x for x in missions if x.target_id == 1)
    assert m.ships >= 16, f"expected ≥16; got {m.ships}"


def test_pred_ships_downsizes_neutral_with_inbound_enemy():
    """PI scenario: neutral close to us is being attacked by an enemy fleet
    that will reduce its garrison below current before we arrive.

    Setup:
      - Us at (0,0) with 50 ships
      - Neutral at (10,0) with 5 ships
      - Enemy at (100,0) with 50 ships, sending a 4-ship fleet inbound to
        the neutral. The 4 ships bounce off (4 < 5), leaving the neutral
        with 1 ship.

    Expected: snipe's target_min should be ~2 (= 1 + 1), not 6 (= 5 + 1).
    """
    me = _planet(0, owner=0, x=0.0, y=0.0, ships=50, production=2)
    neutral = _planet(1, owner=-1, x=10.0, y=0.0, ships=5, production=2)
    enemy_src = _planet(2, owner=1, x=100.0, y=0.0, ships=50, production=2)
    # Enemy 4-ship fleet at (15,0) angled left toward the neutral at (10,0).
    # fleet_speed(4) ≈ 1.45 → eta ≈ ceil(5/1.45) = 4 turns. Our proxy fleet
    # eta is ~7 → enemy lands before us. Combat: 4 vs 5 → enemy bounces,
    # neutral has 1 ship at turn 4. Neutrals don't grow → still 1 at turn 7.
    # Fleet schema: (id, owner, x, y, angle, from_planet_id, ships).
    import math as _m
    fleet = (0, 1, 15.0, 0.0, _m.pi, 2, 4)
    world = _world(my_id=0, planets=[me, neutral, enemy_src], fleets=[fleet])
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    # There should be a snipe mission from our planet to the neutral.
    relevant = [m for m in missions if m.src_id == 0 and m.target_id == 1]
    assert relevant, (
        f"expected a snipe mission to the neutral; got {len(missions)} "
        f"total: {[(m.src_id, m.target_id, m.ships) for m in missions]}"
    )
    m = relevant[0]
    # With pred_ships=1 (after enemy 4 bounces off 5), target_min = 2.
    # PI's "or a bit higher depending on speed" — allow up to 5 (some slack
    # for the aggressive-sizing branch + proxy/refined eta drift).
    assert m.ships <= 5, (
        f"PI scenario: snipe should send ~2 ships when neutral is whittled "
        f"by inbound enemy; got {m.ships}. Bug: snipe sized from current "
        f"garrison (5) instead of pred_ships at arrival (~1)."
    )


def test_pred_ships_does_not_upsize_enemy_with_production_growth():
    """Production growth on an owned enemy target should NOT inflate
    proposer sizing — that's `arrival_size`'s job. Downsize-only fix
    keeps target_min = current + 1 here.
    """
    me = _planet(0, owner=0, x=0.0, y=0.0, ships=50, production=2)
    # Enemy planet far away (long eta → significant production growth).
    enemy = _planet(1, owner=1, x=80.0, y=0.0, ships=10, production=3)
    world = _world(my_id=0, planets=[me, enemy])
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    m = next(x for x in missions if x.target_id == 1)
    # Proposer should ONLY use pred_ships when LOWER than current.
    # pred_ships at long eta = 10 + 3*eta > 10 → we use current = 11.
    # arrival_size will later bump intent.ships to pred_ships+1 = 10+3*eta+1.
    assert m.ships >= 11 and m.ships <= 30, (
        f"proposer should stay near target_min=11 for enemy growth; got {m.ships}"
    )
