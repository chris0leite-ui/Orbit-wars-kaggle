"""H15: hard-reject snipe missions targeting comets that depart before
arrival.

The pre-H15 behaviour was: if `comet_remaining_lifetime <= eta`, the
Mission was still emitted with `time_to_hold = 0` → score ≈ 0. This
consumed the source's per-source slot in `settle_plan` and crowded out
viable runner-up targets.

H15 hard-rejects at proposer level. The source falls through to its
runner-up target instead, restoring an extra opening per turn against
ladder opponents that capture comets aggressively.
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


def _world(my_id, planets, comet_groups=None, step=0):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": [],
        "angular_velocity": 0.0,
        "comet_planet_ids": [
            pid for g in (comet_groups or []) for pid in g["planet_ids"]
        ],
        "comets": comet_groups or [],
        "step": step,
    }
    return World.from_obs(obs)


def test_rejects_when_remaining_equals_eta():
    """Boundary: rem == eta → reject (the comet leaves on the arrival tick)."""
    path = [[float(i), 0.0] for i in range(20)]
    comet_group = {"planet_ids": [1], "paths": [path], "path_index": 15}
    world = _world(
        my_id=0,
        planets=[
            _planet(0, 0, 0.0, 0.0, ships=100),
            _planet(1, -1, 10.0, 0.0, ships=10, production=1),
        ],
        comet_groups=[comet_group],
    )
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    assert [m for m in missions if m.target_id == 1] == []


def test_rejects_when_remaining_less_than_eta():
    """rem < eta → reject."""
    # 20-step path at index 18 → rem = 2. Source at distance ~10 needs
    # eta ≥ 4 → rem < eta → reject.
    path = [[float(i), 0.0] for i in range(20)]
    comet_group = {"planet_ids": [1], "paths": [path], "path_index": 18}
    world = _world(
        my_id=0,
        planets=[
            _planet(0, 0, 0.0, 0.0, ships=100),
            _planet(1, -1, 10.0, 0.0, ships=10, production=1),
        ],
        comet_groups=[comet_group],
    )
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    assert [m for m in missions if m.target_id == 1] == []


def test_keeps_when_remaining_greater_than_eta():
    """rem > eta → keep (the comet is still on the board when we arrive)."""
    # rem = 30; eta ≈ 5 at this distance → keep.
    path = [[float(i), 0.0] for i in range(50)]
    comet_group = {"planet_ids": [1], "paths": [path], "path_index": 20}
    world = _world(
        my_id=0,
        planets=[
            _planet(0, 0, 0.0, 0.0, ships=100),
            _planet(1, -1, 10.0, 0.0, ships=10, production=1),
        ],
        comet_groups=[comet_group],
    )
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    kept = [m for m in missions if m.target_id == 1]
    assert len(kept) == 1
    assert kept[0].score > 0.0


def test_non_comet_target_unaffected():
    """The reject branch only fires when target.id is in comet set.
    Static targets remain unaffected by the H15 change."""
    world = _world(
        my_id=0,
        planets=[
            _planet(0, 0, 0.0, 0.0, ships=100),
            _planet(1, -1, 10.0, 0.0, ships=10, production=1),  # static
        ],
        # No comet groups → planet 1 is a regular neutral.
    )
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    assert any(m.target_id == 1 for m in missions)


def test_runner_up_takes_slot_when_comet_rejected():
    """When a source has a near departing comet AND a viable static
    runner-up at greater distance, the static target gets the per-source
    slot in settle_plan. (Proposer emits only the static one; settle_plan
    then picks it as the source's pick.)"""
    from lib.planner import settle_plan

    path = [[float(i), 0.0] for i in range(20)]
    comet_group = {"planet_ids": [1], "paths": [path], "path_index": 18}
    world = _world(
        my_id=0,
        planets=[
            _planet(0, 0, 0.0, 0.0, ships=100, production=1),
            _planet(1, -1, 5.0, 0.0, ships=2, production=2),  # comet, rem=2
            _planet(2, -1, 30.0, 0.0, ships=2, production=2),  # static
        ],
        comet_groups=[comet_group],
    )
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    # Proposer emits only the static target for source 0.
    assert [m.target_id for m in missions if m.src_id == 0] == [2]
    # settle_plan picks it.
    chosen = settle_plan(missions, world, model)
    assert len(chosen) == 1
    assert chosen[0].src_id == 0
    assert chosen[0].target_id == 2
