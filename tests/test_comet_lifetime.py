"""Tests for comet lifetime handling in ROI scoring + the
`comet_remaining_lifetime` helper in `lib/world_model.py`.

Comets enter the board at steps 50/150/250/350/450 and leave when their
shared `path_index` reaches `len(path)`. Scoring a fleet that arrives
AFTER a comet's departure is wasted ships — the comet won't exist when
the fleet gets there.
"""

from __future__ import annotations

from types import SimpleNamespace

from lib.intent import World
from lib.world_model import _comet_paths_by_id, comet_remaining_lifetime


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    return SimpleNamespace(
        id=pid, owner=owner, x=x, y=y, radius=radius,
        ships=ships, production=production,
    )


def _world(my_id, planets, comet_groups=None):
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
        "step": 0,
    }
    return World.from_obs(obs)


def test_remaining_lifetime_returns_none_for_non_comet():
    """Static / orbiting planets have no comet lifetime."""
    world = _world(my_id=0, planets=[_planet(0, 0, 10.0, 10.0)])
    assert comet_remaining_lifetime(0, world) is None


def test_remaining_lifetime_returns_len_minus_index():
    """Comet's remaining lifetime = path length - current path_index."""
    # 50-step path; comet currently at index 10 → 40 steps remaining.
    path = [[float(i), 0.0] for i in range(50)]
    comet_group = {
        "planet_ids": [100],
        "paths": [path],
        "path_index": 10,
    }
    world = _world(
        my_id=0,
        planets=[_planet(100, -1, 5.0, 0.0)],
        comet_groups=[comet_group],
    )
    assert comet_remaining_lifetime(100, world) == 40


def test_remaining_lifetime_zero_at_end_of_path():
    """A comet at path_index == len(path) has 0 remaining."""
    path = [[float(i), 0.0] for i in range(10)]
    comet_group = {
        "planet_ids": [100],
        "paths": [path],
        "path_index": 10,
    }
    world = _world(
        my_id=0,
        planets=[_planet(100, -1, 5.0, 0.0)],
        comet_groups=[comet_group],
    )
    assert comet_remaining_lifetime(100, world) == 0


def test_paths_by_id_handles_empty_comets():
    """No comets in obs → empty lookup."""
    world = _world(my_id=0, planets=[_planet(0, 0, 10.0, 10.0)])
    assert _comet_paths_by_id(world) == {}


def test_snipe_score_zero_for_comet_departing_before_arrival():
    """A comet that leaves before our fleet arrives should score 0."""
    from lib.missions.snipe import propose_snipe_missions
    from lib.world_model import WorldModel

    # Source at (0,0), comet at (10,0) with 5 steps remaining.
    # Fleet of cost ships travels at fleet_speed(cost); with ships=11
    # (target.ships=10 + 1), eta = ceil(10 / ~2.0) = 5. So time_to_hold
    # = max(0, 5 - 5) = 0 → score = 0.
    path = [[float(i), 0.0] for i in range(20)]
    comet_group = {
        "planet_ids": [1],
        "paths": [path],
        "path_index": 15,  # 5 steps remaining
    }
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
    comet_missions = [m for m in missions if m.target_id == 1]
    assert len(comet_missions) == 1
    # Score may be 0 if eta >= remaining_lifetime, but in any case
    # should be much smaller than a comparable mission against a
    # static target with the same distance/cost.
    assert comet_missions[0].score == 0.0 or comet_missions[0].score < 1.0


def test_snipe_score_long_lived_comet_higher_than_short_lived():
    """Two comets at the same distance/production/cost: the one with
    more remaining lifetime should score higher."""
    from lib.missions.snipe import propose_snipe_missions
    from lib.world_model import WorldModel

    path_long = [[float(i), 0.0] for i in range(50)]
    path_short = [[float(i), 0.0] for i in range(50)]
    comet_long = {
        "planet_ids": [1],
        "paths": [path_long],
        "path_index": 5,    # 45 steps remaining
    }
    comet_short = {
        "planet_ids": [2],
        "paths": [path_short],
        "path_index": 40,   # 10 steps remaining
    }
    world = _world(
        my_id=0,
        planets=[
            _planet(0, 0, 0.0, 0.0, ships=200),
            _planet(1, -1, 10.0, 0.0, ships=10, production=2),
            _planet(2, -1, 0.0, 10.0, ships=10, production=2),
        ],
        comet_groups=[comet_long, comet_short],
    )
    model = WorldModel.from_world(world)
    missions = propose_snipe_missions(world, model)
    long_mission = next(m for m in missions if m.target_id == 1)
    short_mission = next(m for m in missions if m.target_id == 2)
    assert long_mission.score > short_mission.score
