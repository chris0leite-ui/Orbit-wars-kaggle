"""Tests for the swept-pair classifier extension in
`scripts/episode_postmortem.attribute_fleets`.

PI hypothesis on 2026-05-17: most `vanished_in_space` fleets are killed
by comet collisions. The empirical refutation: only ~0.1% (12 of 9,507
fleets across v15's 92 live games) are comet collisions. The remaining
~9% were misclassified by the original `best_d < 5.0` single-point check
because the planet had orbited out of range by obs_vanish.

These tests pin the new behaviour: swept-pair against every planet
using the engine's primitive, with fallbacks for OOB-on-new-pos and
expired comets.
"""

from __future__ import annotations

import math

from scripts.episode_postmortem import (
    _comet_old_new_by_pid,
    _fleet_new_pos,
    _new_pos_oob,
    _swept_pair_planet_hit,
)
from lib.game.interpreter import COMET_RADIUS


def _fleet(fid, owner, x, y, angle, src=0, ships=10):
    """Fleet tuple: [id, owner, x, y, angle, from_planet_id, ships]."""
    return [fid, owner, x, y, angle, src, ships]


def _planet(pid, owner, x, y, radius=2.4, ships=10, prod=1):
    """Planet tuple: [id, owner, x, y, radius, ships, production]."""
    return [pid, owner, x, y, radius, ships, prod]


def _obs(planets, fleets=None, comets=None, comet_pids=None):
    return {
        "planets": planets,
        "fleets": fleets or [],
        "comets": comets or [],
        "comet_planet_ids": comet_pids or [],
    }


# ---------------------------------------------------------------------------
# _fleet_new_pos
# ---------------------------------------------------------------------------


def test_fleet_new_pos_at_angle_zero():
    """Angle 0 → fleet moves +x by speed."""
    # 10 ships -> speed = 1.0 + (6-1)*(log(10)/log(1000))**1.5
    # = 1.0 + 5*(1/3)**1.5 ≈ 1.0 + 5*0.192 ≈ 1.96
    last = _fleet(0, 0, 50.0, 50.0, 0.0, ships=10)
    fx, fy, nx, ny = _fleet_new_pos(last)
    assert fx == 50.0 and fy == 50.0
    assert nx > fx and abs(ny - fy) < 1e-9


# ---------------------------------------------------------------------------
# _new_pos_oob
# ---------------------------------------------------------------------------


def test_new_pos_oob_fires_when_fleet_about_to_exit():
    """Fleet near the right edge moving right should be flagged OOB."""
    last = _fleet(0, 0, 99.5, 50.0, 0.0, ships=400)  # large ships → high speed
    assert _new_pos_oob(last)


def test_new_pos_oob_silent_when_in_bounds():
    """Fleet near the middle moving slowly should not be flagged."""
    last = _fleet(0, 0, 50.0, 50.0, 0.0, ships=10)
    assert not _new_pos_oob(last)


# ---------------------------------------------------------------------------
# _comet_old_new_by_pid
# ---------------------------------------------------------------------------


def test_comet_old_new_by_pid_returns_old_and_next():
    """One comet group with 1 planet, path_index=2 → old=path[2], new=path[3]."""
    comets = [{
        "planet_ids": [42],
        "paths": [[[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]]],
        "path_index": 2,
    }]
    out = _comet_old_new_by_pid(_obs([], comets=comets))
    assert out == {42: ((30.0, 30.0), (40.0, 40.0))}


def test_comet_old_new_by_pid_handles_last_step_stays_put():
    """When idx+1 is past the end, the comet stays at its current pos."""
    comets = [{
        "planet_ids": [42],
        "paths": [[[10.0, 10.0], [20.0, 20.0]]],
        "path_index": 1,  # last reachable index
    }]
    out = _comet_old_new_by_pid(_obs([], comets=comets))
    assert out == {42: ((20.0, 20.0), (20.0, 20.0))}


# ---------------------------------------------------------------------------
# _swept_pair_planet_hit — the headline test
# ---------------------------------------------------------------------------


def test_swept_pair_finds_orbital_planet_hit_that_moved_by_obs_vanish():
    """Planet starts at (53, 50) in obs_prev, orbits to (54.5, 51) in
    obs_vanish (small motion, like an orbital planet shifting one
    tick). Fleet at (50, 50) moving +x with 400 ships (speed≈5.0).
    Fleet's swept segment [50,50]→[~55,50] passes within the planet's
    radius of 2.4 at some point during the tick. The static best_d
    check from obs_vanish would find d≈4.7 from (50,50) — outside the
    5.0 threshold by chance OR within it (borderline). Swept-pair on
    obs_prev's position is the deterministic primitive.
    """
    obs_prev = _obs([_planet(99, -1, 53.0, 50.0, radius=2.4)])
    obs_vanish = _obs([_planet(99, -1, 54.5, 51.0, radius=2.4)])
    last = _fleet(0, 0, 50.0, 50.0, 0.0, ships=400)
    pid, is_comet = _swept_pair_planet_hit(obs_prev, obs_vanish, last)
    assert pid == 99
    assert is_comet is False


def test_swept_pair_no_collision_when_planet_outraces_fleet():
    """If the planet's relative motion carries it AWAY from the fleet
    faster than the fleet approaches, the engine sees no collision
    (swept-pair models relative motion). Documents the
    expectation: don't write a naive distance-to-old-position check —
    this case is correctly a miss in the engine's semantics.
    """
    obs_prev = _obs([_planet(99, -1, 55.0, 50.0, radius=2.4)])
    obs_vanish = _obs([_planet(99, -1, 70.0, 50.0, radius=2.4)])
    last = _fleet(0, 0, 50.0, 50.0, 0.0, ships=400)
    pid, _ = _swept_pair_planet_hit(obs_prev, obs_vanish, last)
    assert pid is None


def test_swept_pair_finds_comet_collision_even_when_expired():
    """Comet was in obs_prev but expired by obs_vanish (gone from planets
    list). We should still classify as comet_collision using the comet
    path data from obs_prev.
    """
    comet_path = [[55.0, 50.0], [56.0, 50.0]]
    obs_prev = _obs(
        # Planet entry for the comet exists in obs_prev's planets list.
        planets=[_planet(33, -1, 55.0, 50.0, radius=COMET_RADIUS)],
        comets=[{"planet_ids": [33], "paths": [comet_path], "path_index": 0}],
        comet_pids=[33],
    )
    # obs_vanish has no comet at all (expired this tick).
    obs_vanish = _obs(planets=[])
    last = _fleet(0, 0, 50.0, 50.0, 0.0, ships=400)
    pid, is_comet = _swept_pair_planet_hit(obs_prev, obs_vanish, last)
    assert pid == 33
    assert is_comet is True


def test_swept_pair_no_hit_returns_none():
    """Fleet flying past empty space with no nearby planets."""
    obs_prev = _obs([_planet(99, -1, 80.0, 80.0)])
    obs_vanish = _obs([_planet(99, -1, 80.0, 80.0)])
    last = _fleet(0, 0, 10.0, 10.0, 0.0, ships=10)
    pid, _ = _swept_pair_planet_hit(obs_prev, obs_vanish, last)
    assert pid is None
