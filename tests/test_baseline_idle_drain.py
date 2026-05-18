"""Unit tests for H1 post-chooser idle drain in agents/baseline/main.

Audit `audit/replays/idle-trajectory-2026-05-17.md` measured 43.8pct
isolated ship-turns in trajectory champion. H1 drains rear-source
surplus only when chooser left them idle.
"""

from __future__ import annotations

from agents.baseline.main import drain_idle_rear
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


class _FakeModel:
    """Stand-in for WorldModel — caller of `time_to_enemy_threat`."""

    def __init__(self, threats: dict[int, int | None] | None = None):
        self._threats = threats or {}

    def time_to_enemy_threat(self, planet_id: int, my_id: int, world):
        return self._threats.get(int(planet_id))


# Planet: (id, owner, x, y, radius, ships, production)


def _planets(ours: list[tuple], theirs: list[tuple]) -> list[Planet]:
    """Build a planet list. Each tuple = (id, x, y, ships)."""
    out = []
    for pid, x, y, ships in ours:
        out.append(Planet(int(pid), 0, float(x), float(y), 1.0, int(ships), 1))
    for pid, x, y, ships in theirs:
        out.append(Planet(int(pid), 1, float(x), float(y), 1.0, int(ships), 1))
    return out


def test_drain_idle_rear_emits_when_source_idle_and_rear():
    # Source at corner (5, 5), enemy at center (50, 50). Distance ~ 63.6.
    # Closer own planet at (40, 40). Distance from there to enemy ~ 14.1.
    # Chooser left source unused → H1 should emit a launch from src toward
    # the closer own planet.
    planets = _planets(
        ours=[(0, 5, 5, 50), (1, 40, 40, 5)],
        theirs=[(2, 50, 50, 100)],
    )
    moves = drain_idle_rear([], planets, my_id=0, world=None,
                             model=_FakeModel())
    assert len(moves) == 1
    src_id, angle, ships = moves[0]
    assert int(src_id) == 0
    assert int(ships) == 50 - 5  # threshold reserve = 5


def test_drain_skips_when_below_threshold():
    # Same setup but src has only 20 ships (below IDLE_DRAIN_THRESHOLD=30).
    planets = _planets(
        ours=[(0, 5, 5, 20), (1, 40, 40, 5)],
        theirs=[(2, 50, 50, 100)],
    )
    moves = drain_idle_rear([], planets, my_id=0, world=None,
                             model=_FakeModel())
    assert moves == []


def test_drain_skips_when_chooser_used_source():
    # Source has surplus but chooser already chose to fire from it.
    planets = _planets(
        ours=[(0, 5, 5, 50), (1, 40, 40, 5)],
        theirs=[(2, 50, 50, 100)],
    )
    chooser_moves = [[0, 0.0, 30]]
    moves = drain_idle_rear(chooser_moves, planets, my_id=0, world=None,
                             model=_FakeModel())
    assert moves == chooser_moves  # H1 left it alone


def test_drain_skips_when_source_already_frontier():
    # Source is close to enemy (distance ~14) — not rear.
    planets = _planets(
        ours=[(0, 40, 40, 50), (1, 5, 5, 5)],
        theirs=[(2, 50, 50, 100)],
    )
    moves = drain_idle_rear([], planets, my_id=0, world=None,
                             model=_FakeModel())
    assert moves == []


def test_drain_skips_when_threat_present():
    # Source is rear AND has surplus BUT enemy threat exists.
    planets = _planets(
        ours=[(0, 5, 5, 50), (1, 40, 40, 5)],
        theirs=[(2, 50, 50, 100)],
    )
    # FakeModel: enemy can reach planet 0 in 10 turns.
    moves = drain_idle_rear([], planets, my_id=0, world=None,
                             model=_FakeModel({0: 10}))
    assert moves == []


def test_drain_skips_when_no_closer_own_target():
    # Only own planet is the rear one itself — no forward staging available.
    planets = _planets(
        ours=[(0, 5, 5, 50)],
        theirs=[(2, 50, 50, 100)],
    )
    moves = drain_idle_rear([], planets, my_id=0, world=None,
                             model=_FakeModel())
    assert moves == []


def test_drain_disabled_when_env_off():
    import os
    os.environ["BASELINE_IDLE_DRAIN"] = "0"
    # Force module-level constant re-read by reimporting? The module reads
    # IDLE_DRAIN_ENABLED at import time, so we monkey-patch.
    import agents.baseline.main as bm
    old = bm.IDLE_DRAIN_ENABLED
    bm.IDLE_DRAIN_ENABLED = False
    try:
        planets = _planets(
            ours=[(0, 5, 5, 50), (1, 40, 40, 5)],
            theirs=[(2, 50, 50, 100)],
        )
        moves = drain_idle_rear([], planets, my_id=0, world=None,
                                 model=_FakeModel())
        assert moves == []  # disabled → no extras
    finally:
        bm.IDLE_DRAIN_ENABLED = old
        os.environ.pop("BASELINE_IDLE_DRAIN", None)


def test_drain_preserves_chooser_moves_and_appends():
    # Chooser emitted a launch from planet 5 (not rear); H1 adds drain
    # from planet 0 (rear). Both should appear.
    planets = _planets(
        ours=[(0, 5, 5, 50), (1, 40, 40, 5), (5, 45, 45, 20)],
        theirs=[(2, 50, 50, 100)],
    )
    chooser_moves = [[5, 0.0, 15]]
    moves = drain_idle_rear(chooser_moves, planets, my_id=0, world=None,
                             model=_FakeModel())
    assert len(moves) == 2
    assert moves[0] == [5, 0.0, 15]
    assert int(moves[1][0]) == 0
