"""Unit tests for lib/mission_book.py."""

from __future__ import annotations

from lib.mission_book import (
    CARRYFORWARD_WEIGHT,
    DEFAULT_TTL,
    MissionBook,
)


class _MockPlanet:
    def __init__(self, pid, owner=0):
        self.id = pid
        self.owner = owner


class _MockWorld:
    def __init__(self, planets):
        self.planets_by_id = {int(p.id): p for p in planets}


def test_reset_on_step_zero():
    book = MissionBook()
    book.commit(src_id=1, target_id=2, score=10.0, step=50, target_owner=-1)
    assert book.size() == 1
    book.reset_if_new_game(step=0)
    assert book.size() == 0


def test_reset_on_step_decrease():
    book = MissionBook()
    book.commit(src_id=1, target_id=2, score=10.0, step=50, target_owner=-1)
    book.reset_if_new_game(step=50)
    assert book.size() == 1
    # Different game in same process — step goes backward.
    book.reset_if_new_game(step=5)
    assert book.size() == 0


def test_carryforward_bonus_positive_for_committed_pair():
    book = MissionBook()
    book.commit(src_id=1, target_id=2, score=10.0, step=50, target_owner=-1)
    bonus = book.carryforward_bonus(src_id=1, target_id=2)
    expected = CARRYFORWARD_WEIGHT * 10.0  # ttl/DEFAULT_TTL = 1.0
    assert abs(bonus - expected) < 1e-9


def test_carryforward_bonus_zero_for_unknown_pair():
    book = MissionBook()
    book.commit(src_id=1, target_id=2, score=10.0, step=50, target_owner=-1)
    assert book.carryforward_bonus(src_id=3, target_id=4) == 0.0


def test_carryforward_bonus_decays_with_ttl():
    book = MissionBook()
    book.commit(src_id=1, target_id=2, score=10.0, step=50,
                target_owner=-1, ttl=DEFAULT_TTL)
    initial = book.carryforward_bonus(1, 2)
    book.decay_ttls()  # TTL: 3 → 2
    second = book.carryforward_bonus(1, 2)
    assert second < initial, f"{second} should be < {initial} after one decay"


def test_decay_drops_expired_commits():
    book = MissionBook()
    book.commit(src_id=1, target_id=2, score=10.0, step=50,
                target_owner=-1, ttl=2)
    book.decay_ttls()  # 2 → 1
    book.decay_ttls()  # 1 → 0 → dropped
    assert book.size() == 0


def test_carryforward_drops_when_src_lost():
    book = MissionBook()
    book.commit(src_id=1, target_id=2, score=10.0, step=50, target_owner=-1)
    # World no longer has planet 1 owned by me.
    world = _MockWorld([_MockPlanet(2, owner=-1)])
    valid = book.carryforward(world, model=None, me=0)
    assert len(valid) == 0
    assert book.size() == 0


def test_carryforward_drops_when_capture_fulfilled():
    book = MissionBook()
    book.commit(src_id=1, target_id=2, score=10.0, step=50,
                target_owner=-1)
    # Both src=1 owned by me=0, and target=2 NOW owned by me=0 — capture
    # fulfilled → commit drops.
    world = _MockWorld([_MockPlanet(1, owner=0), _MockPlanet(2, owner=0)])
    valid = book.carryforward(world, model=None, me=0)
    assert len(valid) == 0
    assert book.size() == 0


def test_carryforward_keeps_valid_commits():
    book = MissionBook()
    book.commit(src_id=1, target_id=2, score=10.0, step=50, target_owner=-1)
    world = _MockWorld([_MockPlanet(1, owner=0),
                        _MockPlanet(2, owner=-1)])
    valid = book.carryforward(world, model=None, me=0)
    assert len(valid) == 1
    assert book.size() == 1


def test_commit_refreshes_existing_ttl():
    book = MissionBook()
    book.commit(src_id=1, target_id=2, score=10.0, step=50, target_owner=-1)
    book.decay_ttls()  # TTL: 3 → 2
    bonus_after_decay = book.carryforward_bonus(1, 2)
    # Re-commit — TTL resets to 3, bonus should increase.
    book.commit(src_id=1, target_id=2, score=10.0, step=51, target_owner=-1)
    bonus_after_recommit = book.carryforward_bonus(1, 2)
    assert bonus_after_recommit > bonus_after_decay
