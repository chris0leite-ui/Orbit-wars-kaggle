"""End-to-end smoke: minimal beats random both seats."""

from __future__ import annotations

import pytest

pytest.importorskip("kaggle_environments")
from kaggle_environments import make  # noqa: E402


@pytest.fixture(scope="module")
def minimal_callable():
    from agents.minimal.main import agent
    return agent


def _play(p0, p1, seed: int):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([p0, p1])
    final = env.steps[-1]
    return final[0].reward, final[1].reward


def _winrate(p0, p1, seeds):
    wins = 0.0
    for s in seeds:
        r0, r1 = _play(p0, p1, s)
        if r0 is None or r1 is None:
            continue
        if r0 > r1:
            wins += 1.0
        elif r0 == r1:
            wins += 0.5
    return wins / len(seeds)


def test_minimal_dominates_random(minimal_callable):
    wr = _winrate(minimal_callable, "random", seeds=range(8))
    assert wr >= 0.80, f"minimal vs random winrate = {wr:.2f}; expected >= 0.80"


def test_minimal_beats_random_as_p1_too(minimal_callable):
    wins = 0.0
    seeds = list(range(8))
    for s in seeds:
        r0, r1 = _play("random", minimal_callable, s)
        if r0 is None or r1 is None:
            continue
        if r1 > r0:
            wins += 1.0
        elif r0 == r1:
            wins += 0.5
    wr = wins / len(seeds)
    assert wr >= 0.80, f"random vs minimal (P1) winrate = {wr:.2f}"
