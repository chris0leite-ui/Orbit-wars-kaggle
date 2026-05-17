"""End-to-end smoke: baseline beats trivial opponents."""

from __future__ import annotations

import time

import pytest
from kaggle_environments import make


@pytest.fixture(scope="module")
def baseline_callable():
    from agents.baseline.main import agent
    return agent


def _play(p0, p1, seed: int):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([p0, p1])
    final = env.steps[-1]
    return final[0].reward, final[1].reward


def _winrate(p0, p1, seeds):
    """p0_win counts as 1.0, draw as 0.5."""
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


def test_baseline_dominates_random(baseline_callable):
    wr = _winrate(baseline_callable, "random", seeds=range(8))
    assert wr >= 0.80, f"baseline vs random winrate = {wr:.2f}; expected >= 0.80"


def test_baseline_beats_random_as_p1_too(baseline_callable):
    """Seat-symmetric: opp as P0, baseline as P1 — must still dominate."""
    wins = 0.0
    seeds = list(range(8))
    for s in seeds:
        r0, r1 = _play("random", baseline_callable, s)
        if r0 is None or r1 is None:
            continue
        if r1 > r0:
            wins += 1.0
        elif r0 == r1:
            wins += 0.5
    wr = wins / len(seeds)
    assert wr >= 0.80, f"random vs baseline (P1) winrate = {wr:.2f}"


def test_baseline_wallclock_under_budget(baseline_callable):
    """Per-turn p95 < 300 ms, max < 800 ms over one full game (1000 ms is the
    env actTimeout — we want safety margin)."""
    times: list[float] = []

    def timed(obs, cfg):
        t0 = time.perf_counter()
        try:
            return baseline_callable(obs, cfg)
        finally:
            times.append((time.perf_counter() - t0) * 1000.0)

    env = make("orbit_wars", configuration={"seed": 13}, debug=False)
    env.run([timed, "random"])
    assert times, "no turns recorded"
    times_sorted = sorted(times)
    p95 = times_sorted[max(0, int(0.95 * (len(times_sorted) - 1)))]
    p_max = times_sorted[-1]
    assert p95 < 300.0, f"p95 = {p95:.0f}ms (target < 300)"
    assert p_max < 800.0, f"max = {p_max:.0f}ms (target < 800)"
