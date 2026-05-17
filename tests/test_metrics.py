"""Unit tests for lib.metrics — pin every baseline.

The metric library is the canonical implementation that every
pre-submit hypothesis names by string. If a metric drifts, these
tests fail and we know the calibration table needs to be revisited.

Each metric has at least one test:
- Rollup-based: verify the formula against a hand-built rollup whose
  numbers reproduce a published baseline.
- Replay-walking: verify the formula against a hand-built replay
  fixture.
"""

from __future__ import annotations

import json
import math

import pytest

from lib import metrics


# ---------------------------------------------------------------------------
# Helper: synthesise a rollup that reproduces the v15 baseline (sub
# 52710995, 9507 fleets, audit/replays/replay-mine-2026-05-17.json).
# ---------------------------------------------------------------------------


def _v15_rollup_fixture() -> dict:
    """Hand-built rollup matching the v15 baseline numbers exactly.

    Source: audit/replays/replay-mine-2026-05-17.{md,json} —
    win 42.8% / defense 32.8% / waste_attack 14.7% / waste_comet 0.1%
    / waste_trajectory 9.0% / inflight 0.6% / unknown 0.0%.
    """
    n = 9507  # total fleets across 92 episodes
    counts = {
        "win":              4069,   # 42.8%
        "defense":          3118,   # 32.8%
        "waste_attack":     1398,   # 14.7%
        "waste_comet":      10,     # ~0.1%
        "waste_trajectory": 856,    # 9.0%
        "inflight":         55,     # ~0.6%
        "unknown":          1,
    }
    # Adjust the count so totals match exactly.
    diff = n - sum(counts.values())
    counts["unknown"] += diff
    raw_outcomes = {
        # Inside waste_trajectory: sun + oob + vanished_in_space.
        "sun": 60,                  # rough; not in published table
        "oob": 100,
        "vanished_in_space": 696,
        # Inside waste_attack
        "bounced_neutral": 800,
        "bounced_enemy": 500,
        "arrived_but_lost": 98,
        # Inside win
        "captured": 4069,
        "reinforced_self": 3118,
        "comet_collision": 10,
        "alive_at_end": 55,
        "hit_planet_unknown_flip": 1,
    }
    return {
        "n_episodes": 92,
        "n_fleets": n,
        "n_ships_launched": 100000,  # not material to bucket fractions
        "raw_outcomes": raw_outcomes,
        "by_bucket": counts,
        "ships_by_bucket": {k: v * 10 for k, v in counts.items()},
        "pct_by_bucket": {k: round(100 * v / n, 1) for k, v in counts.items()},
    }


# ---------------------------------------------------------------------------
# Rollup metrics — pin against v15 baseline within tight tolerance
# ---------------------------------------------------------------------------


def test_win_fraction_matches_v15():
    r = _v15_rollup_fixture()
    assert math.isclose(metrics.win_fraction(r),
                        metrics.V15_BASELINE["win_fraction"], abs_tol=0.005)


def test_defense_fraction_matches_v15():
    r = _v15_rollup_fixture()
    assert math.isclose(metrics.defense_fraction(r),
                        metrics.V15_BASELINE["defense_fraction"], abs_tol=0.005)


def test_waste_attack_fraction_matches_v15():
    r = _v15_rollup_fixture()
    assert math.isclose(metrics.waste_attack_fraction(r),
                        metrics.V15_BASELINE["waste_attack_fraction"],
                        abs_tol=0.005)


def test_trajectory_waste_fraction_matches_v15():
    r = _v15_rollup_fixture()
    assert math.isclose(metrics.trajectory_waste_fraction(r),
                        metrics.V15_BASELINE["trajectory_waste_fraction"],
                        abs_tol=0.005)


def test_waste_comet_fraction_matches_v15():
    r = _v15_rollup_fixture()
    # Tolerance widened because the published v15 number rounds to 0.1%.
    assert math.isclose(metrics.waste_comet_fraction(r),
                        metrics.V15_BASELINE["waste_comet_fraction"],
                        abs_tol=0.002)


def test_inflight_fraction_matches_v15():
    r = _v15_rollup_fixture()
    assert math.isclose(metrics.inflight_fraction(r),
                        metrics.V15_BASELINE["inflight_fraction"],
                        abs_tol=0.005)


def test_sun_clip_rate_reads_raw_outcomes():
    """sun_clip_rate is narrower than trajectory_waste — it's just the
    `sun` raw outcome, not OOB or vanished."""
    r = _v15_rollup_fixture()
    val = metrics.sun_clip_rate(r)
    # 60 sun deaths / 9507 fleets ≈ 0.0063
    assert math.isclose(val, 60 / 9507, abs_tol=0.001)
    # And it's strictly less than trajectory_waste_fraction (which
    # ALSO includes oob + vanished).
    assert val < metrics.trajectory_waste_fraction(r)


def test_comet_kill_rate_equals_waste_comet_fraction():
    """comet_kill_rate is currently the same as waste_comet_fraction
    (until ray-casting lands and we can distinguish 'aimed at comet'
    vs 'killed by comet')."""
    r = _v15_rollup_fixture()
    assert metrics.comet_kill_rate(r) == metrics.waste_comet_fraction(r)


def test_empty_rollup_returns_zero():
    """All rollup metrics handle the empty / no-fleets case gracefully."""
    empty = {"n_fleets": 0, "by_bucket": {}, "raw_outcomes": {}}
    for name in metrics._ROLLUP_METRICS:
        f = metrics.get_metric(name)
        assert f(empty) == 0.0, f"{name} did not return 0.0 on empty rollup"


def test_unknown_metric_raises():
    with pytest.raises(KeyError):
        metrics.get_metric("nonexistent_metric")


# ---------------------------------------------------------------------------
# Replay-walking metrics — hand-built fixtures
# ---------------------------------------------------------------------------


def _make_replay(team_name: str, our_seat: int, num_seats: int,
                 actions_by_turn: dict[int, list]) -> dict:
    """Build a minimal replay JSON skeleton.

    `actions_by_turn` maps turn_idx → our list of [src, angle, ships]
    actions on that turn. Other seats get [].
    """
    teams = ["other"] * num_seats
    teams[our_seat] = team_name
    # Build a 'steps' array up to the highest turn referenced. Each
    # step is a list-of-seats; each seat has at minimum
    # {action, status}.
    max_turn = max(actions_by_turn.keys()) if actions_by_turn else 0
    steps = []
    for t in range(max_turn + 2):  # +1 for the post-DONE step
        per_seat = []
        for seat in range(num_seats):
            seat_action = actions_by_turn.get(t, []) if seat == our_seat else []
            per_seat.append({"action": seat_action, "status": "ACTIVE"})
        steps.append(per_seat)
    return {
        "info": {"TeamNames": teams},
        "steps": steps,
    }


def test_first_launch_step_top10_like():
    """Top-10 mean first-launch step is 4.1. Build 3 replays with first
    launches at steps 3, 4, 5 → mean 4.0."""
    rep_a = _make_replay("us", 0, 2, {3: [[0, 0.0, 10]]})
    rep_b = _make_replay("us", 0, 2, {4: [[0, 0.0, 10]]})
    rep_c = _make_replay("us", 0, 2, {5: [[0, 0.0, 10]]})
    out = metrics.first_launch_step([(rep_a, "us"), (rep_b, "us"),
                                     (rep_c, "us")])
    assert out == pytest.approx(4.0, abs=0.01)


def test_first_launch_step_default_on_silent_game():
    """A replay with no launches contributes `default` (500)."""
    silent = _make_replay("us", 0, 2, {})  # no actions
    active = _make_replay("us", 0, 2, {1: [[0, 0.0, 5]]})
    out = metrics.first_launch_step([(silent, "us"), (active, "us")])
    # (500 + 1) / 2 = 250.5
    assert out == pytest.approx(250.5, abs=0.1)


def test_first_launch_step_unknown_team_skips():
    """If our team isn't in TeamNames, we skip the episode."""
    rep = _make_replay("them", 0, 2, {3: [[0, 0.0, 10]]})
    # 'us' doesn't appear → no episodes contributed → returns default.
    out = metrics.first_launch_step([(rep, "us")], default=42)
    assert out == 42


def test_active_turn_fraction():
    """3 launches across 10 turns → 0.3."""
    rep = _make_replay("us", 0, 2, {
        2: [[0, 0.0, 5]],
        4: [[0, 0.0, 5]],
        7: [[0, 0.0, 5]],
    })
    # Need to make total turns = 10 (build replay with 10 steps).
    # _make_replay already gives max_turn+2 = 9 steps; pad one more.
    rep["steps"].append([{"action": [], "status": "ACTIVE"},
                         {"action": [], "status": "ACTIVE"}])
    # Now 10 steps. 3 with actions.
    out = metrics.active_turn_fraction([(rep, "us")])
    assert out == pytest.approx(3 / 10, abs=0.01)


def test_active_turn_fraction_stops_at_done():
    """A seat that gets DONE at turn 5 doesn't accumulate later turns."""
    rep = _make_replay("us", 0, 2, {1: [[0, 0.0, 5]], 3: [[0, 0.0, 5]]})
    # Mark turn 5 onwards as DONE for our seat.
    for t in range(5, len(rep["steps"])):
        rep["steps"][t][0]["status"] = "DONE"
    out = metrics.active_turn_fraction([(rep, "us")])
    # 2 active / 5 turns = 0.4
    assert out == pytest.approx(2 / 5, abs=0.01)


def test_multi_launch_turn_rate():
    """2 of 3 active turns have ≥2 launches → 0.667."""
    rep = _make_replay("us", 0, 2, {
        1: [[0, 0.0, 5]],                    # 1 launch
        2: [[0, 0.0, 5], [1, 0.0, 5]],       # 2 launches
        4: [[0, 0.0, 5], [1, 0.0, 5], [2, 0.0, 5]],  # 3 launches
    })
    out = metrics.multi_launch_turn_rate([(rep, "us")])
    assert out == pytest.approx(2 / 3, abs=0.01)


def test_multi_launch_turn_rate_no_active_returns_zero():
    rep = _make_replay("us", 0, 2, {})
    assert metrics.multi_launch_turn_rate([(rep, "us")]) == 0.0


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


def test_list_metrics_is_sorted_and_unique():
    names = metrics.list_metrics()
    assert names == sorted(names)
    assert len(names) == len(set(names))
    # We expect at least the 9 metrics implemented in this commit.
    assert len(names) >= 9


def test_rollup_vs_replay_partition_is_disjoint():
    """No metric is registered as BOTH rollup and replay."""
    rollup = set(metrics._ROLLUP_METRICS)
    replay = set(metrics._REPLAY_METRICS)
    assert not (rollup & replay)


def test_get_metric_dispatches_by_name():
    f = metrics.get_metric("win_fraction")
    assert f is metrics.win_fraction
    g = metrics.get_metric("first_launch_step")
    assert g is metrics.first_launch_step


def test_baseline_lookup():
    assert metrics.baseline("win_fraction", "v15") == 0.428
    assert metrics.baseline("first_launch_step", "top10") == 4.1
    assert metrics.baseline("first_launch_step", "midpack") == 10.5
    # Metric without a baseline for that source → None.
    assert metrics.baseline("first_launch_step", "v15") is None
    # Unknown source raises.
    with pytest.raises(KeyError):
        metrics.baseline("win_fraction", "nonexistent_source")
