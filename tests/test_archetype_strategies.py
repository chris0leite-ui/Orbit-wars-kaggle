"""Behavioural tests: does the focal agent play each archetype the way
``lib.archetype_strategy.EXPECTED_BEHAVIOR`` prescribes?

For each of the 32 panel archetypes we run a self-play game on one
representative seed (first in the panel's order for that archetype),
extract a 100-turn behavioural fingerprint + extended-feature dict,
and assert that every metric mentioned in the spec lies in its
expected range.

Archetypes flagged in ``lib.archetype_strategy.KNOWN_REGRESSIONS`` are
marked ``pytest.xfail`` — those cells flipped on the 2026-05-18 A/B vs
v7_0 and document where baseline currently diverges from the spec.
When archetype-aware logic ships and those flip to PASS, that's the
signal we're closing the gap.

Run:
  pytest tests/test_archetype_strategies.py -v
  pytest tests/test_archetype_strategies.py -v -n 8     # parallel
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from kaggle_environments import make

from lib.archetype_strategy import (
    ARCHETYPES,
    KNOWN_REGRESSIONS,
    check,
)
from lib.fingerprint import FEATURE_NAMES, fingerprint
from lib.seed_panel import SEED_PANEL_BY_ARCHETYPE

REPO = Path(__file__).resolve().parents[1]
PREFIX_TURNS = 100  # fingerprint window — short enough to keep tests cheap
EPISODE_STEPS = 200  # truncate games so 32 archetypes run in ~3 min on 8 workers


def _load_tournament_module():
    """scripts/tournament.py provides _build_replay used to convert env -> replay dict."""
    if "tournament" in sys.modules:
        return sys.modules["tournament"]
    spec = importlib.util.spec_from_file_location(
        "tournament", REPO / "scripts" / "tournament.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tournament"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_extended_module():
    """scripts/extended_features.py — replay_extended() for temporal metrics."""
    if "extended_features" in sys.modules:
        return sys.modules["extended_features"]
    spec = importlib.util.spec_from_file_location(
        "extended_features", REPO / "scripts" / "extended_features.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["extended_features"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_baseline_self_play(seed: int, agent_path: str) -> dict[str, float]:
    """Run baseline self-play on ``seed`` and return a flat metric dict.

    ``lib.fingerprint`` and ``scripts.extended_features`` consume different
    replay shapes: fingerprint wants the transformed dict produced by
    ``tournament._build_replay``; replay_extended wants the raw KE
    ``env.steps`` schema (list-of-seat-dicts). We give each what it wants.
    """
    env = make(
        "orbit_wars",
        configuration={"seed": seed, "episodeSteps": EPISODE_STEPS},
        debug=False,
    )
    env.run([agent_path, agent_path])

    tournament = _load_tournament_module()
    extended = _load_extended_module()

    replay = tournament._build_replay(env, seed, "baseline", "baseline")
    n = min(PREFIX_TURNS, len(replay["steps"]))
    fp = fingerprint(replay, player_id=0, prefix_turns=n)
    ext = extended.replay_extended({"steps": env.steps}, 0)

    metrics: dict[str, float] = {name: float(fp[i]) for i, name in enumerate(FEATURE_NAMES)}
    for k, v in ext.items():
        metrics[k] = float(v)
    return metrics


@pytest.fixture(scope="module")
def baseline_path() -> str:
    return str(REPO / "agents" / "baseline" / "main.py")


def _archetype_params() -> list:
    """Build a pytest param list with xfail marks on known-regression cells."""
    out = []
    for arch in ARCHETYPES:
        marks = []
        if arch in KNOWN_REGRESSIONS:
            marks.append(
                pytest.mark.xfail(
                    strict=False,
                    reason="known regression in baseline (A/B vs v7_0, 2026-05-18)",
                )
            )
        out.append(pytest.param(arch, marks=marks, id=arch))
    return out


# ---------------------------------------------------------------------------
# Core: per-archetype behavioural conformance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("archetype", _archetype_params())
def test_archetype_strategy(archetype: str, baseline_path: str) -> None:
    """Baseline self-play on a panel seed must satisfy EXPECTED_BEHAVIOR."""
    seeds = SEED_PANEL_BY_ARCHETYPE[archetype]
    assert seeds, f"empty seed list for archetype {archetype}"
    seed = seeds[0]
    metrics = _run_baseline_self_play(seed, baseline_path)
    violations = check(archetype, metrics)
    assert not violations, (
        f"\narchetype: {archetype}\nseed: {seed}\nviolations:\n  "
        + "\n  ".join(violations)
        + f"\nfull metrics: {metrics}"
    )


# ---------------------------------------------------------------------------
# Spec sanity (cheap; runs without env)
# ---------------------------------------------------------------------------


def test_spec_covers_all_32_archetypes() -> None:
    assert len(ARCHETYPES) == 32
    panel_archs = set(SEED_PANEL_BY_ARCHETYPE.keys())
    spec_archs = set(ARCHETYPES)
    assert spec_archs == panel_archs, (
        f"archetype mismatch:\n  in spec but not in panel: {spec_archs - panel_archs}\n"
        f"  in panel but not in spec: {panel_archs - spec_archs}"
    )


def test_known_regressions_subset_of_archetypes() -> None:
    assert KNOWN_REGRESSIONS.issubset(set(ARCHETYPES))


def test_check_returns_violations_for_obviously_wrong_metrics() -> None:
    """Sanity: `check()` must actually flag a no-launch agent on a
    high_prod archetype (the strictest opening tempo)."""
    # No launches at all → fails first_launch_step + early_launches + multi_launch
    metrics = {
        "first_launch_step": 50,
        "early_launches": 0,
        "launches_per_turn": 0.0,
        "multi_launch_turn_rate": 0.0,
        "mean_fleet_size": 0.0,
    }
    violations = check("high_prod__mostly_static__big_static", metrics)
    assert len(violations) >= 3, f"expected ≥3 violations, got: {violations}"


def test_check_accepts_conforming_metrics() -> None:
    """Sanity: the inverse — pass plausible high_prod conforming metrics."""
    metrics = {
        "first_launch_step": 3,
        "early_launches": 10,
        "launches_per_turn": 2.0,
        "multi_launch_turn_rate": 0.4,
        "mean_fleet_size": 35.0,
        "mean_target_production": 2.0,
        "launch_angle_var": 1.0,
    }
    violations = check("high_prod__mostly_static__big_static", metrics)
    assert not violations, f"unexpected violations: {violations}"
