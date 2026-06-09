"""Tests for scripts/bundle_agent.py — guards against silent bundler regressions.

The bundler is load-bearing: every shipped agent passes through it, and a
silent failure (e.g. forgetting to rebind an `as`-aliased import, or letting
two `from __future__` lines through) would invalidate the submission. These
tests assert the contract end-to-end:

1. Bundler produces a syntactically-valid file.
2. Imported, the bundle exposes `agent(obs)` and runs without crashing.
3. Bundle outcome == original outcome on a fixed seed pair (rewards match).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Load the bundler itself.
_bundler_spec = importlib.util.spec_from_file_location("bundle_agent", SCRIPTS / "bundle_agent.py")
bundle_agent = importlib.util.module_from_spec(_bundler_spec)  # type: ignore[arg-type]
sys.modules["bundle_agent"] = bundle_agent
_bundler_spec.loader.exec_module(bundle_agent)  # type: ignore[union-attr]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Pure-string parsing tests
# ---------------------------------------------------------------------------


def test_extract_aliases_from_simple_aliased_import():
    out = bundle_agent._extract_aliases("from lib.fleet import speed as fleet_speed")
    assert out == [("fleet_speed", "speed")]


def test_extract_aliases_handles_multiple_aliases():
    out = bundle_agent._extract_aliases("from lib.x import a as A, b as B, c")
    assert out == [("A", "a"), ("B", "b")]  # `c` has no alias


def test_extract_aliases_returns_empty_for_no_alias():
    assert bundle_agent._extract_aliases("from lib.geometry import dist") == []


def test_extract_aliases_returns_empty_for_non_import_line():
    assert bundle_agent._extract_aliases("def foo(): pass") == []


# ---------------------------------------------------------------------------
# End-to-end bundle of v1_orbitfix
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bundled_v1(tmp_path_factory):
    """Build the v1 bundle into a tmpdir; return (path, imported module).

    Uses the bundler's DEFAULT_LIB_ORDER so the test catches drift when new
    lib modules are added without being included in the default bundle list.
    """
    out_dir = tmp_path_factory.mktemp("submissions")
    path = bundle_agent.bundle(
        REPO / "agents" / "v1_orbitfix",
        lib_modules=bundle_agent.DEFAULT_LIB_ORDER,
        out_dir=out_dir,
    )
    mod = _load_module("bundled_v1_for_tests", path)
    return path, mod


def test_bundle_file_compiles(bundled_v1):
    path, _ = bundled_v1
    src = path.read_text()
    compile(src, str(path), "exec")  # raises SyntaxError if malformed


def test_bundle_exposes_agent_callable(bundled_v1):
    _, mod = bundled_v1
    assert callable(getattr(mod, "agent", None))


def test_bundle_has_only_one_future_import(bundled_v1):
    """Two `from __future__` lines is a SyntaxError; the bundler must dedup."""
    path, _ = bundled_v1
    src = path.read_text()
    count = sum(1 for line in src.splitlines() if line.strip().startswith("from __future__"))
    assert count == 1


def test_bundle_alias_is_rebound_at_module_level(bundled_v1):
    """`from lib.fleet import speed as fleet_speed` requires a `fleet_speed`
    name in the bundle's module namespace.
    """
    _, mod = bundled_v1
    assert hasattr(mod, "fleet_speed")
    assert callable(mod.fleet_speed)
    assert mod.fleet_speed(1) == 1.0


def test_bundle_outcome_matches_original_on_fixed_seeds(bundled_v1):
    """Bundling must not change game outcomes vs the unbundled agent.
    Two seeds = ~6-15s of game runtime — kept short for the test budget.
    """
    from kaggle_environments import make

    _, bundled = bundled_v1
    orig = _load_module("orig_v1_for_tests", REPO / "agents" / "v1_orbitfix" / "main.py")
    baseline = str(REPO / "data" / "main.py")
    for seed in (42, 1):
        env_b = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env_b.run([bundled.agent, baseline])
        env_o = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env_o.run([orig.agent, baseline])
        rb = [s.reward for s in env_b.steps[-1]]
        ro = [s.reward for s in env_o.steps[-1]]
        assert rb == ro, f"seed={seed}: bundled rewards {rb} != orig {ro}"


def test_bundle_self_play_validation_gate(bundled_v1):
    """E.2: every Kaggle submission runs a self-vs-self validation episode
    before joining the ladder. We replicate that locally — 3 games (smoke-budget)
    must all reach DONE for both players. If this fails, do NOT submit.
    """
    from kaggle_environments import make

    _, bundled = bundled_v1
    for seed in (1000, 1001, 1002):
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([bundled.agent, bundled.agent])
        final = env.steps[-1]
        statuses = [s.status for s in final]
        assert all(s == "DONE" for s in statuses), f"seed={seed}: statuses={statuses}"


# ---------------------------------------------------------------------------
# End-to-end bundle of reach_frontier
#
# Mirrors the bundled_v1 fixture pattern; covers the reach-frontier doctrine
# chooser (knowledge-base/concepts/reach-frontier-doctrine.md). Each phase of
# the build (skeleton -> my-reach -> opp-reach -> assignment) must keep this
# fixture green per Rule 46 (bundle + parity smoke before submission).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bundled_reach_frontier(tmp_path_factory):
    """Build the reach_frontier bundle into a tmpdir; return (path, module)."""
    out_dir = tmp_path_factory.mktemp("submissions_rf")
    path = bundle_agent.bundle(
        REPO / "agents" / "reach_frontier",
        lib_modules=bundle_agent.DEFAULT_LIB_ORDER,
        out_dir=out_dir,
    )
    mod = _load_module("bundled_reach_frontier_for_tests", path)
    return path, mod


def test_reach_frontier_bundle_compiles(bundled_reach_frontier):
    path, _ = bundled_reach_frontier
    src = path.read_text()
    compile(src, str(path), "exec")


def test_reach_frontier_bundle_exposes_agent_callable(bundled_reach_frontier):
    _, mod = bundled_reach_frontier
    assert callable(getattr(mod, "agent", None))


def test_reach_frontier_bundle_has_only_one_future_import(bundled_reach_frontier):
    path, _ = bundled_reach_frontier
    src = path.read_text()
    count = sum(1 for line in src.splitlines()
                if line.strip().startswith("from __future__"))
    assert count == 1


def test_reach_frontier_bundle_outcome_matches_original_on_fixed_seeds(
    bundled_reach_frontier,
):
    """Bundling must not change game outcomes vs the unbundled agent."""
    from kaggle_environments import make

    _, bundled = bundled_reach_frontier
    orig = _load_module(
        "orig_reach_frontier_for_tests",
        REPO / "agents" / "reach_frontier" / "main.py",
    )
    baseline = str(REPO / "data" / "main.py")
    for seed in (42, 1):
        env_b = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env_b.run([bundled.agent, baseline])
        env_o = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env_o.run([orig.agent, baseline])
        rb = [s.reward for s in env_b.steps[-1]]
        ro = [s.reward for s in env_o.steps[-1]]
        assert rb == ro, (
            f"seed={seed}: bundled rewards {rb} != orig {ro}"
        )


def test_reach_frontier_bundle_self_play_validation_gate(bundled_reach_frontier):
    """Self-vs-self DONE-status gate, mirror of E.2 kaggle validation."""
    from kaggle_environments import make

    _, bundled = bundled_reach_frontier
    for seed in (1000, 1001, 1002):
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([bundled.agent, bundled.agent])
        final = env.steps[-1]
        statuses = [s.status for s in final]
        assert all(s == "DONE" for s in statuses), (
            f"seed={seed}: statuses={statuses}"
        )
