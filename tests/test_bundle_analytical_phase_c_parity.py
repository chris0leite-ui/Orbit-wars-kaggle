"""Bug #5 — bundle-vs-source per-turn parity.

`submissions/analytical_phase_c.py` is produced by
`scripts/bundle_analytical_phase_c.py` with import-strip + sys.modules
self-registration shim. The bundle MUST emit identical moves to the
source agent (`agents/analytical_phase_c/main.py`) on every turn.

Submission 52863735 ERRORed on Kaggle because the bundle was
behaviourally divergent (missing sys.modules shim) and no test caught
it — by the time we noticed, the slot was burned. This test pins the
parity so any future divergence fails locally before a push.

Pattern adapted from `tests/test_pipeline_parity.py:_make_dual_agent`:
a per-turn wrapper that calls both the source and the bundle, normalises
the move lists `[src_id, angle, ships]`, and records any divergence.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUNDLE_PATH = REPO / "submissions" / "analytical_phase_c.py"


def _load_bundle_agent():
    """Load the bundled file as a Python module and return its agent.

    We register the module in sys.modules before exec_module so the
    bundle's dataclass introspection (Python 3.11 KW_ONLY check) finds
    a valid entry — that's the same role the bundle's sys.modules shim
    plays on Kaggle.
    """
    if not BUNDLE_PATH.exists():
        pytest.skip(f"bundle not present at {BUNDLE_PATH}")
    mod_name = "analytical_phase_c_bundle_under_test"
    spec = importlib.util.spec_from_file_location(mod_name, str(BUNDLE_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module.agent


def _make_dual_agent_bundle_vs_source():
    """Wrapper that calls source and bundle on the same obs each turn."""
    bundle_agent = _load_bundle_agent()
    from agents.analytical_phase_c.main import agent as source_agent

    discrepancies: list[tuple] = []

    def dual(obs, configuration=None):
        src_moves = source_agent(obs, configuration)
        bun_moves = bundle_agent(obs, configuration)
        src_norm = [
            [int(m[0]), float(m[1]), int(m[2])] for m in (src_moves or [])
        ]
        bun_norm = [
            [int(m[0]), float(m[1]), int(m[2])] for m in (bun_moves or [])
        ]
        if src_norm != bun_norm:
            step = int(obs.get("step", -1)) if isinstance(obs, dict) else -1
            discrepancies.append((step, src_norm, bun_norm))
        return bun_moves

    return dual, discrepancies


@pytest.mark.parametrize("seed", [42, 7])
def test_bundle_phase_c_parity_short(seed: int):
    """50-turn smoke: bundle == source per-turn bit-exact."""
    from kaggle_environments import make

    env = make("orbit_wars", configuration={
        "seed": seed, "episodeSteps": 50,
    }, debug=False)
    dual, discrepancies = _make_dual_agent_bundle_vs_source()
    try:
        env.run([dual, "agents/simple/nearest.py"])
    except Exception as e:
        pytest.fail(
            f"env.run raised: {e}; first divergences: {discrepancies[:3]}"
        )
    assert not discrepancies, (
        f"seed {seed}: {len(discrepancies)} divergent turn(s); "
        f"first 3 (step, source, bundle): {discrepancies[:3]}"
    )


@pytest.mark.parametrize("seed", [42])
def test_bundle_phase_c_parity_full(seed: int):
    """Full-length game: zero per-turn divergence between bundle and source.

    Slower (~30-60s). Covers turns past the opening window where
    persistent-schedule decants fire — those are the steps most likely
    to expose subtle bundling drift.
    """
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    dual, discrepancies = _make_dual_agent_bundle_vs_source()
    try:
        env.run([dual, "agents/simple/nearest.py"])
    except Exception as e:
        pytest.fail(
            f"env.run raised: {e}; first divergences: {discrepancies[:3]}"
        )
    assert not discrepancies, (
        f"seed {seed} full game: {len(discrepancies)} divergent turn(s); "
        f"first 3 (step, source, bundle): {discrepancies[:3]}"
    )
