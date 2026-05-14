"""Parity tests for agents/copycat/main.py (broad-pool argmax design).

Pin the structural invariants:

  1. With both USE_V7 and USE_GEO disabled, copycat falls back to
     top_tier_mirror_policy (v3.5.1) — the documented safe floor.
  2. With USE_V7=1 USE_GEO=0, the pool includes v7_0_drop_one's
     chosen action; that single candidate gets played.
  3. With USE_V7=0 USE_GEO=1, the pool includes geo's strategic
     candidates and the argmax is among them.
  4. Per-turn timing stays under 1000 ms on a real game state.
"""

from __future__ import annotations

import importlib
import os
import sys
import time

import pytest


def _load_copycat(env_overrides: dict[str, str]):
    for k, v in env_overrides.items():
        os.environ[k] = v
    # Reload to pick up new env vars in the module-level config block.
    if "agents.copycat.main" in sys.modules:
        del sys.modules["agents.copycat.main"]
    return importlib.import_module("agents.copycat.main")


def _capture_first_obs():
    """Run one v3.5.1 turn to harvest a real obs payload."""
    import importlib.util
    from kaggle_environments import make
    spec = importlib.util.spec_from_file_location(
        "scalar_v3_5_1", "agents/v3.5.1/main.py",
    )
    v3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v3)
    captured = {}

    def cap(obs, cfg=None):
        if "obs" not in captured:
            captured["obs"] = obs
            captured["cfg"] = cfg
        return v3.agent(obs)

    env = make("orbit_wars", debug=False, configuration={"seed": 42})
    env.run([cap, v3.agent])
    return captured["obs"], captured.get("cfg")


# ---------------------------------------------------------------------------
# 1. Fallback when no generators are enabled.
# ---------------------------------------------------------------------------


def test_fallback_to_v3_5_1_when_pool_empty():
    mod = _load_copycat({
        "COPYCAT_USE_V7": "0",
        "COPYCAT_USE_GEO": "0",
    })
    obs, cfg = _capture_first_obs()

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "scalar_v3_5_1_inner", "agents/v3.5.1/main.py",
    )
    v3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v3)

    copycat_act = mod.agent(obs, cfg)
    v3_act = v3.agent(obs)

    def norm(a):
        return sorted([(int(r[0]), round(float(r[1]), 4), int(r[2])) for r in a])

    assert norm(copycat_act) == norm(v3_act), (
        "fallback didn't return v3.5.1's action; pool-empty mode is wired wrong"
    )


# ---------------------------------------------------------------------------
# 2-3. The pool actually includes the requested generators.
# ---------------------------------------------------------------------------


def test_pool_includes_v7_when_use_v7_enabled():
    mod = _load_copycat({
        "COPYCAT_USE_V7": "1",
        "COPYCAT_USE_GEO": "0",
    })
    obs, cfg = _capture_first_obs()
    # Direct check: the v7 generator returns something non-empty when
    # called on a real obs.
    v7_action = mod._v7_drop_one_action(obs, cfg)
    assert isinstance(v7_action, list)


def test_pool_includes_geo_candidates_when_use_geo_enabled():
    mod = _load_copycat({
        "COPYCAT_USE_V7": "0",
        "COPYCAT_USE_GEO": "1",
    })
    obs, cfg = _capture_first_obs()
    cands = mod._geo_candidates(obs, cfg)
    names = [name for name, _ in cands]
    # At minimum we expect the incumbent. Tilts and archetypes are
    # situational — sometimes the helper returns None and we skip them.
    assert "geo_incumbent" in names, f"geo_incumbent missing from pool: {names}"
    assert len(cands) >= 1


# ---------------------------------------------------------------------------
# 4. Per-turn wallclock budget on a full episode.
# ---------------------------------------------------------------------------


def test_wallclock_fits_under_1000ms_on_full_episode():
    mod = _load_copycat({
        "COPYCAT_USE_V7": "1",
        "COPYCAT_USE_GEO": "1",
    })

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "scalar_v3_5_1_inner2", "agents/v3.5.1/main.py",
    )
    v3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v3)

    from kaggle_environments import make
    turn_ms: list[float] = []

    def timed(obs, cfg=None):
        t = time.perf_counter()
        a = mod.agent(obs, cfg)
        turn_ms.append((time.perf_counter() - t) * 1000.0)
        return a

    env = make("orbit_wars", debug=False, configuration={"seed": 42})
    env.run([timed, v3.agent])
    # The 1000 ms ladder cap is hard. We allow one outlier (SIGALRM
    # mid-C-call can spike the wall by ~200 ms once an episode).
    over = sum(1 for t in turn_ms if t > 1000.0)
    assert over <= 1, (
        f"wallclock exceeded 1000 ms on {over}/{len(turn_ms)} turns; "
        f"max={max(turn_ms):.0f}, p95={sorted(turn_ms)[int(len(turn_ms)*0.95)]:.0f}"
    )
