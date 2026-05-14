"""Parity tests for agents/copycat/main.py.

The copycat agent must satisfy structural invariants before any A/B run
becomes meaningful. These tests pin those invariants:

  1. With COPYCAT_TAU=inf and COPYCAT_ROSTER=v3_5_1, the agent emits
     EXACTLY the v3.5.1 action — the sigma-equivariant floor is
     preserved bit-for-bit.
  2. sigma-paired drops produce candidates that respect the
     pair-structure (no asymmetric singletons when both pair-members
     are firing).
  3. sigma-paired angle perturbations use conjugate signs (+delta /
     -delta) to keep the perturbed action 180-rotationally symmetric.
  4. The bijection cache is keyed per-episode (different planet sets
     get different bijections).
"""

from __future__ import annotations

import importlib
import os

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_module(monkeypatch):
    """Reload copycat.main between tests so env-var overrides apply."""
    # Clear and reset; the test re-imports.
    yield


def _load_copycat(env_overrides: dict[str, str]):
    """Re-import copycat.main with the given env vars."""
    import sys
    for k, v in env_overrides.items():
        os.environ[k] = v
    # Reload to pick up new env vars in the module-level config block.
    if "agents.copycat.main" in sys.modules:
        del sys.modules["agents.copycat.main"]
    mod = importlib.import_module("agents.copycat.main")
    return mod


# ---------------------------------------------------------------------------
# 1. tau=inf singleton parity with v3.5.1
# ---------------------------------------------------------------------------


def _load_scalar_v3_5_1():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "scalar_v3_5_1", "agents/v3.5.1/main.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _norm_action(a):
    return sorted([(int(r[0]), round(float(r[1]), 4), int(r[2])) for r in a])


def test_tau_inf_v3_5_1_singleton_matches_v3_5_1():
    """copycat(tau=inf, roster=v3_5_1) emits the same action as v3.5.1."""
    from kaggle_environments import make

    copycat = _load_copycat({
        "COPYCAT_TAU": "inf",
        "COPYCAT_ROSTER": "v3_5_1",
    })
    v3 = _load_scalar_v3_5_1()

    mismatches = 0
    samples = 0

    captured: list = []
    def cap_agent(seat):
        def f(obs, cfg=None):
            captured.append((seat, obs, cfg))
            return v3.agent(obs)
        return f

    # Generate diverse obs via a real episode.
    env = make("orbit_wars", debug=False, configuration={"seed": 42})
    env.run([cap_agent(0), cap_agent(1)])
    # Sample 20 obs evenly across the episode.
    step = max(1, len(captured) // 20)
    for seat, obs, cfg in captured[::step][:20]:
        # Set the player in case the obs is shared.
        if isinstance(obs, dict):
            obs["player"] = seat
        scalar_act = v3.agent(obs)
        copycat_act = copycat.agent(obs, cfg)
        if _norm_action(scalar_act) != _norm_action(copycat_act):
            mismatches += 1
        samples += 1

    assert samples >= 10, f"too few samples: {samples}"
    assert mismatches == 0, (
        f"copycat(tau=inf, v3_5_1) diverged from scalar v3.5.1 on "
        f"{mismatches}/{samples} obs — sigma-equivariance broken"
    )


# ---------------------------------------------------------------------------
# 2. sigma-paired drops
# ---------------------------------------------------------------------------


def test_sigma_paired_drops_drop_both_pair_members():
    """A sigma-paired drop variant removes BOTH (s, sigma(s)) launches."""
    mod = _load_copycat({})

    # Synthetic action: launches from sources 0, 3, 1, 2.
    # Synthetic bijection: 0<->3, 1<->2.
    action = [
        [0, 0.0, 10],
        [3, 3.14, 10],
        [1, 1.0, 5],
        [2, -2.14, 5],
    ]
    bij = {0: 3, 3: 0, 1: 2, 2: 1}

    variants = mod._sigma_paired_drops(action, bij)

    # We expect two variants: drop {0,3} and drop {1,2}.
    assert len(variants) == 2, (
        f"expected 2 sigma-paired drops, got {len(variants)}"
    )

    # Each variant should remove exactly 2 launches.
    for v in variants:
        assert len(v) == 2, f"variant has {len(v)} launches, expected 2"
        srcs = sorted(int(r[0]) for r in v)
        # The remaining sources should also be a sigma-pair.
        assert (srcs == [0, 3]) or (srcs == [1, 2]), (
            f"variant sources {srcs} are not a sigma-pair — drop broke symmetry"
        )


def test_sigma_paired_drops_no_duplicates_when_singleton_pair():
    """Sources that are sigma-self-mapped should produce only one drop variant."""
    mod = _load_copycat({})

    # Synthetic: source 5 has sigma(5) = 5 (e.g., a fixed point).
    action = [[5, 0.0, 10], [0, 1.0, 5], [3, -2.14, 5]]
    bij = {5: 5, 0: 3, 3: 0}

    variants = mod._sigma_paired_drops(action, bij)

    # Source 5 alone -> drop {5}; pair {0,3} -> drop {0,3}.
    # We expect 2 variants.
    assert len(variants) == 2


# ---------------------------------------------------------------------------
# 3. sigma-paired angle perturbations are conjugate
# ---------------------------------------------------------------------------


def test_sigma_pair_angle_perturb_uses_conjugate_signs():
    """For a sigma-paired pair (s, sigma(s)), the angle nudge is +delta on
    one and -delta on the other — keeping the pair 180-rotationally consistent."""
    mod = _load_copycat({})

    action = [[0, 0.5, 10], [3, 0.5 + 3.14159, 10]]
    bij = {0: 3, 3: 0}

    variants = mod._sigma_pair_angle_perturb(action, bij, delta=0.1)

    assert len(variants) == 1, (
        f"expected 1 angle-perturb variant, got {len(variants)}"
    )
    v = variants[0]
    # Sort by source so we deterministically index.
    by_src = {int(r[0]): r for r in v}
    # Source 0 gets +delta, source 3 gets -delta.
    assert abs(by_src[0][1] - (0.5 + 0.1)) < 1e-9
    assert abs(by_src[3][1] - (0.5 + 3.14159 - 0.1)) < 1e-9


# ---------------------------------------------------------------------------
# 4. Bijection cache
# ---------------------------------------------------------------------------


def test_bijection_cache_keyed_on_planet_ids():
    """Two episodes with different planet rosters get different cached bijections."""
    mod = _load_copycat({})
    mod._BIJECTION_CACHE.clear()

    # Synthetic obs with different planet sets. The 180-deg rotation pivot
    # is (50, 50) per lib.geometry.CENTER.
    # Set A: planets at (10,10) and (90,90) — sigma-pair.
    obs_a = {
        "planets": [
            [0, -1, 10.0, 10.0, 0, 0, 1],
            [1, -1, 90.0, 90.0, 0, 0, 1],
        ],
        "player": 0,
    }
    # Set B: planets at (20,30) and (80,70) — different sigma-pair.
    obs_b = {
        "planets": [
            [10, -1, 20.0, 30.0, 0, 0, 1],
            [11, -1, 80.0, 70.0, 0, 0, 1],
        ],
        "player": 0,
    }

    bij_a = mod._bijection_for(obs_a)
    bij_b = mod._bijection_for(obs_b)

    assert bij_a == {0: 1, 1: 0}
    assert bij_b == {10: 11, 11: 10}
    assert bij_a is not bij_b  # different cache entries
