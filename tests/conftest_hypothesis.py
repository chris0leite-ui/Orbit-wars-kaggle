"""Hypothesis profile setup shared by the foundation fuzz suite.

Why a shared profile: `tests/test_fast_sim_parity.py` uses fixed seeds
and is in the fast lane. The foundation fuzz suite
(`tests/test_foundation_parity_fuzz.py`) is the random-search
complement and needs derandomized, CI-deterministic Hypothesis
settings so two CI runs on the same commit produce the same
example set.

This file is imported by `tests/test_foundation_parity_fuzz.py`
explicitly (not via conftest auto-discovery) so it doesn't pollute
the rest of the test suite's Hypothesis defaults.
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings

# CI profile: derandomised (deterministic example order), no deadline
# (env.step is slow), suppress data-too-large health checks (we
# intentionally build full-size game states).
settings.register_profile(
    "foundation_fuzz_ci",
    max_examples=20,
    deadline=None,
    derandomize=True,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.function_scoped_fixture,
    ],
)

# Nightly profile: 10× the examples for the slow lane.
settings.register_profile(
    "foundation_fuzz_nightly",
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.function_scoped_fixture,
    ],
)
