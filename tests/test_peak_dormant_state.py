"""Pin the *dormant* state at peak (PV_ETA anchor, sub 53111837, μ=1163.5).

`state/PEAK_BASELINE.md` documents a set of env vars whose default values
keep specific code paths inert. A future "cleanup" that flips any of these
defaults to ON without an isolated n=32 A/B is a silent behavior change —
sub 53083109 (μ=921 disaster) is a recorded example of this failure mode.

This test asserts the *module-level* booleans driven by those env vars
are all False when imported with no wrapper preamble in the environment.
It does NOT exercise scoring; it only guards the active-vs-dormant
classification in PEAK_BASELINE.md from drift.

If a future change intentionally flips a default ON, update this test
in the same commit so the doc and code stay in sync.
"""

from __future__ import annotations

import importlib
import os

import pytest


DORMANT_ENV_VARS = (
    # ME-reacts/defends asymmetric scaffold — Fragility #4.
    "BASELINE_ME_REACTS",
    "BASELINE_ME_DEFENDS",
    # Post-chooser pass families — all default OFF; bodies return moves
    # unchanged at main.py:966-969 when disabled.
    "BASELINE_IDLE_DRAIN",
    "BASELINE_STAGNANT_DRAIN",
    "BASELINE_COMBAT_STACK",
    "BASELINE_SNIPER",
    "BASELINE_OPENING_MILP",
    "BASELINE_MACRO",
)


@pytest.fixture
def clean_env(monkeypatch):
    for var in DORMANT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _reload_baseline():
    import agents.baseline.main as bm
    import agents.baseline.chooser_trajectory as ct
    importlib.reload(ct)
    importlib.reload(bm)
    return bm, ct


def test_me_reacts_and_defends_disabled_at_default(clean_env):
    _, ct = _reload_baseline()
    assert ct._ME_REACTS_ENABLED is False
    assert ct._ME_DEFENDS_ENABLED is False, (
        "Enabling BASELINE_ME_DEFENDS without mirroring in the baseline "
        "rollout breaks Δ-symmetry. See Fragility #4 in PEAK_BASELINE.md."
    )


def test_post_chooser_pass_families_disabled_at_default(clean_env):
    bm, _ = _reload_baseline()
    assert bm.IDLE_DRAIN_ENABLED is False
    assert bm.STAGNANT_DRAIN_ENABLED is False
    assert bm.COMBAT_STACK_ENABLED is False
    assert bm.SNIPER_ENABLED is False
    assert bm.OPENING_MILP_ENABLED is False
    assert bm.MACRO_ENABLED is False
