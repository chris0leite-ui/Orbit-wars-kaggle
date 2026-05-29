"""Document the ME-reacts/defends symmetry contract (Fragility #4).

`agents/baseline/chooser_trajectory.py` has parallel "what the agent
does after the candidate fires" logic in two places:

  - The CANDIDATE-rollout leg (`:523-540`, `:579-583`, `:666-680`),
    gated on `_ME_REACTS_ENABLED` / `_ME_DEFENDS_ENABLED`.
  - The BASELINE-rollout leg (`build_trajectory_baseline`, `:481-498`),
    which does NOT have the same gates.

At peak both flags are False (see test_peak_dormant_state.py), so
Δ = leaf − baseline is well-defined. If a future change enables
`BASELINE_ME_DEFENDS=1` (or `_REACTS=1`) WITHOUT mirroring the same
defensive/reactive launches in the baseline leg, Δ silently changes
meaning — previously-positive candidates can flip negative.

This file pins the contract: if either flag is True, the baseline
rollout MUST mirror the candidate rollout's defensive/reactive
behavior. The current code does not provide that mirror, so the
test is xfail until the mirror is implemented (or the flags are
removed).

Pulling this test from xfail to passing is the gate for ever
enabling these flags in production.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Fragility #4 in state/PEAK_BASELINE.md: candidate rollout has "
        "_ME_REACTS/DEFENDS gates that baseline rollout lacks. If either "
        "flag is enabled in production, build_trajectory_baseline must be "
        "updated to mirror the candidate's defensive/reactive launches."
    ),
)
def test_me_defends_baseline_mirror_exists():
    """Static contract: if `_ME_DEFENDS` is wired into the candidate
    rollout, `build_trajectory_baseline` must reference it too.

    Currently the baseline leg ignores both flags entirely. This
    asserts the *intended future* symmetry by grep — when the mirror
    lands the test passes; until then it xfails as a documented contract.
    """
    import agents.baseline.chooser_trajectory as ct
    importlib.reload(ct)

    source = pathlib_text(ct.__file__)
    # Locate build_trajectory_baseline's body and verify it gates on the
    # same flags as the candidate-rollout leg.
    baseline_start = source.find("def build_trajectory_baseline")
    assert baseline_start != -1, "build_trajectory_baseline disappeared"
    baseline_end = source.find("\ndef ", baseline_start + 1)
    baseline_body = source[baseline_start:baseline_end]

    assert "_ME_DEFENDS_ENABLED" in baseline_body, (
        "build_trajectory_baseline does not reference _ME_DEFENDS_ENABLED. "
        "Enabling BASELINE_ME_DEFENDS=1 elsewhere will break Δ-symmetry."
    )
    assert "_ME_REACTS_ENABLED" in baseline_body, (
        "build_trajectory_baseline does not reference _ME_REACTS_ENABLED. "
        "Enabling BASELINE_ME_REACTS=1 elsewhere will break Δ-symmetry."
    )


def test_me_flags_off_at_peak_is_enforced():
    """The above xfail is acceptable ONLY because both flags are off at
    peak. This test guards that precondition; if either flag's default
    flips, the xfail above becomes a real failure mode immediately."""
    import os
    assert os.environ.get("BASELINE_ME_REACTS", "0") == "0", (
        "BASELINE_ME_REACTS is set; Fragility #4 is now active. Either "
        "wire the mirror in build_trajectory_baseline, or unset the env var."
    )
    assert os.environ.get("BASELINE_ME_DEFENDS", "0") == "0", (
        "BASELINE_ME_DEFENDS is set; Fragility #4 is now active. Either "
        "wire the mirror in build_trajectory_baseline, or unset the env var."
    )


def pathlib_text(path):
    import pathlib
    return pathlib.Path(path).read_text()
