"""Pin the wrapper preamble of the live champion (PV_ETA anchor).

`submissions/baseline_pv_eta_anchor_1163.py` is the frozen bundle for
sub 53111837 (μ=1163.5, SHA-256 prefix `7964bfa4`). Its first ~17 lines
set 10 env-var setdefaults that drive the *active* surface of the agent.

`state/PEAK_BASELINE.md` previously misclassified four of these as
"DORMANT" — that was true before sub 53083109's plumbing landed, but it
has been incorrect ever since. The wrapper values are now load-bearing:

  - BASELINE_NEUTRAL_BONUS=2.0
  - BASELINE_NEUTRAL_EARLY_HORIZON=50
  - BASELINE_NEUTRAL_EARLY_EXTRA=1.5
  - BASELINE_PV_ETA=1

Stripping any of these as "clean up dormant env vars" silently changes
the agent's scoring behavior. This test fails loudly if anyone touches
the frozen anchor without updating the doc first.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest


ANCHOR = (
    pathlib.Path(__file__).parent.parent
    / "submissions"
    / "baseline_pv_eta_anchor_1163.py"
)
ANCHOR_TRACKED_TWIN = (
    pathlib.Path(__file__).parent.parent
    / "submissions"
    / "baseline_pv_eta.py"
)

# Frozen SHA — see state/PEAK_BASELINE.md.
EXPECTED_SHA = (
    "7964bfa4b0ceaef7942c515179fbd549687aec2db1faf1baedb7016a23e6dfff"
)

REQUIRED_SETDEFAULTS = (
    ('BASELINE_JOINT_AGGR', '1'),
    ('BASELINE_JOINT_TOP_K', '5'),
    ('BASELINE_JOINT_MAX_PAIRS', '60'),
    ('BASELINE_REINFORCE_EMIT', '1'),
    ('BASELINE_REINFORCE_ANTICIPATE', '1'),
    ('BASELINE_NEUTRAL_BONUS', '2.0'),
    ('BASELINE_NEUTRAL_EARLY_EXTRA', '1.5'),
    ('BASELINE_NEUTRAL_EARLY_HORIZON', '50'),
    ('BASELINE_ORBITAL_SAFETY', '1'),
    ('BASELINE_PV_ETA', '1'),
)


@pytest.fixture(scope="module")
def anchor_text():
    return ANCHOR.read_text()


def test_anchor_sha_matches_frozen():
    assert ANCHOR.exists(), f"Frozen anchor missing: {ANCHOR}"
    digest = hashlib.sha256(ANCHOR.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA, (
        f"Anchor SHA drifted from {EXPECTED_SHA[:8]}… to {digest[:8]}…; "
        "the frozen bundle was modified. Restore from "
        "submissions/baseline_pv_eta.py (same bytes) or "
        "`git checkout HEAD -- submissions/baseline_pv_eta_anchor_1163.py`."
    )


def test_tracked_twin_matches_anchor_bytes():
    """baseline_pv_eta.py is the tracked twin; both must be byte-identical."""
    assert ANCHOR_TRACKED_TWIN.exists()
    assert (
        hashlib.sha256(ANCHOR_TRACKED_TWIN.read_bytes()).hexdigest()
        == EXPECTED_SHA
    ), "baseline_pv_eta.py drifted from the frozen anchor."


@pytest.mark.parametrize("name,value", REQUIRED_SETDEFAULTS)
def test_wrapper_setdefault_present(anchor_text, name, value):
    needle = f'environ.setdefault("{name}", "{value}")'
    assert needle in anchor_text, (
        f"Wrapper missing `{needle}`. See PEAK_BASELINE.md: this env var "
        f"is documented as ACTIVE — stripping it changes agent behavior."
    )
