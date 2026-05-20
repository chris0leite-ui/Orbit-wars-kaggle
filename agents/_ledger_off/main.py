"""A/B wrapper: baseline agent with BASELINE_LEDGER=off.

Mirrors the bundled-champion behaviour (52827111). Exists only for local
A/B testing.
"""

from __future__ import annotations

import os

from agents.baseline.main import agent as _agent  # noqa: E402


def agent(obs, configuration=None):
    os.environ["BASELINE_LEDGER"] = "off"
    return _agent(obs, configuration)
