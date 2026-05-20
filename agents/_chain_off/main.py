"""A/B wrapper: baseline agent with BASELINE_CHAIN_BONUS=0 (default).

Local-only — explicit off so the on-wrapper in the same process can't
contaminate via leftover env state.
"""

from __future__ import annotations

import os

from agents.baseline.main import agent as _agent  # noqa: E402


def agent(obs, configuration=None):
    os.environ["BASELINE_CHAIN_BONUS"] = "0"
    return _agent(obs, configuration)
