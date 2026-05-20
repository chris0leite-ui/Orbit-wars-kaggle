"""A/B wrapper: baseline agent with BASELINE_CHAIN_BONUS=1.

Local-only — flips Phase 7 chain-bonus on. Per-call env set, mirroring
_ledger_on pattern so the off-wrapper in the same process can't
contaminate.
"""

from __future__ import annotations

import os

from agents.baseline.main import agent as _agent  # noqa: E402


def agent(obs, configuration=None):
    os.environ["BASELINE_CHAIN_BONUS"] = "1"
    return _agent(obs, configuration)
