"""A/B wrapper: baseline agent with BASELINE_LEDGER=on, MODE=soft.

Exists only for local A/B testing. Production uses bundled
submissions/baseline.py with env-var defaults set at bundle time.

The wrapper resets env vars on EVERY call (rather than once at import)
because the agent module's module-level dict (`_PENDING_LAUNCHES`) is
shared with the off-wrapper when both are imported in the same process,
and the agent reads `BASELINE_LEDGER` at call time as a safety check.
Setting per-call ensures this wrapper always sees ledger=on regardless
of what the other wrapper did last.
"""

from __future__ import annotations

import os

from agents.baseline.main import agent as _agent  # noqa: E402


def agent(obs, configuration=None):
    os.environ["BASELINE_LEDGER"] = "on"
    os.environ["BASELINE_LEDGER_MODE"] = "soft"
    return _agent(obs, configuration)
