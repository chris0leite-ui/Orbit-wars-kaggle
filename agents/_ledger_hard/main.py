"""Hard-mode ledger wrapper: src reserved across the wait window."""
import os
from agents.baseline.main import agent as _agent

def agent(obs, configuration=None):
    os.environ["BASELINE_LEDGER"] = "on"
    os.environ["BASELINE_LEDGER_MODE"] = "hard"
    return _agent(obs, configuration)
