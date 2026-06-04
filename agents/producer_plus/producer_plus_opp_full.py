"""All-opp-foresight-ON variant of ``producer_plus``.

Currently sets: opp projector + source exposure (M1) + counter capture
(M3). Race-loss (M2) was ablated after n=8 evidence showed it
double-penalises candidates the augmented scorer had already correctly
discounted. M3 stays wired but is currently a NET REGRESSION (2/8 in
n=8 triage) — keeping the env gate available for the future
multi-size / wait_N axis port that gives it ETA control. For the
clean shipping candidate, use ``producer_plus_source_exposure`` (M1
only). See ``state/MIGRATION_PLAN.md`` opp-foresight plan.
"""
import os
import importlib.util

os.environ.setdefault("PRODUCER_PLUS_OPP_PROJECTOR", "lite_greedy")
os.environ.setdefault("PRODUCER_PLUS_SOURCE_EXPOSURE", "1")
os.environ.setdefault("PRODUCER_PLUS_COUNTER_CAPTURE", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "producer_plus_opp_full_inner",
    os.path.join(_HERE, "producer_agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def agent(obs, configuration=None):
    return _module.agent(obs, configuration)
