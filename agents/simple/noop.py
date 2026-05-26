"""No-op agent — emits no actions on every turn.

Used as the passive opponent in solo benchmarks where the focal agent's
terminal ship count is the measurement.
"""


def agent(obs):
    return []
