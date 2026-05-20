"""MPC wrapper: drop wait_N>0 candidates entirely from the chooser.

Pure receding-horizon: only fire-now plans are scored. The simpler
alternative to the stateful ledger; tested in the what-if rollout
where it beat baseline on final-planet count. Validate in h2h.
"""
import os
from agents.baseline.main import agent as _agent
import agents.baseline.chooser_trajectory as CHOOSER

# Patch score_candidate_v4 to reject wait_N>0 once at module-import time.
# This affects ALL subsequent calls through this module. Since the mpc
# wrapper is the only consumer here, this is fine — the off/on/hard
# wrappers in this dir don't share state.
_orig_score = CHOOSER.score_candidate_v4

def _score_no_wait(snap_base, src, tgt, ships, angle, me, num_seats, world,
                   baseline_favors, favor_fn, gamma, horizon,
                   skip_admissibility=False, wait_N=0):
    if int(wait_N) != 0:
        return (float("-inf"), "skipped_wait", 0)
    return _orig_score(snap_base, src, tgt, ships, angle, me, num_seats, world,
                       baseline_favors, favor_fn, gamma, horizon,
                       skip_admissibility=skip_admissibility, wait_N=wait_N)


def agent(obs, configuration=None):
    os.environ["BASELINE_LEDGER"] = "off"
    # Hot-patch on each call (defensive against other wrappers in same
    # process resetting the function).
    CHOOSER.score_candidate_v4 = _score_no_wait
    return _agent(obs, configuration)
