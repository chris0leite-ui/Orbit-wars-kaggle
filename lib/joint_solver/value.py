"""Closed-form value function for the joint LP's column coefficients.

Reuses the W1/W2 / migration value primitives from agents/baseline/
(predicates.py, strategic_lp.py, chooser_lp.py) but DROPS the wait_N==0
single-turn restriction — Phase 3's multi-turn LP needs values for
wait_N>0 columns too.

Layered behaviour:
  - Capture (tgt.owner != me): W1 lower bound (Wald-conservative).
  - Reinforce own with inbound enemy threat: W2 provably-held bound.
  - Migration own→own with no threat: solver's `cheap_delta` (the
    proposer-attached migration EV from agents/baseline/migration_solver).
  - Anything else: 0.

The "closed-form value" semantics match chooser_lp._compute_candidate_value
when wait_N==0; for wait_N>0 the underlying W1/W2 helpers already accept
the wait_N parameter and return correctly scaled values.
"""

from __future__ import annotations

from agents.baseline.predicates import _w1_value_bounds
from agents.baseline.predicates import w2_provably_held_reinforce
from lib.scoring import pv_horizon


EPISODE_END: int = 500
DEFAULT_GAMMA: float = 0.99
W2_VALUE_MULTIPLIER: float = 2.0


def value_for_candidate(c, world, model, *, my_id: int,
                        gamma: float = DEFAULT_GAMMA) -> float:
    """Return the closed-form value for a single prerank candidate.

    `c` is the prerank tuple `(cheap_delta, src, tgt, ships, angle, eta,
    horizon_hint, wait_N)`. `world`/`model` are the current World and
    WorldModel snapshots.

    Differs from chooser_lp._compute_candidate_value in ONE way: the
    `wait_N != 0` early-return is removed, so wait_N>0 candidates get
    their actual W1/W2/migration value rather than a forced 0.
    """
    cheap_delta, src, tgt, ships, _angle, eta, _horizon_hint, wait_N = c

    if int(tgt.owner) == int(my_id):
        # Own→own classification: migration vs defensive reinforce.
        try:
            threat_eta = model.time_to_enemy_threat(int(tgt.id), int(my_id), world)
        except Exception:
            threat_eta = None

        if threat_eta is None:
            # Migration — solver's cheap_delta is the value.
            value = float(cheap_delta)
            return value if value > 0.0 else 0.0

        # Defensive reinforce — W2 verdict.
        try:
            verdict = w2_provably_held_reinforce(
                src, tgt, int(ships), int(wait_N), int(eta),
                world, model, int(my_id),
            )
        except Exception:
            return 0.0
        if verdict.kind != "commit":
            return 0.0
        step = int(getattr(world, "step", 0) or 0)
        arrival = int(wait_N) + int(eta)
        pv = pv_horizon(int(step), int(arrival), gamma=float(gamma),
                        t_total=EPISODE_END)
        return W2_VALUE_MULTIPLIER * float(int(tgt.production)) * float(pv)

    # Capture — bounded interval (lo, hi).
    #
    # Phase 4 (2026-05-20): mid-bound fallback when Wald multi-opp hold
    # check fails. _w1_value_bounds returns (0, hi) in that case, where
    # `hi = production × pv_full` is the "no opp counter" ceiling.
    #
    # Phase 3 used `lo` directly → 60-90% of turns had `no_positive_columns`
    # because Wald fails often. With Stackelberg projection ALREADY baking
    # opp's projected counter-launches into the model (see mpc._model_with_opp_projection),
    # the Wald check is double-counting opp pessimism. The mid-bound `hi/2`
    # is a defensible expected-value proxy: not as conservative as `lo` (which
    # assumes worst-case Wald-coordinated opp counter on top of Stackelberg),
    # not as optimistic as `hi` (which assumes no opp counter at all).
    try:
        lo, hi = _w1_value_bounds(
            src, tgt, int(ships), int(wait_N), int(eta),
            world, model, int(my_id), gamma=float(gamma),
        )
    except Exception:
        return 0.0
    if lo > 0.0:
        return float(lo)
    if hi > 0.0:
        return 0.5 * float(hi)
    return 0.0
