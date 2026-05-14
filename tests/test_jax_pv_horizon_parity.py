"""Parity test for PV_GAMMA between scalar (lib.scoring.pv_horizon) and
JAX (lib.game.jax.jax_missions._pv_horizon_jax). Both forms must agree
to 1e-5 for any (step, eta, t_total, gamma) triple, ensuring the JAX
A/B path can drive the same PV configuration as the scalar path.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from lib import scoring
from lib.game.jax import jax_missions


def _scalar_pv(step, eta, t_total, gamma):
    return scoring.pv_horizon(step=step, eta=eta, gamma=gamma, t_total=t_total)


def _jax_pv(step, eta, t_total, gamma):
    return float(
        jax_missions._pv_horizon_jax(
            jnp.int32(step), jnp.int32(eta), jnp.int32(t_total), gamma=gamma,
        )
    )


def test_jax_pv_identity_at_gamma_one_matches_scalar():
    for step, eta in [(0, 5), (100, 20), (400, 10), (499, 1), (500, 0)]:
        s = _scalar_pv(step, eta, 500, gamma=1.0)
        j = _jax_pv(step, eta, 500, gamma=1.0)
        assert s == j, (step, eta, s, j)


def test_jax_pv_geometric_at_gamma_099_matches_scalar():
    for step, eta in [(0, 5), (50, 10), (100, 20), (200, 30), (300, 5)]:
        s = _scalar_pv(step, eta, 500, gamma=0.99)
        j = _jax_pv(step, eta, 500, gamma=0.99)
        assert math.isclose(s, j, rel_tol=1e-5, abs_tol=1e-5), (step, eta, s, j)


def test_jax_pv_zero_at_or_past_game_end():
    for step, eta in [(500, 0), (490, 20), (400, 200)]:
        assert _jax_pv(step, eta, 500, gamma=0.99) == 0.0


def test_jax_pv_comet_case_step_zero():
    # Comet branch uses step=0 and a per-target rem_lifetime in place
    # of EPISODE_STEPS. Confirm the helper handles that without sign
    # issues.
    for eta, rem in [(5, 40), (10, 20), (3, 30), (0, 80)]:
        s = _scalar_pv(0, eta, rem, gamma=0.99)
        j = _jax_pv(0, eta, rem, gamma=0.99)
        assert math.isclose(s, j, rel_tol=1e-5, abs_tol=1e-5), (eta, rem, s, j)


def test_jax_pv_short_horizon_clamps_to_zero():
    # If eta >= t_total, time_to_hold = 0 in both forms.
    assert _jax_pv(0, 100, 50, gamma=0.99) == 0.0
    assert _scalar_pv(0, 100, 50, gamma=0.99) == 0.0
