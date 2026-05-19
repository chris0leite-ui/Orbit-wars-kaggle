"""Unit + integration tests for the v3_snipe-style ROI cost denominator
on `cheap_marginal_value`.

See plan/ROI plan for the formula:
    capture / reinforce:  base / (ships + arrival_step + roi_denom_floor)
    bounce:               unchanged (-0.5 * ships)
"""
from __future__ import annotations

import os

import pytest

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from agents.baseline.proposer import (
    cheap_marginal_value,
    propose,
)
from lib.intent import World
from lib.world_model import WorldModel


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    return Planet(pid, owner, x, y, radius, ships, production)


def _world_for(my_id, planets, *, step=0, omega=0.0):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": [],
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "step": step,
    }
    return World.from_obs(obs)


# ---------- function-level byte-identity at roi_enabled=False ----------

def test_cheap_marginal_value_roi_off_matches_pre_pivot():
    """roi_enabled=False keeps the pre-pivot formula exactly. Sweeps a
    grid of (ships, eta) at one (src, tgt). Function default is
    roi_enabled=False so an explicit kwarg-absent call must also match."""
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=3)
    tgt = _planet(1, -1, 14.0, 50.0, ships=5, production=2)
    world = _world_for(0, [src, tgt])
    model = WorldModel.from_world(world)

    for ships in (3, 7, 12, 20):
        for eta in (1, 5, 12):
            v_off = cheap_marginal_value(
                src, tgt, ships, eta, world, model, me=0, wait_N=0,
                roi_enabled=False,
            )
            v_default = cheap_marginal_value(
                src, tgt, ships, eta, world, model, me=0, wait_N=0,
            )
            assert v_off == v_default, (
                f"ships={ships} eta={eta}: off={v_off} default={v_default}"
            )


def test_cheap_marginal_value_roi_divides_positive_base():
    """For a capture (positive base), ROI on must equal off / (ships + eta + floor)."""
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=3)
    tgt = _planet(1, -1, 14.0, 50.0, ships=5, production=2)  # capture, neutral
    world = _world_for(0, [src, tgt])
    model = WorldModel.from_world(world)

    ships, eta = 12, 5
    base_off = cheap_marginal_value(
        src, tgt, ships, eta, world, model, me=0, wait_N=0,
        roi_enabled=False,
    )
    assert base_off > 0, "test setup precondition: capture base must be positive"

    base_on = cheap_marginal_value(
        src, tgt, ships, eta, world, model, me=0, wait_N=0,
        roi_enabled=True, roi_denom_floor=1.0,
    )
    expected = base_off / (ships + eta + 1.0)
    assert base_on == pytest.approx(expected, rel=1e-9)


def test_cheap_marginal_value_roi_denom_floor_softens():
    """A higher denominator floor shrinks the ROI division impact."""
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=3)
    tgt = _planet(1, -1, 14.0, 50.0, ships=5, production=2)
    world = _world_for(0, [src, tgt])
    model = WorldModel.from_world(world)

    ships, eta = 12, 5
    floor_1 = cheap_marginal_value(
        src, tgt, ships, eta, world, model, me=0, wait_N=0,
        roi_enabled=True, roi_denom_floor=1.0,
    )
    floor_100 = cheap_marginal_value(
        src, tgt, ships, eta, world, model, me=0, wait_N=0,
        roi_enabled=True, roi_denom_floor=100.0,
    )
    base_off = cheap_marginal_value(
        src, tgt, ships, eta, world, model, me=0, wait_N=0,
        roi_enabled=False,
    )
    # floor_100 divides by larger denom, so it's smaller than floor_1.
    assert 0 < floor_100 < floor_1 < base_off


def test_cheap_marginal_value_roi_does_not_touch_bounces():
    """Bounce branch (`-0.5 * ships`) must be untouched by ROI division.
    Verified by constructing a target whose pred_ships > attacker ships."""
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=3)
    # Heavily-garrisoned neutral: attacker can never out-ship.
    tgt = _planet(1, -1, 14.0, 50.0, ships=200, production=1)
    world = _world_for(0, [src, tgt])
    model = WorldModel.from_world(world)

    ships, eta = 5, 5
    v_off = cheap_marginal_value(
        src, tgt, ships, eta, world, model, me=0, wait_N=0,
        roi_enabled=False,
    )
    v_on = cheap_marginal_value(
        src, tgt, ships, eta, world, model, me=0, wait_N=0,
        roi_enabled=True, roi_denom_floor=1.0,
    )
    assert v_off < 0, "test setup precondition: must hit bounce branch"
    assert v_off == v_on, f"bounce changed by ROI: off={v_off} on={v_on}"


def test_cheap_marginal_value_roi_composes_with_priors_associatively():
    """priority * (base / denom) == (priority * base) / denom for any priority.
    This is the math invariant that lets us multiply or divide in any order."""
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=3)
    tgt = _planet(1, -1, 14.0, 50.0, ships=5, production=2)
    world = _world_for(0, [src, tgt])
    model = WorldModel.from_world(world)

    ships, eta = 12, 5

    base_off = cheap_marginal_value(
        src, tgt, ships, eta, world, model, me=0, wait_N=0,
        roi_enabled=False,
    )
    base_on_with_prior = cheap_marginal_value(
        src, tgt, ships, eta, world, model, me=0, wait_N=0,
        roi_enabled=True, roi_denom_floor=1.0,
        priority_by_planet={1: 1.35},
    )
    expected = (base_off * 1.35) / (ships + eta + 1.0)
    assert base_on_with_prior == pytest.approx(expected, rel=1e-9)


# ---------- propose-level wiring ----------

def test_propose_threads_roi_kwargs_to_call_sites():
    """propose(roi_enabled=True) divides each cheap_delta from the
    positive-base branches; sort order can change but no entry exceeds
    the corresponding ROI-off entry on the same (src, tgt, wait_N)."""
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=3)
    tgt1 = _planet(1, -1, 14.0, 50.0, ships=5, production=2)
    tgt2 = _planet(2, -1, 50.0, 20.0, ships=8, production=3)
    world = _world_for(0, [src, tgt1, tgt2])
    model = WorldModel.from_world(world)

    out_off = propose(
        my_planets=[src], target_pool=[tgt1, tgt2],
        world=world, model=model, me=0, omega=0.0, baseline_len=50,
        roi_enabled=False,
    )
    out_on = propose(
        my_planets=[src], target_pool=[tgt1, tgt2],
        world=world, model=model, me=0, omega=0.0, baseline_len=50,
        roi_enabled=True, roi_denom_floor=1.0,
    )

    # Pair every (src_id, tgt_id, ships, eta, wait_N) entry across the two
    # outputs; the on-version must be <= the off-version (positive base
    # branch only -- bounces are unchanged).
    by_key_off = {
        (int(e[1].id), int(e[2].id), int(e[3]), int(e[5]), int(e[7])): e[0]
        for e in out_off
    }
    for e in out_on:
        key = (int(e[1].id), int(e[2].id), int(e[3]), int(e[5]), int(e[7]))
        off_delta = by_key_off.get(key)
        if off_delta is None:
            continue  # dedup may have picked different bucket winner
        # Positive entries should shrink under ROI; negative bounces are
        # untouched (off == on); priors are off so no other multiplier.
        if off_delta > 0:
            assert e[0] < off_delta, f"key={key} off={off_delta} on={e[0]}"
        else:
            assert e[0] == off_delta


# ---------- integration smoke + env-knob plumbing ----------

def test_agent_at_full_active_defaults_runs_clean():
    """Default env (ROI on, lambda_alpha=3, lambda_gap=2) runs end-to-end."""
    for key in ("BASELINE_ROI_ENABLED", "BASELINE_ROI_DENOM_FLOOR",
                "BASELINE_LAMBDA_ALPHA", "BASELINE_LAMBDA_GAP"):
        os.environ.pop(key, None)

    from agents.baseline.main import agent

    obs = {
        "player": 0,
        "planets": [
            (0, 0, 30.0, 50.0, 1.5, 20, 3),
            (1, 1, 70.0, 50.0, 1.5, 15, 2),
            (2, -1, 50.0, 30.0, 1.5, 8, 1),
            (3, -1, 50.0, 70.0, 1.5, 6, 4),
        ],
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
        "angular_velocity": 0.0,
        "step": 0,
    }
    actions = agent(obs, configuration={"actTimeout": 1.0})
    assert isinstance(actions, list)


def test_agent_roi_off_env_var_matches_monkey_patched_bypass():
    """BASELINE_ROI_ENABLED=0 plus monkey-patched proposer (roi_enabled
    forced off) must produce identical actions on a fixed obs.

    This is the integration ablation: setting the env knob to 0 is
    equivalent to the proposer never receiving the ROI kwarg at all."""
    from agents.baseline import main as baseline_main
    from agents.baseline import proposer as baseline_proposer

    obs = {
        "player": 0,
        "planets": [
            (0, 0, 30.0, 50.0, 1.5, 20, 3),
            (1, 1, 70.0, 50.0, 1.5, 15, 2),
            (2, -1, 50.0, 30.0, 1.5, 8, 1),
            (3, -1, 50.0, 70.0, 1.5, 6, 4),
        ],
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
        "angular_velocity": 0.0,
        "step": 0,
    }
    cfg = {"actTimeout": 1.0}

    # Path A: env var disables ROI (also clear priors so we're isolating ROI).
    os.environ["BASELINE_ROI_ENABLED"] = "0"
    os.environ["BASELINE_LAMBDA_ALPHA"] = "0"
    os.environ["BASELINE_LAMBDA_GAP"] = "0"
    try:
        actions_env_off = baseline_main.agent(obs, cfg)
    finally:
        os.environ.pop("BASELINE_ROI_ENABLED", None)
        os.environ.pop("BASELINE_LAMBDA_ALPHA", None)
        os.environ.pop("BASELINE_LAMBDA_GAP", None)

    # Path B: monkey-patch propose to force roi_enabled=False and drop priors.
    real_propose = baseline_proposer.propose

    def propose_no_features(*args, **kwargs):
        kwargs.pop("priority_by_planet", None)
        kwargs["roi_enabled"] = False
        kwargs.pop("roi_denom_floor", None)
        return real_propose(*args, **kwargs)

    baseline_main.proposer.propose = propose_no_features
    try:
        actions_bypass = baseline_main.agent(obs, cfg)
    finally:
        baseline_main.proposer.propose = real_propose

    assert actions_env_off == actions_bypass, (
        f"ROI env-off diverged from bypass:\n"
        f"  env_off: {actions_env_off}\n"
        f"  bypass:  {actions_bypass}"
    )


def test_agent_roi_env_parses_truthy_variants():
    """The _roi_enabled helper accepts 1/true/yes (case-insensitive) as on,
    and 0/false/no/off (case-insensitive) as off. Empty string is off."""
    from agents.baseline.main import _roi_enabled

    for v in ("1", "true", "True", "TRUE", "yes", "on"):
        os.environ["BASELINE_ROI_ENABLED"] = v
        assert _roi_enabled() is True, f"{v!r} should be truthy"

    for v in ("0", "false", "False", "FALSE", "no", "off", ""):
        os.environ["BASELINE_ROI_ENABLED"] = v
        assert _roi_enabled() is False, f"{v!r} should be falsy"

    os.environ.pop("BASELINE_ROI_ENABLED", None)
    assert _roi_enabled() is True, "default (unset) must be True"
