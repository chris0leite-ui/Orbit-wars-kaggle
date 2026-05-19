"""Unit + integration tests for lib/priority_prior.

See knowledge-base/concepts/per-class-priority-prior.md for the design.
"""
from __future__ import annotations

import math
import os

import pytest

from lib.priority_prior import (
    ALPHA_BY_CLASS,
    TOP10_SHARE_BY_CLASS,
    compute_class_of,
    compute_opp_share_in_flight,
    priority_by_planet,
)


# ---------- Layer 1: pure-function unit tests ----------

def test_alpha_and_top10_tables_cover_all_eight_classes():
    expected = {
        "high_prod_rotating_inner", "high_prod_rotating_outer",
        "high_prod_static_inner",   "high_prod_static_outer",
        "low_prod_rotating_inner",  "low_prod_rotating_outer",
        "low_prod_static_inner",    "low_prod_static_outer",
    }
    assert set(ALPHA_BY_CLASS) == expected
    assert set(TOP10_SHARE_BY_CLASS) == expected


def test_priority_collapses_to_one_at_zero_lambdas():
    class_of = {7: "low_prod_rotating_inner", 11: "high_prod_static_outer"}
    opp_share = {c: 1.0 / 8 for c in ALPHA_BY_CLASS}
    pri = priority_by_planet(class_of, opp_share, 0.0, 0.0)
    assert pri == {7: 1.0, 11: 1.0}


def test_priority_matches_closed_form_at_design_defaults():
    """priority(c) = exp(lambda_alpha * alpha + lambda_gap * gap)."""
    cls = "low_prod_rotating_inner"
    opp_share = {c: 0.0 for c in ALPHA_BY_CLASS}
    pri = priority_by_planet({42: cls}, opp_share, 3.0, 2.0)
    expected = math.exp(3.0 * ALPHA_BY_CLASS[cls]
                        + 2.0 * (TOP10_SHARE_BY_CLASS[cls] - 0.0))
    assert pri[42] == pytest.approx(expected, rel=1e-9)


def test_priority_negative_alpha_suppresses():
    """high_prod_static_outer (alpha = -0.049) gets a sub-1.0 multiplier."""
    cls = "high_prod_static_outer"
    opp_share = {c: TOP10_SHARE_BY_CLASS[c] for c in ALPHA_BY_CLASS}  # gap = 0
    pri = priority_by_planet({1: cls}, opp_share, 3.0, 2.0)
    assert pri[1] < 1.0
    assert pri[1] == pytest.approx(math.exp(3.0 * ALPHA_BY_CLASS[cls]), rel=1e-9)


def test_priority_positive_alpha_boosts():
    cls = "low_prod_rotating_inner"
    opp_share = {c: TOP10_SHARE_BY_CLASS[c] for c in ALPHA_BY_CLASS}  # gap = 0
    pri = priority_by_planet({1: cls}, opp_share, 3.0, 2.0)
    assert pri[1] > 1.0
    assert pri[1] == pytest.approx(math.exp(3.0 * ALPHA_BY_CLASS[cls]), rel=1e-9)


def test_priority_unknown_class_defaults_to_one():
    class_of = {99: "this_class_does_not_exist"}
    opp_share = {c: 0.0 for c in ALPHA_BY_CLASS}
    pri = priority_by_planet(class_of, opp_share, 3.0, 2.0)
    assert pri[99] == 1.0


# ---------- compute_class_of: end-to-end on raw obs ----------

def _make_planet(pid, owner, x, y, *, radius=1.5, ships=10, production=2):
    """Raw planet tuple matching the obs schema (id, owner, x, y, radius, ships, prod)."""
    return (pid, owner, x, y, radius, ships, production)


def test_compute_class_of_returns_one_label_per_planet():
    planets = [
        _make_planet(0, 0, 30.0, 50.0, production=5),  # high-ish prod, inner
        _make_planet(1, 1, 70.0, 50.0, production=1),  # low prod, outer
        _make_planet(2, -1, 50.0, 80.0, production=3),
        _make_planet(3, -1, 50.0, 20.0, production=4),
    ]
    class_of = compute_class_of(planets)
    assert set(class_of.keys()) == {0, 1, 2, 3}
    for label in class_of.values():
        assert label in ALPHA_BY_CLASS


def test_compute_class_of_empty_returns_empty():
    assert compute_class_of([]) == {}


# ---------- compute_opp_share_in_flight: synthetic ledger ----------

class _FakeModel:
    def __init__(self, ledger):
        self.ledger = ledger


def test_opp_share_empty_ledger_returns_top10_fallback():
    """Turn-0 opening: no fleets in flight -> opp_share == TOP10_SHARE_BY_CLASS."""
    model = _FakeModel({})
    class_of = {1: "low_prod_rotating_inner"}
    share = compute_opp_share_in_flight(model, me=0, class_of=class_of)
    assert share == TOP10_SHARE_BY_CLASS


def test_opp_share_all_my_fleets_returns_top10_fallback():
    """If only MY fleets are in flight, opp signal is empty -> fallback."""
    model = _FakeModel({
        1: [(5, 0, 10), (8, 0, 20)],  # owner=0=me, both mine
    })
    class_of = {1: "low_prod_rotating_inner"}
    share = compute_opp_share_in_flight(model, me=0, class_of=class_of)
    assert share == TOP10_SHARE_BY_CLASS


def test_opp_share_counts_enemy_fleets_by_class():
    """Three enemy fleets: 2 -> class A, 1 -> class B. Share = 2/3, 1/3."""
    model = _FakeModel({
        1: [(5, 1, 10)],          # enemy -> class A
        2: [(7, 1, 5)],           # enemy -> class A
        3: [(3, 1, 8)],           # enemy -> class B
        4: [(4, 0, 99)],          # MINE, must be skipped
    })
    class_of = {
        1: "low_prod_rotating_inner",   # class A
        2: "low_prod_rotating_inner",   # class A
        3: "high_prod_static_outer",    # class B
        4: "high_prod_static_inner",
    }
    share = compute_opp_share_in_flight(model, me=0, class_of=class_of)
    assert share["low_prod_rotating_inner"] == pytest.approx(2 / 3)
    assert share["high_prod_static_outer"] == pytest.approx(1 / 3)
    assert share["high_prod_static_inner"] == 0.0
    assert sum(share.values()) == pytest.approx(1.0)


def test_opp_share_ignores_neutral_owner():
    """Owner = -1 (neutral fleet, shouldn't really exist) is skipped."""
    model = _FakeModel({
        1: [(5, -1, 10), (5, 1, 10)],   # neutral skipped, enemy counted
    })
    class_of = {1: "low_prod_rotating_inner"}
    share = compute_opp_share_in_flight(model, me=0, class_of=class_of)
    assert share["low_prod_rotating_inner"] == pytest.approx(1.0)


# ---------- Ablation byte-identity (Rule 38 safety invariant) ----------
#
# At lambda_alpha = lambda_gap = 0, priority_by_planet returns {pid: 1.0
# for pid in ...}. cheap_marginal_value's `base * 1.0` must yield exactly
# `base`. This proves that shipping the code with both knobs at 0 is
# functionally byte-identical to the pre-priority-prior baseline.

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


def test_cheap_marginal_value_none_vs_all_ones_is_byte_identical():
    """priority_by_planet=None vs priority_by_planet={pid: 1.0 ...} must
    return the exact same float for every (src, tgt, ships, eta) input."""
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=3)
    tgt = _planet(1, -1, 14.0, 50.0, ships=5, production=2)
    world = _world_for(0, [src, tgt])
    model = WorldModel.from_world(world)

    all_ones = {0: 1.0, 1: 1.0}
    for ships in (3, 7, 12, 20):
        for eta in (1, 5, 12):
            v_none = cheap_marginal_value(
                src, tgt, ships, eta, world, model, me=0, wait_N=0,
                priority_by_planet=None,
            )
            v_ones = cheap_marginal_value(
                src, tgt, ships, eta, world, model, me=0, wait_N=0,
                priority_by_planet=all_ones,
            )
            v_default = cheap_marginal_value(
                src, tgt, ships, eta, world, model, me=0, wait_N=0,
            )
            assert v_none == v_default == v_ones, (
                f"ships={ships} eta={eta}: none={v_none} default={v_default} ones={v_ones}"
            )


def test_cheap_marginal_value_priority_multiplier_flows_through():
    """A priority of 2.0 on target doubles the cheap delta (sign preserved)."""
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=3)
    tgt = _planet(1, -1, 14.0, 50.0, ships=5, production=2)
    world = _world_for(0, [src, tgt])
    model = WorldModel.from_world(world)

    base = cheap_marginal_value(
        src, tgt, 12, 5, world, model, me=0, wait_N=0,
        priority_by_planet=None,
    )
    doubled = cheap_marginal_value(
        src, tgt, 12, 5, world, model, me=0, wait_N=0,
        priority_by_planet={1: 2.0},
    )
    assert doubled == pytest.approx(2.0 * base)


def test_propose_ablation_byte_identity():
    """At priority={pid: 1.0 for pid}, propose() output is byte-identical
    to propose(priority_by_planet=None)."""
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=3)
    tgt1 = _planet(1, -1, 14.0, 50.0, ships=5, production=2)
    tgt2 = _planet(2, -1, 50.0, 20.0, ships=8, production=3)
    world = _world_for(0, [src, tgt1, tgt2])
    model = WorldModel.from_world(world)

    out_none = propose(
        my_planets=[src], target_pool=[tgt1, tgt2],
        world=world, model=model, me=0, omega=0.0, baseline_len=50,
        priority_by_planet=None,
    )
    out_ones = propose(
        my_planets=[src], target_pool=[tgt1, tgt2],
        world=world, model=model, me=0, omega=0.0, baseline_len=50,
        priority_by_planet={0: 1.0, 1: 1.0, 2: 1.0},
    )

    # Tuple shape: (cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N).
    # Sort key (cheap_delta) must match exactly; other fields too.
    assert len(out_none) == len(out_ones)
    for a, b in zip(out_none, out_ones):
        assert a[0] == b[0], f"cheap_delta diverged: {a[0]} vs {b[0]}"
        assert int(a[1].id) == int(b[1].id)
        assert int(a[2].id) == int(b[2].id)
        assert a[3:] == b[3:]


def test_agent_priority_active_smokes_clean():
    """Integration smoke: agent runs end-to-end at default lambdas (3, 2)
    and returns a list (possibly empty if no valid moves)."""
    # Force env defaults — explicit so the test doesn't depend on shell state.
    os.environ.pop("BASELINE_LAMBDA_ALPHA", None)
    os.environ.pop("BASELINE_LAMBDA_GAP", None)

    from agents.baseline.main import agent

    obs = {
        "player": 0,
        "planets": [
            (0, 0, 30.0, 50.0, 1.5, 20, 3),   # mine
            (1, 1, 70.0, 50.0, 1.5, 15, 2),   # enemy
            (2, -1, 50.0, 30.0, 1.5, 8, 1),   # neutral
            (3, -1, 50.0, 70.0, 1.5, 6, 4),   # neutral high-prod
        ],
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
        "angular_velocity": 0.0,
        "step": 0,
    }
    actions = agent(obs, configuration={"actTimeout": 1.0})
    assert isinstance(actions, list)


def test_agent_ablated_matches_pre_priority_path():
    """With env vars BASELINE_LAMBDA_ALPHA=0 and BASELINE_LAMBDA_GAP=0, the
    agent's action list on a fixed obs equals the same agent with the
    priority dict forcibly set to None (monkey-patched).

    This is the integration-level ablation invariant. Together with the
    cheap_marginal_value byte-identity test above, it proves shipping
    with both knobs at 0 is functionally indistinguishable from the
    pre-priority-prior baseline."""
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

    # Path A: env vars at 0 -> priority_dict is {pid: 1.0 for pid}
    os.environ["BASELINE_LAMBDA_ALPHA"] = "0"
    os.environ["BASELINE_LAMBDA_GAP"] = "0"
    try:
        actions_zero = baseline_main.agent(obs, cfg)
    finally:
        os.environ.pop("BASELINE_LAMBDA_ALPHA", None)
        os.environ.pop("BASELINE_LAMBDA_GAP", None)

    # Path B: monkey-patch propose to drop the priority kwarg entirely.
    real_propose = baseline_proposer.propose

    def propose_no_priority(*args, **kwargs):
        kwargs.pop("priority_by_planet", None)
        return real_propose(*args, **kwargs)

    baseline_main.proposer.propose = propose_no_priority
    try:
        actions_bypass = baseline_main.agent(obs, cfg)
    finally:
        baseline_main.proposer.propose = real_propose

    assert actions_zero == actions_bypass, (
        f"ablation diverged:\n  zero-lambda: {actions_zero}\n"
        f"  bypass:      {actions_bypass}"
    )
