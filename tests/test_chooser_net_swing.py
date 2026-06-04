"""Net-ship-swing value lens (`BASELINE_VALUE_HEAD=net_swing`) — Producer's
exact `competitive_score` brought into our own chooser.

The lens = `my_total_ships − Σ_opp(opp_total_ships)` (on-planet + in-flight)
read at the leaf of the forward-sim rollout. Paired with
`BASELINE_OPP_PASSIVE=1` (frozen opponent), `Δ = leaf(after move) − baseline`
equals Producer's net-ship swing over the horizon, by ship conservation:
`terminal_ships = initial_ships + produced − combat_lost`, and the identical
`initial` term cancels in the subtraction.

These tests pin: (1) the head math, (2) registration + champion byte-parity
when unset, (3) the favor-tuned post-leaf stack is bypassed under net_swing
(so the lens is unconfounded), (4) the rollout horizon is pinned to Producer's
window (18 in 2P / 13 in 4P), and (5) the conservation identity — the load-
bearing one — via the production integral the real engine applies
(`interpreter.py:667-669`: every non-neutral planet gains `production`/tick).

Fixture pattern mirrors `tests/test_chooser_pv_eta.py`.
"""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace

import pytest

from lib.fast_sim import from_obs as fs_from_obs
from lib.intent import World
from lib.world_model import WorldModel


# --------------------------------------------------------------------------
# Head math (no chooser needed)
# --------------------------------------------------------------------------

def test_net_swing_head_value_2p_counts_fleets_excludes_neutral():
    from agents.baseline.value import favor_net_swing
    obs = {
        "planets": [
            [0, 0, 0, 0, 5, 10, 2],    # mine: 10 ships
            [1, 1, 50, 0, 5, 4, 1],    # opp:  4 ships
            [2, -1, 25, 25, 5, 7, 1],  # neutral: excluded
        ],
        "fleets": [[9, 0, 10, 0, 0, 0, 3]],  # my in-flight: +3
        "step": 5,
    }
    # my = 10 + 3 = 13 ; opp = 4 ; neutral ignored → swing = 9
    assert favor_net_swing(obs, 0, 2) == pytest.approx(9.0)


def test_net_swing_head_plain_sum_4p_not_weighted():
    """4P must be a PLAIN sum over opponents (Producer's me − Σ_opp), NOT
    favor's 1.5×-weakest weighting — guards the copy-mistake."""
    from agents.baseline.value import favor_net_swing
    obs = {
        "planets": [
            [0, 0, 0, 0, 5, 20, 2],
            [1, 1, 50, 0, 5, 5, 1],
            [2, 2, 25, 25, 5, 3, 1],
            [3, 3, -25, 0, 5, 2, 1],
        ],
        "fleets": [],
        "step": 5,
    }
    # 20 − (5 + 3 + 2) = 10. A 1.5×-weakest scheme would inflate the weakest
    # (the 2-ship seat) and give a different number.
    assert favor_net_swing(obs, 0, 4) == pytest.approx(10.0)


# --------------------------------------------------------------------------
# Registration + champion byte-parity
# --------------------------------------------------------------------------

def test_net_swing_registration_and_default(monkeypatch):
    import agents.baseline.value as value
    monkeypatch.setenv("BASELINE_VALUE_HEAD", "net_swing")
    assert value.net_swing_active() is True
    assert value.select_favor_fn() is value.favor_net_swing

    monkeypatch.delenv("BASELINE_VALUE_HEAD", raising=False)
    assert value.net_swing_active() is False
    assert value.select_favor_fn() is value.favor  # champion path unchanged


# --------------------------------------------------------------------------
# Chooser-level fixtures (mirror test_chooser_pv_eta.py)
# --------------------------------------------------------------------------

def _planet(pid, owner, x, y, *, ships=20, production=2, radius=1.5):
    return SimpleNamespace(
        id=pid, owner=owner, x=x, y=y, radius=radius,
        ships=ships, production=production,
    )


def _build_world_and_snap(planets, my_id=0, num_seats=2, step=0):
    obs = {
        "player": my_id,
        "planets": [
            [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
            for p in planets
        ],
        "fleets": [],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
        "step": step,
    }
    world = World.from_obs(obs)
    snap = fs_from_obs(obs, num_seats=num_seats)
    model = WorldModel.from_world(world)
    return world, snap, model


def _reload_chooser():
    import agents.baseline.chooser_trajectory as ct
    importlib.reload(ct)
    return ct


def _score_solo(ct, *, ships=20, horizon=24, wait_N=0, eta_hint=0,
                tgt_xy=(40.0, 15.0), num_seats=2, baseline_h=None):
    """Score one solo capture candidate; baseline built deep enough that
    the horizon-pin can index it."""
    favor_fn = ct.select_favor_fn()
    src = _planet(0, 0, 15.0, 15.0, ships=80, production=2)
    tgt = _planet(1, -1, tgt_xy[0], tgt_xy[1], ships=5, production=2)
    opp = _planet(2, 1, 15.0, 95.0, ships=40, production=2)
    world, snap, model = _build_world_and_snap([src, tgt, opp],
                                               num_seats=num_seats)
    bh = baseline_h if baseline_h is not None else max(horizon, 24)
    baseline = ct.build_trajectory_baseline(
        snap, me=0, num_seats=num_seats, horizon=bh,
        favor_fn=favor_fn, gamma=0.99,
    )
    delta, status, eta = ct.score_candidate_v4(
        snap, src, tgt, ships=int(ships), angle=0.0,
        me=0, num_seats=num_seats, world=world,
        baseline_favors=baseline, favor_fn=favor_fn,
        gamma=0.99, horizon=horizon,
        skip_admissibility=False, wait_N=int(wait_N),
        eta_hint=int(eta_hint), model=model,
    )
    assert status == "scored", f"expected scored, got {status}"
    return float(delta), int(eta)


# --------------------------------------------------------------------------
# (3) The favor-tuned post-leaf stack is bypassed under net_swing
# --------------------------------------------------------------------------

def test_net_swing_bypasses_favor_tuned_terms(monkeypatch):
    """Under net_swing the raw `leaf − baseline` is returned, so the
    additive favor-tuned terms (SHIP_TURN_KAPPA / FOLLOWON / EXPAND_CREDIT)
    have NO effect on the score. Under the default favor head, SHIP_TURN
    DOES move the score — proving the bypass is real, not a no-op fixture."""
    # favor head: SHIP_TURN penalty changes the score.
    monkeypatch.delenv("BASELINE_VALUE_HEAD", raising=False)
    ct = _reload_chooser()
    favor_base, _ = _score_solo(ct, ships=30)
    ct.SHIP_TURN_KAPPA = 0.5
    favor_pen, _ = _score_solo(ct, ships=30)
    assert favor_pen != favor_base, (
        "control: under favor, SHIP_TURN_KAPPA must change the score"
    )

    # net_swing head: the same monkeypatched constants are bypassed.
    monkeypatch.setenv("BASELINE_VALUE_HEAD", "net_swing")
    monkeypatch.setenv("BASELINE_OPP_PASSIVE", "1")
    ct = _reload_chooser()
    ns_base, _ = _score_solo(ct, ships=30)
    ct.SHIP_TURN_KAPPA = 0.5
    ct.FOLLOWON_BONUS_WEIGHT = 5.0
    ct.EXPAND_CREDIT_WEIGHT = 5.0
    ct.NEUTRAL_BONUS_WEIGHT = 9.0
    ns_pen, _ = _score_solo(ct, ships=30)
    assert ns_pen == pytest.approx(ns_base), (
        "under net_swing the favor-tuned post-leaf terms must be bypassed; "
        f"base={ns_base!r} after-monkeypatch={ns_pen!r}"
    )


# --------------------------------------------------------------------------
# (4) Horizon pinned to Producer's window (18 in 2P)
# --------------------------------------------------------------------------

def test_net_swing_horizon_pinned_to_producer_window(monkeypatch):
    """A requested horizon ABOVE Producer's 2P window (18) is clamped to 18;
    a request AT 18 gives the same score; a request BELOW (10) differs."""
    monkeypatch.setenv("BASELINE_VALUE_HEAD", "net_swing")
    monkeypatch.setenv("BASELINE_OPP_PASSIVE", "1")
    ct = _reload_chooser()
    # baseline built to 30 in all three so indexing 18 is always valid.
    s_req30, _ = _score_solo(ct, ships=30, horizon=30, baseline_h=30)
    s_req18, _ = _score_solo(ct, ships=30, horizon=18, baseline_h=30)
    s_req10, _ = _score_solo(ct, ships=30, horizon=10, baseline_h=30)
    assert s_req30 == pytest.approx(s_req18), (
        f"horizon 30 must pin to 18 (same score); 30→{s_req30} 18→{s_req18}"
    )
    assert s_req10 != pytest.approx(s_req18), (
        f"horizon 10 (< window) must NOT pin; 10→{s_req10} 18→{s_req18}"
    )


# --------------------------------------------------------------------------
# (5) Conservation identity — the production integral (LOAD-BEARING)
# --------------------------------------------------------------------------

def test_net_swing_conservation_production_integral(monkeypatch):
    """With a frozen opponent (BASELINE_OPP_PASSIVE=1), the idle net-swing
    baseline over H ticks must grow by EXACTLY `(my_prod − opp_prod) * H`.

    Ground truth is hand-computed from the engine's production rule
    (`interpreter.py:667-669`: every non-neutral planet gains `production`
    ships/tick, no cap; neutrals never grow). This validates that the
    passive rollout + the net_swing leaf integrate production on BOTH sides
    correctly — i.e. the `produced` half of `terminal = initial + produced
    − combat_lost`. (fast_sim == real env is proven separately in
    test_fast_sim_parity.) No combat occurs (opponent launches nothing,
    no fleets in flight), so combat_lost = 0 and the identity is exact."""
    monkeypatch.setenv("BASELINE_VALUE_HEAD", "net_swing")
    monkeypatch.setenv("BASELINE_OPP_PASSIVE", "1")
    ct = _reload_chooser()
    favor_fn = ct.select_favor_fn()

    my_prod, opp_prod = 2, 1
    mine = _planet(0, 0, 15.0, 15.0, ships=10, production=my_prod)
    opp = _planet(1, 1, 85.0, 85.0, ships=5, production=opp_prod)
    # neutral far away, untouched (verifies neutrals don't grow / aren't counted)
    neutral = _planet(2, -1, 15.0, 85.0, ships=7, production=3)
    _world, snap, _model = _build_world_and_snap([mine, opp, neutral])

    H = 18
    baseline = ct.build_trajectory_baseline(
        snap, me=0, num_seats=2, horizon=H, favor_fn=favor_fn, gamma=0.99,
    )
    # net_swing[0] = my_ships - opp_ships = 10 - 5 = 5 (neutral excluded).
    assert baseline[0] == pytest.approx(10.0 - 5.0)
    # Each tick: my += 2, opp += 1 → swing grows by (2-1)=1/tick.
    delta = baseline[H] - baseline[0]
    assert delta == pytest.approx((my_prod - opp_prod) * H), (
        f"frozen-opponent net-swing must grow by (my_prod-opp_prod)*H = "
        f"{(my_prod - opp_prod) * H}; got {delta}. baseline[0]={baseline[0]} "
        f"baseline[H]={baseline[H]}"
    )
    # Monotonic, strictly (production is positive every tick).
    assert all(baseline[t + 1] >= baseline[t] for t in range(H))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("BASELINE_VALUE_HEAD", "BASELINE_OPP_PASSIVE",
              "BASELINE_PV_ETA", "BASELINE_SHIP_TURN_KAPPA",
              "BASELINE_FOLLOWON_BONUS", "BASELINE_EXPAND_CREDIT",
              "BASELINE_NEUTRAL_BONUS"):
        monkeypatch.delenv(k, raising=False)
    yield
    _reload_chooser()
