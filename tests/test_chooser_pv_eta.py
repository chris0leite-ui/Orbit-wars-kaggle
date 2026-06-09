"""Present-value time-discount on candidate Δ in `score_candidate_v4` /
`score_candidate_v4_joint` (2026-05-28, BASELINE_PV_ETA).

Pre-fix: `favor` calls `pv_horizon(step, 0)` — eta hardcoded to zero — so
a candidate that arrives in 40 turns is valued ~99% of one that arrives
in 10 turns. The actual EV gap is γ^(40−10) ≈ 0.74 at γ=0.99.

Post-fix: when `BASELINE_PV_ETA=1`, the chooser multiplies each
candidate's final Δ by γ^(wait_N + eta). No new tuning knob — γ is the
already-active chooser discount. Default OFF preserves byte-for-byte
legacy.

Fixture pattern mirrors `tests/test_chooser_ship_turn_penalty.py`.
"""

from __future__ import annotations

import importlib
import math
import os
from types import SimpleNamespace

import pytest

from lib.fast_sim import from_obs as fs_from_obs
from lib.intent import World
from lib.world_model import WorldModel


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


def _score_solo(ct, *, ships, wait_N, eta_hint=0,
                src_xy=(15.0, 15.0), tgt_xy=(40.0, 15.0),
                skip_admissibility=False, horizon=8):
    """Score one solo candidate. Geometry stays in the y<35 strip so the
    sun at (50,50) r=10 doesn't block straight-line paths."""
    favor_fn = ct.select_favor_fn()
    src = _planet(0, 0, src_xy[0], src_xy[1], ships=80, production=2)
    tgt = _planet(1, -1, tgt_xy[0], tgt_xy[1], ships=5, production=2)
    opp = _planet(2, 1, 85.0, 15.0, ships=40, production=2)
    world, snap, model = _build_world_and_snap([src, tgt, opp])

    baseline = ct.build_trajectory_baseline(
        snap, me=0, num_seats=2, horizon=horizon,
        favor_fn=favor_fn, gamma=0.99,
    )
    delta, status, eta = ct.score_candidate_v4(
        snap, src, tgt, ships=int(ships), angle=0.0,
        me=0, num_seats=2, world=world,
        baseline_favors=baseline, favor_fn=favor_fn,
        gamma=0.99, horizon=horizon,
        skip_admissibility=skip_admissibility, wait_N=int(wait_N),
        eta_hint=int(eta_hint),
        model=model,
    )
    assert status == "scored", f"expected scored, got {status}"
    return float(delta), int(eta)


def _score_joint(ct, *, legs):
    """Score one joint candidate. `legs` is a list of
    `(ships, tgt_xy, wait_N)` tuples. Returns (delta, leg_etas)."""
    favor_fn = ct.select_favor_fn()
    src = _planet(0, 0, 15.0, 15.0, ships=200, production=2)
    targets = []
    for i, (_ships, tgt_xy, _wait_N) in enumerate(legs):
        targets.append(_planet(10 + i, -1, tgt_xy[0], tgt_xy[1],
                               ships=5, production=2))
    opp = _planet(2, 1, 85.0, 85.0, ships=40, production=2)
    world, snap, model = _build_world_and_snap([src] + targets + [opp])

    horizon = 8
    baseline = ct.build_trajectory_baseline(
        snap, me=0, num_seats=2, horizon=horizon,
        favor_fn=favor_fn, gamma=0.99,
    )
    launches = []
    for i, (_ships, _tgt_xy, _wait_N) in enumerate(legs):
        ang = math.atan2(_tgt_xy[1] - src.y, _tgt_xy[0] - src.x)
        launches.append(
            (src, targets[i], int(_ships), float(ang), int(_wait_N))
        )
    delta, status = ct.score_candidate_v4_joint(
        snap, launches, me=0, num_seats=2, world=world,
        baseline_favors=baseline, favor_fn=favor_fn,
        gamma=0.99, horizon=horizon, skip_admissibility=False,
    )
    assert status == "scored", f"expected scored, got {status}"
    from lib.trajectory import predict_fleet_fate
    leg_etas = [
        int(predict_fleet_fate(L[0], L[1], L[3], L[2], world).step)
        for L in launches
    ]
    return float(delta), leg_etas


def test_pv_eta_default_off_preserves_legacy():
    """Env unset or =0 → no discount applied; scoring is byte-for-byte
    identical between unset and "0". Rule 38: reproduces the failure
    state (current peak behavior) before the new path is exercised."""
    os.environ.pop("BASELINE_PV_ETA", None)
    ct = _reload_chooser()
    assert ct.PV_ETA_ENABLED is False, "default must be OFF"
    delta_unset, eta_unset = _score_solo(ct, ships=20, wait_N=0)

    os.environ["BASELINE_PV_ETA"] = "0"
    ct = _reload_chooser()
    assert ct.PV_ETA_ENABLED is False
    delta_zero, eta_zero = _score_solo(ct, ships=20, wait_N=0)

    assert delta_unset == delta_zero, (
        f"BASELINE_PV_ETA unset vs '0' must produce identical Δ; "
        f"unset={delta_unset!r} zero={delta_zero!r}"
    )
    assert eta_unset == eta_zero


def test_pv_eta_on_discounts_solo_by_eta():
    """With PV_ETA=1, two candidates differing only in eta have Δ in the
    ratio γ^Δeta × (raw_Δ_b / raw_Δ_a). Holds for wait_N==0 path where
    eta is computed from predict_fleet_fate."""
    gamma = 0.99
    os.environ.pop("BASELINE_PV_ETA", None)
    ct = _reload_chooser()
    delta_near_off, eta_near = _score_solo(
        ct, ships=20, wait_N=0, tgt_xy=(40.0, 15.0),
    )
    delta_far_off, eta_far = _score_solo(
        ct, ships=20, wait_N=0, tgt_xy=(80.0, 15.0),
    )
    assert eta_far > eta_near, (
        f"fixture: far target must have larger eta; "
        f"near={eta_near} far={eta_far}"
    )

    os.environ["BASELINE_PV_ETA"] = "1"
    ct = _reload_chooser()
    assert ct.PV_ETA_ENABLED is True
    delta_near_on, eta_near2 = _score_solo(
        ct, ships=20, wait_N=0, tgt_xy=(40.0, 15.0),
    )
    delta_far_on, eta_far2 = _score_solo(
        ct, ships=20, wait_N=0, tgt_xy=(80.0, 15.0),
    )
    assert eta_near2 == eta_near and eta_far2 == eta_far

    expected_near = delta_near_off * (gamma ** eta_near)
    expected_far = delta_far_off * (gamma ** eta_far)
    assert delta_near_on == pytest.approx(expected_near, abs=1e-9), (
        f"near: Δ_on must equal Δ_off × γ^eta; "
        f"on={delta_near_on:.6f} expected={expected_near:.6f}"
    )
    assert delta_far_on == pytest.approx(expected_far, abs=1e-9), (
        f"far: Δ_on must equal Δ_off × γ^eta; "
        f"on={delta_far_on:.6f} expected={expected_far:.6f}"
    )


def test_pv_eta_on_discounts_solo_by_wait_plus_eta():
    """For wait_N>0 candidates, admissibility is skipped and the scorer
    uses eta_hint. With PV_ETA=1, Δ is discounted by γ^(wait_N + eta_hint)."""
    gamma = 0.99
    os.environ.pop("BASELINE_PV_ETA", None)
    ct = _reload_chooser()
    delta_off, eta_off = _score_solo(
        ct, ships=20, wait_N=5, eta_hint=10,
    )
    # wait_N>0 + skip_admissibility=False (default) → eta picks up eta_hint.
    assert eta_off == 10, (
        f"wait_N>0 with skip_admissibility=False should use eta_hint=10; "
        f"got eta={eta_off}"
    )

    os.environ["BASELINE_PV_ETA"] = "1"
    ct = _reload_chooser()
    delta_on, eta_on = _score_solo(
        ct, ships=20, wait_N=5, eta_hint=10,
    )
    assert eta_on == 10

    expected = delta_off * (gamma ** (5 + 10))
    assert delta_on == pytest.approx(expected, abs=1e-9), (
        f"Δ_on must equal Δ_off × γ^(wait_N + eta_hint) = Δ_off × γ^15; "
        f"on={delta_on:.6f} expected={expected:.6f}"
    )


def test_pv_eta_on_discounts_joint_by_max_leg_arrival():
    """Joint discount uses max(wait_N + leg_eta) across legs — the
    coalition's payoff is gated by the slowest arrival."""
    gamma = 0.99
    os.environ.pop("BASELINE_PV_ETA", None)
    ct = _reload_chooser()
    legs = [
        (15, (40.0, 15.0), 0),  # near target along +x
        (25, (15.0, 80.0), 0),  # far target along +y
    ]
    delta_off, leg_etas = _score_joint(ct, legs=legs)

    os.environ["BASELINE_PV_ETA"] = "1"
    ct = _reload_chooser()
    delta_on, leg_etas2 = _score_joint(ct, legs=legs)
    assert leg_etas == leg_etas2

    max_arrival = max(
        int(_wait_N) + int(eta_leg)
        for (_ships, _xy, _wait_N), eta_leg in zip(legs, leg_etas)
    )
    expected = delta_off * (gamma ** max_arrival)
    assert delta_on == pytest.approx(expected, abs=1e-9), (
        f"joint Δ_on must equal Δ_off × γ^max(wait_N+leg_eta) "
        f"= Δ_off × γ^{max_arrival}; "
        f"on={delta_on:.6f} expected={expected:.6f} leg_etas={leg_etas}"
    )


def _score_solo_isolated(ct, *, ships, tgt_xy, horizon):
    """Variant of _score_solo with the opp planet moved out of reach so
    it can't pre-capture nearby targets — needed when running both a
    near (40,15) and a far (80,15) target through the same rollout.
    The default fixture's opp at (85,15) sits right beside (80,15) and
    grabs it before our fleet arrives."""
    favor_fn = ct.select_favor_fn()
    src = _planet(0, 0, 15.0, 15.0, ships=80, production=2)
    tgt = _planet(1, -1, tgt_xy[0], tgt_xy[1], ships=5, production=2)
    # Opp placed in the lower-right corner where it can't reach either
    # (40,15) or (80,15) before our 20-ship fleet does.
    opp = _planet(2, 1, 15.0, 95.0, ships=40, production=2)
    world, snap, model = _build_world_and_snap([src, tgt, opp])
    baseline = ct.build_trajectory_baseline(
        snap, me=0, num_seats=2, horizon=horizon,
        favor_fn=favor_fn, gamma=0.99,
    )
    delta, status, eta = ct.score_candidate_v4(
        snap, src, tgt, ships=int(ships), angle=0.0,
        me=0, num_seats=2, world=world,
        baseline_favors=baseline, favor_fn=favor_fn,
        gamma=0.99, horizon=horizon,
        skip_admissibility=False, wait_N=0, eta_hint=0,
        model=model,
    )
    assert status == "scored", f"expected scored, got {status}"
    return float(delta), int(eta)


def test_pv_eta_ranking_inversion():
    """Strategic intent: PV_ETA shrinks the far/near Δ ratio so the
    chooser prefers fast captures over slow ones for the same ship
    investment. Uses a horizon long enough that both fleets land, and
    a fixture where the opp can't pre-capture either target."""
    os.environ.pop("BASELINE_PV_ETA", None)
    ct = _reload_chooser()

    h = 35
    delta_near_off, eta_near = _score_solo_isolated(
        ct, ships=20, tgt_xy=(40.0, 15.0), horizon=h,
    )
    delta_far_off, eta_far = _score_solo_isolated(
        ct, ships=20, tgt_xy=(80.0, 15.0), horizon=h,
    )
    assert eta_far > eta_near
    assert delta_near_off > 0 and delta_far_off > 0, (
        f"both captures must land within horizon={h}; "
        f"near={delta_near_off:.4f} far={delta_far_off:.4f}"
    )

    os.environ["BASELINE_PV_ETA"] = "1"
    ct = _reload_chooser()
    delta_near_on, _ = _score_solo_isolated(
        ct, ships=20, tgt_xy=(40.0, 15.0), horizon=h,
    )
    delta_far_on, _ = _score_solo_isolated(
        ct, ships=20, tgt_xy=(80.0, 15.0), horizon=h,
    )

    ratio_off = delta_far_off / delta_near_off
    ratio_on = delta_far_on / delta_near_on
    assert ratio_on < ratio_off, (
        f"PV_ETA must shrink the far/near Δ ratio; "
        f"off={ratio_off:.4f} on={ratio_on:.4f} "
        f"(eta_near={eta_near}, eta_far={eta_far})"
    )
    expected_shrink = 0.99 ** (eta_far - eta_near)
    assert ratio_on / ratio_off == pytest.approx(expected_shrink, abs=1e-9)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("BASELINE_PV_ETA", "BASELINE_SHIP_TURN_KAPPA",
              "BASELINE_MIN_DELTA", "BASELINE_NEUTRAL_BONUS",
              "BASELINE_FOLLOWON_BONUS"):
        monkeypatch.delenv(k, raising=False)
    yield
    _reload_chooser()
