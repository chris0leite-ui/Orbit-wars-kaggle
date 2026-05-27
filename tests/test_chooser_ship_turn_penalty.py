"""Ship-turn opportunity-cost penalty in `score_candidate_v4` /
`score_candidate_v4_joint` (2026-05-27 Step 2B).

Pre-fix: leaf `favor` returns ~297-340 for any positive-prod capture
regardless of eta — `pv_horizon(leaf_step, 0)` ≈ 99 for any reachable
leaf step in 25..50 with γ=0.99, t_total=500. Slow captures (eta=40)
score ~88% of fast captures (eta=10) when they tie up ships 4x longer.
Post-fix: `BASELINE_SHIP_TURN_KAPPA` env var subtracts
`κ × ships × (wait_N + eta)` from delta, pricing the opportunity cost.

Fixture pattern mirrors `tests/test_chooser_min_delta_floor.py` —
synthetic 2P World, direct `score_candidate_v4` call, env-var reload
via `importlib.reload`.
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


def _score_solo(ct, *, ships, wait_N, src_xy=(15.0, 15.0),
                tgt_xy=(40.0, 15.0), skip_admissibility=False):
    """Score one solo candidate with controllable ships / wait_N.

    Geometry stays in the y<35 strip so the sun at (50,50) r=10 doesn't
    block straight-line paths.

    With `skip_admissibility=True`, the function sets eta=0 (the line-459
    default) — useful for `test_default_off_preserves_legacy` and
    `test_penalty_scales_with_wait_n`. With False, `eta` is the actual
    fate.step from predict_fleet_fate.
    """
    favor_fn = ct.select_favor_fn()
    src = _planet(0, 0, src_xy[0], src_xy[1], ships=80, production=2)
    tgt = _planet(1, -1, tgt_xy[0], tgt_xy[1], ships=5, production=2)
    opp = _planet(2, 1, 85.0, 15.0, ships=40, production=2)
    world, snap, model = _build_world_and_snap([src, tgt, opp])

    horizon = 8
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
        # Distinct neutral targets so admissibility per-leg succeeds.
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
    # Independently compute the leg etas via predict_fleet_fate for
    # cross-check (the penalty math uses these).
    from lib.trajectory import predict_fleet_fate
    leg_etas = [
        int(predict_fleet_fate(L[0], L[1], L[3], L[2], world).step)
        for L in launches
    ]
    return float(delta), leg_etas


def test_default_off_preserves_legacy():
    """κ=0.0 → no penalty applied; scoring is byte-for-byte identical
    whether the env var is unset or set to 0.0."""
    os.environ.pop("BASELINE_SHIP_TURN_KAPPA", None)
    ct = _reload_chooser()
    delta_unset, eta_unset = _score_solo(
        ct, ships=20, wait_N=0, skip_admissibility=True,
    )

    os.environ["BASELINE_SHIP_TURN_KAPPA"] = "0.0"
    ct = _reload_chooser()
    delta_zero, eta_zero = _score_solo(
        ct, ships=20, wait_N=0, skip_admissibility=True,
    )

    assert delta_unset == delta_zero, (
        f"default κ=0.0 must preserve legacy bit-for-bit; "
        f"unset={delta_unset!r} vs zero={delta_zero!r}"
    )
    assert eta_unset == eta_zero


def test_positive_kappa_penalizes_slow_more_than_fast():
    """Two candidates from the same source — near (fast) and far (slow)
    targets. With κ>0, the far target's delta drops by MORE absolute
    points than the near target's. Reproduces the symptom the penalty
    targets: slow-launch overscoring relative to fast-launch."""
    os.environ.pop("BASELINE_SHIP_TURN_KAPPA", None)
    ct = _reload_chooser()
    # Baseline (κ=0): score the two candidates.
    delta_fast_off, eta_fast = _score_solo(
        ct, ships=20, wait_N=0, tgt_xy=(40.0, 15.0),
    )
    delta_slow_off, eta_slow = _score_solo(
        ct, ships=20, wait_N=0, tgt_xy=(80.0, 15.0),
    )
    assert eta_slow > eta_fast, (
        f"fixture: far target must have larger eta; "
        f"fast={eta_fast} slow={eta_slow}"
    )

    # With κ=1.0, both should drop, but slow should drop MORE.
    os.environ["BASELINE_SHIP_TURN_KAPPA"] = "1.0"
    ct = _reload_chooser()
    delta_fast_on, _ = _score_solo(
        ct, ships=20, wait_N=0, tgt_xy=(40.0, 15.0),
    )
    delta_slow_on, _ = _score_solo(
        ct, ships=20, wait_N=0, tgt_xy=(80.0, 15.0),
    )

    drop_fast = delta_fast_off - delta_fast_on
    drop_slow = delta_slow_off - delta_slow_on
    assert drop_slow > drop_fast, (
        f"slow capture must lose more delta than fast; "
        f"drop_fast={drop_fast:.4f} drop_slow={drop_slow:.4f}"
    )
    # Quantitative: drop = κ × ships × eta = 20 × eta for each.
    expected_drop_fast = 1.0 * 20.0 * float(eta_fast)
    expected_drop_slow = 1.0 * 20.0 * float(eta_slow)
    assert drop_fast == pytest.approx(expected_drop_fast, abs=1e-9)
    assert drop_slow == pytest.approx(expected_drop_slow, abs=1e-9)


def test_penalty_scales_with_ships():
    """Same (src, tgt), two ship sizes. Penalty difference equals
    κ × (ships_b − ships_a) × eta."""
    os.environ.pop("BASELINE_SHIP_TURN_KAPPA", None)
    ct = _reload_chooser()
    delta_a_off, eta_a = _score_solo(ct, ships=10, wait_N=0)
    delta_b_off, eta_b = _score_solo(ct, ships=20, wait_N=0)
    # Different ship counts → different fleet speeds → potentially
    # different etas. Penalty is κ × ships × eta per candidate; check
    # the absolute drop for each separately.

    kappa = 1.0
    os.environ["BASELINE_SHIP_TURN_KAPPA"] = repr(kappa)
    ct = _reload_chooser()
    delta_a_on, _ = _score_solo(ct, ships=10, wait_N=0)
    delta_b_on, _ = _score_solo(ct, ships=20, wait_N=0)

    drop_a = delta_a_off - delta_a_on
    drop_b = delta_b_off - delta_b_on
    assert drop_a == pytest.approx(kappa * 10.0 * float(eta_a), abs=1e-9)
    assert drop_b == pytest.approx(kappa * 20.0 * float(eta_b), abs=1e-9)


def test_penalty_scales_with_wait_n():
    """Same (src, tgt, ships), wait_N ∈ {0, 5}. Penalty for wait_N=5
    is `κ × ships × (5 + 0)` (the wait>0 path takes the
    skip_admissibility branch, so eta stays at 0)."""
    os.environ.pop("BASELINE_SHIP_TURN_KAPPA", None)
    ct = _reload_chooser()
    delta_w0_off, eta_w0 = _score_solo(
        ct, ships=20, wait_N=0, skip_admissibility=True,
    )
    delta_w5_off, eta_w5 = _score_solo(
        ct, ships=20, wait_N=5, skip_admissibility=True,
    )
    # With skip_admissibility=True, eta is the line-459 default of 0
    # for both calls.
    assert eta_w0 == 0
    assert eta_w5 == 0

    kappa = 1.0
    os.environ["BASELINE_SHIP_TURN_KAPPA"] = repr(kappa)
    ct = _reload_chooser()
    delta_w0_on, _ = _score_solo(
        ct, ships=20, wait_N=0, skip_admissibility=True,
    )
    delta_w5_on, _ = _score_solo(
        ct, ships=20, wait_N=5, skip_admissibility=True,
    )

    drop_w0 = delta_w0_off - delta_w0_on
    drop_w5 = delta_w5_off - delta_w5_on
    # wait_N=0, eta=0 → penalty = 0.
    assert drop_w0 == pytest.approx(0.0, abs=1e-9)
    # wait_N=5, eta=0 → penalty = κ × 20 × (5+0) = 100.
    assert drop_w5 == pytest.approx(kappa * 20.0 * 5.0, abs=1e-9)


def test_joint_penalty_sums_per_leg():
    """Joint candidate of two legs; total penalty equals
    `κ × Σ ships_i × eta_i`."""
    os.environ.pop("BASELINE_SHIP_TURN_KAPPA", None)
    ct = _reload_chooser()
    legs = [
        (15, (40.0, 15.0), 0),  # near target along +x, small fleet
        (25, (15.0, 80.0), 0),  # far target along +y, larger fleet
    ]
    delta_off, leg_etas = _score_joint(ct, legs=legs)

    kappa = 1.0
    os.environ["BASELINE_SHIP_TURN_KAPPA"] = repr(kappa)
    ct = _reload_chooser()
    delta_on, leg_etas2 = _score_joint(ct, legs=legs)
    assert leg_etas == leg_etas2, "etas should be deterministic across reloads"

    drop = delta_off - delta_on
    expected = kappa * sum(
        float(ships) * float(eta_leg)
        for (ships, _xy, _wait), eta_leg in zip(legs, leg_etas)
    )
    assert drop == pytest.approx(expected, abs=1e-9), (
        f"joint penalty must sum per-leg κ × ships × eta; "
        f"drop={drop:.4f} expected={expected:.4f} leg_etas={leg_etas}"
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("BASELINE_SHIP_TURN_KAPPA", "BASELINE_MIN_DELTA",
              "BASELINE_NEUTRAL_BONUS", "BASELINE_FOLLOWON_BONUS"):
        monkeypatch.delenv(k, raising=False)
    yield
    _reload_chooser()
