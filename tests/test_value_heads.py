"""Sanity tests for the v9 value heads."""

from __future__ import annotations

import math
import random

import pytest

from kaggle_environments import make

from lib import fast_sim
from lib import value_heads as _value_heads
from lib.value_heads import (
    CAPTURE_REWARD_WEIGHT,
    INFLIGHT_EXTRA_HORIZON,
    INFLIGHT_WEIGHT,
    composite_capture_value,
    delta_us_minus_them_obs,
    inflight_value,
)


def _warmed_snap(seed: int = 42, warmup: int = 15):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    rng = random.Random(seed)
    for _ in range(warmup):
        obs = env.state[0].observation
        a = [[p[0], rng.uniform(0, 6.28), int(p[5] // 2)]
             for p in obs["planets"] if p[1] == 0 and p[5] > 5 and rng.random() < 0.3]
        b = [[p[0], rng.uniform(0, 6.28), int(p[5] // 2)]
             for p in obs["planets"] if p[1] == 1 and p[5] > 5 and rng.random() < 0.3]
        env.step([a, b])
    snap = fast_sim.from_obs(
        env.state[0].observation, env.configuration,
        episode_seed=env.info["seed"], num_seats=2,
    )
    return snap


# ---------------------------------------------------------------------------
# delta_us_minus_them_obs
# ---------------------------------------------------------------------------


def test_delta_us_minus_them_basic():
    """Total ships per side; us minus them."""
    snap = _warmed_snap()
    d0 = delta_us_minus_them_obs(snap.state[0].observation, my_id=0)
    d1 = delta_us_minus_them_obs(snap.state[1].observation, my_id=1)
    # Antisymmetric: from seat 0 it's us-them; from seat 1 it's them-us.
    # The MAGNITUDES match (one observation, two POVs).
    assert d0 == -d1


def test_delta_us_minus_them_empty_obs():
    """Empty world → 0.0."""
    obs = {"planets": [], "fleets": []}
    assert delta_us_minus_them_obs(obs, my_id=0) == 0.0


# ---------------------------------------------------------------------------
# inflight_value
# ---------------------------------------------------------------------------


def test_inflight_value_reduces_to_base_when_no_inflight_captures():
    """If no in-flight fleets will flip ownership, inflight credit = 0
    and the head returns the base ship-delta."""
    obs = {"planets": [], "fleets": [], "player": 0, "angular_velocity": 0.0,
           "initial_planets": [], "comet_planet_ids": [], "comets": [],
           "step": 0, "next_fleet_id": 0}
    # Empty world → base 0, bonus 0.
    assert inflight_value(obs, my_id=0) == 0.0


def test_inflight_value_credits_predicted_capture():
    """A fleet en route to an enemy planet that will flip to us
    within extra_horizon adds production × weight to the score."""
    snap = _warmed_snap(seed=7, warmup=20)
    obs = snap.state[0].observation

    base = delta_us_minus_them_obs(obs, my_id=0)
    composite = inflight_value(obs, my_id=0)

    # The composite is base + bonus ≥ 0 (bonus is non-negative).
    assert composite >= base


def test_inflight_value_weight_calibration():
    """Setting weight=0 reduces inflight_value to base ship-delta."""
    snap = _warmed_snap()
    obs = snap.state[0].observation
    base = delta_us_minus_them_obs(obs, my_id=0)
    zero_bonus = inflight_value(obs, my_id=0, weight=0.0)
    assert base == zero_bonus


def test_inflight_value_extra_horizon_affects_score():
    """Larger extra_horizon should see more (or equal) captures."""
    snap = _warmed_snap(seed=42, warmup=25)
    obs = snap.state[0].observation
    short = inflight_value(obs, my_id=0, extra_horizon=10)
    long_ = inflight_value(obs, my_id=0, extra_horizon=50)
    # More predicted captures with longer horizon → score >= short.
    # Equal allowed for boards where the arrival ledger is empty.
    assert long_ >= short - 1e-9


def test_inflight_value_works_as_value_fn_in_score_candidate():
    """The signature `value_fn(obs, my_id)` is what score_candidate
    expects. inflight_value must accept that."""
    from lib.v7_search import score_candidate
    snap = _warmed_snap()
    # Score the incumbent's empty action.
    score = score_candidate(snap, [], my_id=0, K=5, opp_tier=1,
                             value_fn=inflight_value)
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# composite_capture_value — comet lifetime cap (2026-05-17)
# ---------------------------------------------------------------------------


def _obs_with_comet_target(path_len: int, path_index: int = 0):
    """Synthetic obs where our fleet is one tick from hitting a comet
    of known remaining-lifetime = path_len - path_index.

    Geometry: sun is at (50, 50) radius 10. Place comet at (15, 50)
    (35 units from sun-center — well outside SUN_RADIUS, so the sun
    gate doesn't false-positive on this trajectory). Our fleet is at
    (13, 50) moving +x. Our home at (5, 50). Enemy at (90, 50).
    """
    comet_id = 99
    return {
        "player": 0,
        "step": 100,
        "angular_velocity": 0.0,
        "next_fleet_id": 1,
        "initial_planets": [],
        # Planet tuple: (id, owner, x, y, radius, ships, production)
        "planets": [
            (0, 0, 5.0, 50.0, 2.0, 50, 1),     # our home
            (comet_id, -1, 15.0, 50.0, 1.0, 0, 2),  # comet target
            (1, 1, 90.0, 50.0, 2.0, 50, 1),    # enemy (far away)
        ],
        # Fleet tuple: (id, owner, x, y, angle, from_planet_id, ships)
        "fleets": [(0, 0, 13.0, 50.0, 0.0, 0, 10)],
        "comet_planet_ids": [comet_id],
        "comets": [{
            "planet_ids": [comet_id],
            # Path stays put around (15, 50) — the test cares about
            # lifetime, not the comet's motion. Path coords irrelevant
            # for comet_remaining_lifetime which only reads len + idx.
            "paths": [[[15.0, 50.0]] * path_len],
            "path_index": path_index,
        }],
    }


def test_composite_penalises_doomed_comet_target():
    """A fleet aimed at a comet that will expire BEFORE arrival hits
    empty space — composite must waste-penalise the launch. The
    WorldModel's per-planet timeline doesn't model comet expiry, so
    without the comet-lifetime gate the launch would slip through.
    Direct test of the 2026-05-17 PI direction: 'use comets only if
    really worth the risk and short lifetime'."""
    # Fleet at (48, 50) heading toward comet at (50, 50). Distance ~2,
    # speed for 10 ships ~1.96 → eta ≈ 1 tick.
    # path_len=1, path_index=0 → remaining lifetime = 1 tick at start.
    # comet_remaining_lifetime returns 1, eta=1, gate fires
    # (lifetime <= eta).
    doomed = _obs_with_comet_target(path_len=1, path_index=0)
    v_doomed = composite_capture_value(doomed, my_id=0)
    # Base (ship-delta): us=50+10=60, them=50 → 10. Waste penalty:
    # waste_weight (=0.5) * 10 ships = -5. Net: 10 - 5 = 5.
    # Without the gate composite would return ≥10 (no penalty).
    assert v_doomed < 10.0, (
        f"comet-lifetime-gate did not fire: v_doomed={v_doomed}, "
        f"expected < base ship-delta (10) due to waste penalty"
    )


def test_composite_allows_long_lived_comet_target():
    """A comet with plenty of life left is a legitimate target — the
    comet-lifetime gate must NOT fire (no waste penalty).

    Bug #15 fix v2 (2026-05-18 PM): the per-fleet capture-credit was
    DROPPED after the ablation showed it double-credited captures
    alongside the production-PV term and the chooser over-emitted.
    The PV term in the base does the work: planet ownership at the
    LEAF state credits the capture. Here the test is run on the
    current obs (not a leaf), so the comet is still neutral and the
    PV term contributes 0 (my_prod = opp_prod = 1 from the
    non-neutral planets). The fleet hits a healthy comet so the
    comet gate doesn't fire, and pred_owner == my_id at eta so no
    waste penalty. Result: just the base ship-delta = 10.
    """
    alive = _obs_with_comet_target(path_len=200, path_index=0)
    v_alive = composite_capture_value(alive, my_id=0)
    assert math.isclose(v_alive, 10.0, abs_tol=1e-6), (
        f"long-lived comet target should leave composite at base "
        f"ship-delta (no waste penalty fires, no per-fleet capture "
        f"credit in v2); got {v_alive}, expected 10.0"
    )


def test_composite_gate_distinguishes_short_vs_long_comet():
    """Direct A/B: same fleet, same target, only the comet's remaining
    lifetime differs. Short-life is penalised; long-life is not."""
    short = composite_capture_value(_obs_with_comet_target(path_len=1, path_index=0), my_id=0)
    long_ = composite_capture_value(_obs_with_comet_target(path_len=200, path_index=0), my_id=0)
    assert short < long_, (
        f"short-life comet value ({short}) should be < long-life "
        f"({long_}) — gate must penalise the doomed launch"
    )


def test_composite_penalises_sun_crossing_trajectory():
    """A fleet aimed at a planet behind the sun (chord passes within
    SUN_RADIUS of (CENTER, CENTER)) is killed by the engine at the
    crossing tick. composite must waste-penalise the launch.

    Live observation 2026-05-17 PM (PI): 'we shot a large fleet into
    the sun'. Without the gate composite credits the capture (or
    skips via pred_owner==me) and never penalises the lost ships.
    """
    # Our home at (20, 50), target at (80, 50), straight chord
    # passes through (50, 50) — the sun center. Gate must fire.
    obs = {
        "player": 0,
        "step": 100,
        "angular_velocity": 0.0,
        "next_fleet_id": 1,
        "initial_planets": [],
        "planets": [
            (0, 0, 20.0, 50.0, 2.0, 50, 1),
            (1, 1, 80.0, 50.0, 2.0, 10, 1),   # target behind the sun
        ],
        "fleets": [(0, 0, 22.0, 50.0, 0.0, 0, 100)],  # 100 ships, heading +x
        "comet_planet_ids": [],
        "comets": [],
    }
    v = composite_capture_value(obs, my_id=0)
    # Base ship-delta: us=50+100=150, them=10 → 140. With sun gate:
    # 140 - 0.5 * 100 = 90. Without gate: 140 (capture credit skipped
    # by pred_owner==me, no penalty fires).
    assert v < 140.0, (
        f"sun-crossing gate did not fire: v={v}, expected < base "
        f"ship-delta (140) due to waste penalty"
    )


# ---------------------------------------------------------------------------
# composite_capture_value — counterfactual capture credit (bug #15 fix)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    not _value_heads._COMPOSITE_PV_ENABLED,
    reason="PV term disabled in production (2026-05-18 wrap)",
    strict=False,
)
def test_composite_credits_post_arrival_capture_via_pv():
    """Bug #15 fix v2 (2026-05-18 PM). The per-fleet capture-credit
    was dropped; PV-term-in-the-base is now the sole capture-crediting
    mechanism. Verify it fires for a POST-ARRIVAL leaf state where the
    captured planet now reads as ours.

    The chooser calls composite on the rollout's LEAF observation. For
    short-eta candidates the fleet has already arrived by the leaf
    horizon, so we see the post-capture world: target owned by us, no
    in-flight fleet to credit. The PV term picks up the ownership flip
    via `(my_prod − opp_prod) × pv` — without it the leaf base equals
    the idle baseline and Δ = 0 (this was the bug #15 sanity-oracle
    failure mode).
    """
    # Post-arrival leaf: we own both planets, no fleets.
    leaf = {
        "player": 0,
        "step": 11,
        "angular_velocity": 0.0,
        "next_fleet_id": 0,
        "initial_planets": [],
        "planets": [
            (0, 0, 30.0, 30.0, 2.0, 11, 1),   # our home (post-launch growth)
            (1, 0, 70.0, 30.0, 2.0, 84, 1),   # just-captured opp planet
        ],
        "fleets": [],
        "comet_planet_ids": [],
        "comets": [],
    }
    # Same step idle baseline: opp still owns P1.
    idle = dict(leaf)
    idle["planets"] = [
        (0, 0, 30.0, 30.0, 2.0, 111, 1),
        (1, 1, 70.0, 30.0, 2.0, 16, 1),
    ]
    v_leaf = composite_capture_value(leaf, my_id=0)
    v_idle = composite_capture_value(idle, my_id=0)
    # Base ship-delta is symmetric (both = 95 by construction). The Δ
    # must come from the PV term: leaf has my_prod=2, opp_prod=0; idle
    # has my_prod=opp_prod=1. PV at step 11 ≈ 99.
    assert v_leaf > v_idle + 50.0, (
        f"PV term must credit the ownership flip at the leaf; "
        f"got v_leaf={v_leaf:.1f} vs v_idle={v_idle:.1f} "
        f"(expected difference >~ 100 from production-PV)"
    )


@pytest.mark.xfail(
    not _value_heads._COMPOSITE_PV_ENABLED,
    reason="PV term disabled in production (2026-05-18 wrap)",
    strict=False,
)
def test_composite_no_credit_for_over_reinforcement():
    """Per-fleet counterfactual fix (2026-05-18, bug #15). TWO of our
    identical fleets arrive at the SAME tick on the same opp target,
    either alone captures. For each fleet, the counterfactual (remove
    this fleet) leaves the other fleet still capturing → cf_owner ==
    my_id → no credit. Total per-fleet credit must be 0 (the PV term
    is the only signal). Guards against the regression where the fix
    over-credits redundant fleets.
    """
    obs = {
        "player": 0,
        "step": 100,
        "angular_velocity": 0.0,
        "next_fleet_id": 2,
        "initial_planets": [],
        "planets": [
            (0, 0, 15.0, 20.0, 2.0, 0, 1),     # our home A
            (1, 0, 15.0, 25.0, 2.0, 0, 1),     # our home B (off-axis)
            (2, 1, 80.0, 20.0, 2.0, 5, 1),     # opp planet
        ],
        # Two fleets at IDENTICAL position so they ray-cast to the
        # same eta on the same target → same ledger entry → both
        # over-reinforce under the counterfactual remove-one check.
        "fleets": [
            (0, 0, 22.0, 20.0, 0.0, 0, 100),
            (1, 0, 22.0, 20.0, 0.0, 1, 100),
        ],
        "comet_planet_ids": [],
        "comets": [],
    }
    from lib.scoring import pv_horizon as _pv
    v = composite_capture_value(obs, my_id=0)
    base = delta_us_minus_them_obs(obs, my_id=0)
    # Base = (0 + 0 + 100 + 100) - 5 = 195. PV = (2 - 1) × pv@step100.
    # Both fleets: cf timeline (without this fleet) still shows the
    # OTHER fleet captures → cf_owner = me → no credit each.
    pv = _pv(100, 0, gamma=0.99, t_total=500)
    expected = base + pv
    assert abs(v - expected) < 0.5, (
        f"redundant fleets should NOT each get capture credit; "
        f"got v={v}, expected ~{expected} (no per-fleet credit, only PV term). "
        f"If much higher (≈19 more), counterfactual is mis-attributing credit."
    )


def test_composite_penalises_bouncing_fleet():
    """Per-fleet logic: if our fleet has too few ships to capture
    (pred_owner stays opp at eta), waste penalty fires."""
    obs = {
        "player": 0,
        "step": 100,
        "angular_velocity": 0.0,
        "next_fleet_id": 1,
        "initial_planets": [],
        "planets": [
            (0, 0, 20.0, 20.0, 2.0, 50, 1),    # our home
            (1, 1, 80.0, 20.0, 2.0, 100, 1),   # well-defended opp
        ],
        "fleets": [(0, 0, 22.0, 20.0, 0.0, 0, 20)],  # 20 ships vs 100+
        "comet_planet_ids": [],
        "comets": [],
    }
    v = composite_capture_value(obs, my_id=0)
    base = delta_us_minus_them_obs(obs, my_id=0)
    # Bouncing fleet: -0.5 × 20 = -10. PV = 0 (1 - 1 = 0).
    assert v < base, (
        f"bounce penalty must fire on insufficient-ships launch; "
        f"got v={v}, base={base}"
    )


def test_composite_does_not_penalise_non_sun_crossing():
    """Fleet chord runs along y=20 (well below the sun at y=50);
    perp distance to sun = 30 > SUN_RADIUS. Sun gate must NOT fire
    (no waste penalty).

    Bug #15 fix v2 (2026-05-18 PM): per-fleet capture credit was
    dropped. In this single-fleet IN-FLIGHT scenario the target is
    still opp at obs time so the PV term contributes 0 (my_prod ==
    opp_prod == 1); pred_owner at eta is us (we'd capture) so no
    waste penalty either. composite returns plain base ship-delta.
    """
    obs = {
        "player": 0,
        "step": 100,
        "angular_velocity": 0.0,
        "next_fleet_id": 1,
        "initial_planets": [],
        "planets": [
            (0, 0, 20.0, 20.0, 2.0, 50, 1),    # our home
            (1, 1, 80.0, 20.0, 2.0, 10, 1),    # target on the +x ray
        ],
        # Fleet at (22, 20) heading +x along y=20 (below the sun).
        "fleets": [(0, 0, 22.0, 20.0, 0.0, 0, 100)],
        "comet_planet_ids": [],
        "comets": [],
    }
    v = composite_capture_value(obs, my_id=0)
    # Base ship-delta: us=50+100=150, them=10 → 140. No sun gate, no
    # comet gate, no waste penalty (pred_owner == me at eta), no
    # per-fleet capture credit (removed in v2). composite returns base.
    assert math.isclose(v, 140.0, abs_tol=1e-6), (
        f"non-sun-crossing trajectory should return base ship-delta "
        f"(no waste penalty, no per-fleet capture credit in v2); "
        f"got {v}, expected 140.0"
    )
