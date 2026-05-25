"""Pin all three terms of `favor_strategic` plus parity vs base `favor`.

Phase F refactor (2026-05-25):
- Use pytest's `monkeypatch` fixture to set env vars (auto-restores;
  fixes the test-order-dependence the previous `os.environ[k]=v`
  pattern introduced).
- Add 4P parity test (Phase D's parity test only covered 2P).
- Add 7 new tests covering Term C edge cases (zero/negative threshold,
  subsume of ELIMINATION_BONUS in 4P, exclude zero-prod opps), Term A
  narrowed-exception behavior, and the fallback counter.
"""
from __future__ import annotations

import importlib
import math

import pytest

# Knobs that govern favor_strategic; tests reset ALL of them in every
# test function to make tests order-independent.
_STRATEGIC_KNOBS = (
    "BASELINE_HOLD_HORIZON",
    "BASELINE_FORWARD_REACH_WEIGHT",
    "BASELINE_FORWARD_REACH_HORIZON",
    "BASELINE_FINISH_BONUS",
    "BASELINE_FINISH_THRESHOLD",
    "BASELINE_VALUE_HEAD",
    "BASELINE_STOCKPILE_PENALTY",
)


def _reload_value_with(monkeypatch, env_overrides):
    """Set env vars via monkeypatch (auto-restores) then reload value.py.

    Every knob is set EXPLICITLY (defaulting to "0" / "" when unspecified),
    so module-level constants are deterministic regardless of prior test
    state or shell environment.
    """
    defaults = {
        "BASELINE_HOLD_HORIZON": "0",
        "BASELINE_FORWARD_REACH_WEIGHT": "0",
        "BASELINE_FORWARD_REACH_HORIZON": "15",
        "BASELINE_FINISH_BONUS": "0",
        "BASELINE_FINISH_THRESHOLD": "200",
        "BASELINE_VALUE_HEAD": "",
        "BASELINE_STOCKPILE_PENALTY": "0",
    }
    defaults.update(env_overrides)
    for k, v in defaults.items():
        monkeypatch.setenv(k, v)
    import agents.baseline.value as vmod
    importlib.reload(vmod)
    return vmod


def _mk_obs(planets, step=0, player=0):
    return {
        "player": player,
        "planets": planets,
        "fleets": [],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": step,
    }


# -- Parity tests ----------------------------------------------------


def test_parity_with_knobs_off_2p(monkeypatch):
    """All three terms OFF → favor_strategic ≡ favor (2P)."""
    v = _reload_value_with(monkeypatch, {})
    obs = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0, 10, 3),
        (1, 1, 70.0, 50.0, 1.0,  8, 2),
    ])
    base = v.favor(obs, 0, num_seats=2, gamma=0.99)
    strat = v.favor_strategic(obs, 0, num_seats=2, gamma=0.99)
    assert strat == pytest.approx(base, abs=1e-9), (
        f"2P parity broken: base={base} strategic={strat}"
    )


def test_4p_max_of_opps_at_knobs_off(monkeypatch):
    """2026-05-26 Change 2: in 4P at knobs OFF, favor_strategic uses
    max-of-opps aggregation (NOT the original `favor`'s weighted-sum
    with 1.5x-weakest). They diverge intentionally. Verify the new
    invariant: F1 = my_ships - max(opp_ships), F2 = (my_prod - max(opp_prod))*pv,
    plus the discrete elim_bonus when FINISH_BONUS=0 and the elim gate
    fires."""
    v = _reload_value_with(monkeypatch, {})
    obs = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0,  10, 3),
        (1, 1, 70.0, 50.0, 1.0,   8, 2),
        (2, 2, 50.0, 30.0, 1.0,   4, 1),  # weakest opp (strength = 4 + 15 = 19)
        (3, 3, 50.0, 70.0, 1.0,  10, 2),
    ])
    # By hand: my_ships=10, opp_ships per opp = 8, 4, 10. max = 10.
    # my_prod=3, opp_prod per opp = 2, 1, 2. max = 2.
    # pv at step=0, gamma=0.99, t_total=500 ≈ 99.3.
    # F1 = 10 - 10 = 0. F2 = (3 - 2) * 99.3 ≈ 99.3.
    # weakest_str = 4 + 1*15 = 19 (≤ 110), my_strength = 10 + 3*15 = 55.
    # Elim gate: my_strength ≥ 0.9 * 19 = 17.1? 55 ≥ 17.1 → YES.
    # FINISH_BONUS=0 default in this test → discrete elim_bonus = +55 fires.
    # Expected strat = 0 + 99.3 + 55 ≈ 154.3.
    strat = v.favor_strategic(obs, 0, num_seats=4, gamma=0.99)
    from lib.scoring import pv_horizon
    pv = pv_horizon(0, 0, gamma=0.99, t_total=500)
    expected = (10.0 - 10.0) + (3.0 - 2.0) * pv + 55.0
    assert strat == pytest.approx(expected, abs=1e-6), (
        f"4P max-of-opps at knobs-off broke: expected={expected} got={strat}"
    )

    # Sanity: confirm `favor` (the base) still uses weighted-sum (i.e. they DO
    # differ — Change 2 is intentional, not accidental).
    base = v.favor(obs, 0, num_seats=4, gamma=0.99)
    assert base != pytest.approx(strat, abs=1e-3), (
        f"`favor` and `favor_strategic` should differ in 4P (Change 2 makes "
        f"strategic use max-of-opps, favor still weighted-sum): both={base}"
    )


# -- Term A ----------------------------------------------------------


def test_term_a_discounts_threatened_planet(monkeypatch):
    """A planet with imminent threat scores lower than the same planet
    safe — Term A discounts my prod when opp can credibly attack me.

    2026-05-26 rewrite: set up so opp CAN credibly capture (opp has
    enough ships for capture-size launch); the symmetric discount on
    OPP's prod doesn't fire because I lack ships for a capture-size
    counter-launch. This isolates the my-side discount.
    """
    v = _reload_value_with(monkeypatch, {"BASELINE_HOLD_HORIZON": "20"})
    # Asymmetric capability: me with 5 ships, opp with 100. Opp can
    # capture me (capture-size = 6, opp has 100); I can't capture opp
    # (capture-size = 101, I have 5). Only my side gets the Term A
    # discount.
    obs_safe = _mk_obs([
        (0, 0,  5.0,  5.0, 1.0,   5, 3),
        (1, 1, 95.0, 95.0, 1.0, 100, 2),
    ])
    obs_close = _mk_obs([
        (0, 0, 40.0, 50.0, 1.0,   5, 3),
        (1, 1, 60.0, 50.0, 1.0, 100, 2),
    ])
    s_safe = v.favor_strategic(obs_safe, 0, num_seats=2, gamma=0.99)
    s_close = v.favor_strategic(obs_close, 0, num_seats=2, gamma=0.99)
    assert s_safe > s_close, (
        f"Term A failed to penalise threatened planet: "
        f"safe={s_safe} close={s_close}"
    )


def test_term_a_fallback_counter_increments(monkeypatch):
    """When Term A's threat-eta computation raises a typed-bug error, the
    counter bumps and we silently fall back to undiscounted my_prod (with
    a one-time RuntimeWarning).
    """
    v = _reload_value_with(monkeypatch, {"BASELINE_HOLD_HORIZON": "20"})
    # Force fallback by giving `math.hypot` a non-numeric coordinate.
    obs_bad = _mk_obs([
        (0, 0, "X", 50.0, 1.0, 10, 3),   # malformed: str position triggers TypeError
        (1, 1, 60.0, 50.0, 1.0,  8, 2),
    ])
    before = v._TERM_A_FALLBACK_COUNT
    # Will emit RuntimeWarning ON FIRST fallback.
    with pytest.warns(RuntimeWarning):
        v.favor_strategic(obs_bad, 0, num_seats=2, gamma=0.99)
    after = v._TERM_A_FALLBACK_COUNT
    assert after == before + 1, (
        f"Term A fallback counter did not bump: before={before} after={after}"
    )


def test_term_a_propagates_systemexit(monkeypatch):
    """Bare except was narrowed to (KeyError, IndexError, AttributeError,
    ValueError, TypeError, ZeroDivisionError). SystemExit must propagate.
    """
    v = _reload_value_with(monkeypatch, {"BASELINE_HOLD_HORIZON": "20"})
    # Monkey-patch math.hypot inside the value module to raise SystemExit.
    def _raise_systemexit(*a, **kw):
        raise SystemExit("propagate me")

    monkeypatch.setattr(v.math, "hypot", _raise_systemexit)
    # Make opp strong enough to credibly threaten me (capture-size = my+1 = 11,
    # opp has 100 >= 11), so the threat-ETA helper reaches math.hypot.
    obs = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0,  10, 3),
        (1, 1, 70.0, 50.0, 1.0, 100, 2),
    ])
    with pytest.raises(SystemExit):
        v.favor_strategic(obs, 0, num_seats=2, gamma=0.99)


# -- Term B ----------------------------------------------------------


def test_term_b_rewards_reachable_enemy_supply(monkeypatch):
    """Identical my-position; vary number of enemies in reach. More reachable
    enemy production → higher favor (Term B additive)."""
    v = _reload_value_with(monkeypatch, {
        "BASELINE_FORWARD_REACH_WEIGHT": "1.0",
        "BASELINE_FORWARD_REACH_HORIZON": "30",
    })
    obs_one = _mk_obs([
        (0, 0, 50.0, 50.0, 1.0, 10, 3),
        (1, 1, 60.0, 50.0, 1.0,  5, 2),
    ])
    obs_three = _mk_obs([
        (0, 0, 50.0, 50.0, 1.0, 10, 3),
        (1, 1, 60.0, 50.0, 1.0,  5, 2),
        (2, 1, 50.0, 60.0, 1.0,  5, 2),
        (3, 1, 40.0, 50.0, 1.0,  5, 2),
    ])
    base_one = v.favor(obs_one, 0, num_seats=2, gamma=0.99)
    base_three = v.favor(obs_three, 0, num_seats=2, gamma=0.99)
    strat_one = v.favor_strategic(obs_one, 0, num_seats=2, gamma=0.99)
    strat_three = v.favor_strategic(obs_three, 0, num_seats=2, gamma=0.99)
    delta_one = strat_one - base_one
    delta_three = strat_three - base_three
    assert delta_three > delta_one, (
        f"Term B did not scale with enemy count: "
        f"delta_one={delta_one} delta_three={delta_three}"
    )


# -- Term C ----------------------------------------------------------


def test_term_c_rewards_weak_opp(monkeypatch):
    """Term C with FINISH_BONUS>0 prefers boards where opp is weak."""
    v = _reload_value_with(monkeypatch, {
        "BASELINE_FINISH_BONUS": "50",
        "BASELINE_FINISH_THRESHOLD": "200",
    })
    obs_strong = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0,  10, 3),
        (1, 1, 70.0, 50.0, 1.0, 200, 2),
    ])
    obs_weak = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0,  10, 3),
        (1, 1, 70.0, 50.0, 1.0,   5, 2),
    ])
    base_strong = v.favor(obs_strong, 0, num_seats=2, gamma=0.99)
    base_weak = v.favor(obs_weak, 0, num_seats=2, gamma=0.99)
    strat_strong = v.favor_strategic(obs_strong, 0, num_seats=2, gamma=0.99)
    strat_weak = v.favor_strategic(obs_weak, 0, num_seats=2, gamma=0.99)
    delta_strong = strat_strong - base_strong
    delta_weak = strat_weak - base_weak
    assert delta_weak > delta_strong
    # Magnitude check: at strength=35, pressure ≈ 0.825, bonus ≈ 41.25.
    assert delta_weak == pytest.approx(50.0 * (1.0 - 35.0/200.0), abs=1.0)


def test_term_c_disabled_when_zero(monkeypatch):
    """FINISH_BONUS=0 → no Term C contribution AND discrete 4P
    ELIMINATION_BONUS path is restored (F2 back-compat)."""
    v_off = _reload_value_with(monkeypatch, {
        "BASELINE_HOLD_HORIZON": "20",
        "BASELINE_FORWARD_REACH_WEIGHT": "0.5",
        "BASELINE_FORWARD_REACH_HORIZON": "15",
        "BASELINE_FINISH_BONUS": "0",
    })
    obs = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0,  10, 3),
        (1, 1, 70.0, 50.0, 1.0,   5, 2),
    ])
    s_off = v_off.favor_strategic(obs, 0, num_seats=2, gamma=0.99)
    v_on = _reload_value_with(monkeypatch, {
        "BASELINE_HOLD_HORIZON": "20",
        "BASELINE_FORWARD_REACH_WEIGHT": "0.5",
        "BASELINE_FORWARD_REACH_HORIZON": "15",
        "BASELINE_FINISH_BONUS": "50",
        "BASELINE_FINISH_THRESHOLD": "200",
    })
    s_on = v_on.favor_strategic(obs, 0, num_seats=2, gamma=0.99)
    assert s_on > s_off


def test_term_c_threshold_zero_does_not_crash(monkeypatch):
    """Phase F F1: BASELINE_FINISH_THRESHOLD=0 must NOT raise ZeroDivisionError."""
    v = _reload_value_with(monkeypatch, {
        "BASELINE_FINISH_BONUS": "50",
        "BASELINE_FINISH_THRESHOLD": "0",
    })
    obs = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0,  10, 3),
        (1, 1, 70.0, 50.0, 1.0,   5, 2),
    ])
    # Must not raise.
    result = v.favor_strategic(obs, 0, num_seats=2, gamma=0.99)
    assert math.isfinite(result), f"favor_strategic returned non-finite: {result}"


def test_term_c_threshold_negative_does_not_crash_or_explode(monkeypatch):
    """Phase F F1: BASELINE_FINISH_THRESHOLD<0 must NOT produce unbounded bonus."""
    v = _reload_value_with(monkeypatch, {
        "BASELINE_FINISH_BONUS": "50",
        "BASELINE_FINISH_THRESHOLD": "-50",
    })
    obs = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0,  10, 3),
        (1, 1, 70.0, 50.0, 1.0,   5, 2),
    ])
    base = v.favor(obs, 0, num_seats=2, gamma=0.99)
    strat = v.favor_strategic(obs, 0, num_seats=2, gamma=0.99)
    delta = strat - base
    # The Term C contribution must be bounded by FINISH_BONUS=50.
    assert -1e-6 <= delta <= 50.0 + 1e-6, (
        f"Term C bonus unbounded under negative threshold: delta={delta}"
    )


def test_term_c_subsumes_elim_bonus_in_4p(monkeypatch):
    """In 4P, Term C REPLACES the discrete ELIMINATION_BONUS (does not
    stack with it). The Change 2 4P aggregation (max-of-opps) is held
    fixed by comparing strategic-with-Term-C-ON to strategic-with-Term-C-OFF;
    the F1/F2 difference vs `favor` (weighted-sum) is NOT in scope here.
    """
    obs = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0, 200, 8),  # me — strong
        (1, 1, 70.0, 50.0, 1.0,  30, 2),  # opp 1, strength = 30 + 2*15 = 60
        (2, 2, 50.0, 30.0, 1.0,   5, 2),  # opp 2 weakest, strength = 5 + 2*15 = 35
        (3, 3, 50.0, 70.0, 1.0,  40, 3),  # opp 3, strength = 40 + 3*15 = 85
    ])
    # Knobs OFF run: HOLD_HORIZON=0, FORWARD_REACH_WEIGHT=0, FINISH_BONUS=0.
    # → discrete elim_bonus = +55 fires (weakest_str=35 ≤ 110, my clears gate).
    v_off = _reload_value_with(monkeypatch, {})
    strat_off = v_off.favor_strategic(obs, 0, num_seats=4, gamma=0.99)

    # Knobs ON for Term C: FINISH_BONUS=50, THRESHOLD=200.
    # → Term C fires (per-opp sum), discrete elim_bonus SUPPRESSED.
    v_on = _reload_value_with(monkeypatch, {
        "BASELINE_FINISH_BONUS": "50",
        "BASELINE_FINISH_THRESHOLD": "200",
    })
    strat_on = v_on.favor_strategic(obs, 0, num_seats=4, gamma=0.99)

    # F1/F2 identical between off/on (knobs only gate the elim/Term-C path).
    # Δ = (Term C value) - (discrete elim_bonus = 55).
    # Term C: opp 1 (60) → 35; opp 2 (35) → 41.25; opp 3 (85) → 28.75. Sum ≈ 105.
    # Δ ≈ 105 - 55 = 50.
    delta = strat_on - strat_off
    expected_term_c = sum(
        50.0 * max(0.0, 1.0 - s/200.0) for s in (60.0, 35.0, 85.0)
    )
    expected_delta = expected_term_c - 55.0
    assert delta == pytest.approx(expected_delta, abs=1.0), (
        f"Term C sum/subsume wrong: delta={delta} expected≈{expected_delta} "
        f"(Term C alone ≈{expected_term_c}, discrete elim_bonus suppressed)"
    )


def test_term_c_credits_zero_strength_opp_fully(monkeypatch):
    """An opp at strength=0 is effectively finished. Term C now credits
    them with the FULL FINISH_BONUS — same as if they were gone from the
    obs entirely. This is the fix for the anti-elimination cliff.

    Hold F1/F2 fixed across the comparison by comparing
    favor_strategic-with-knobs-off vs favor_strategic-with-Term-C-on on
    the SAME obs (both use the same Change 2 max-of-opps aggregation).
    """
    obs = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0, 200, 8),
        (1, 1, 70.0, 50.0, 1.0, 100, 5),
        (2, 2, 50.0, 30.0, 1.0,   0, 0),  # opp 2 — effectively eliminated
        (3, 3, 50.0, 70.0, 1.0, 100, 5),
    ])
    v_off = _reload_value_with(monkeypatch, {})
    strat_off = v_off.favor_strategic(obs, 0, num_seats=4, gamma=0.99)
    v_on = _reload_value_with(monkeypatch, {
        "BASELINE_FINISH_BONUS": "50",
        "BASELINE_FINISH_THRESHOLD": "200",
    })
    strat_on = v_on.favor_strategic(obs, 0, num_seats=4, gamma=0.99)
    # Δ = Term C contribution − discrete elim_bonus (suppressed at on).
    # Term C: opp1@175 → 6.25; opp2@0 → 50.0 (the fix); opp3@175 → 6.25. Sum=62.5.
    # Discrete elim_bonus at off = +55 (weakest_str=0, gate trivially clears).
    # Δ ≈ 62.5 - 55 = 7.5.
    delta = strat_on - strat_off
    term_c_only = delta + 55.0
    expected = 50.0 + 2 * 50.0 * (1.0 - 175.0/200.0)  # 62.5
    assert term_c_only == pytest.approx(expected, abs=1.0), (
        f"Term C did not credit zero-strength opp fully: "
        f"term_c_only={term_c_only}, expected ~{expected}"
    )


def test_term_c_no_cliff_at_elimination_boundary(monkeypatch):
    """Continuity through elimination — when an opp transitions from
    strength=0 (still in obs) to gone-from-obs (truly eliminated),
    Term C must contribute the SAME amount in both cases. The prior
    `min over finishable_opps` filter dropped Term C to 0 the moment
    an opp left the finishable set; the new sum-over-expected-seats
    accounting routes through the same FINISH_BONUS credit in both
    boundary states.

    NOTE: this test is in 2P so the 4P 1.5×-weakest opp-aggregation
    (a pre-existing F1/F2 cliff in `favor`) does not confound the
    Term C-only continuity claim being tested here.
    """
    v = _reload_value_with(monkeypatch, {
        "BASELINE_FINISH_BONUS": "50",
        "BASELINE_FINISH_THRESHOLD": "200",
    })
    obs_zero_strength = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0, 200, 8),
        (1, 1, 70.0, 50.0, 1.0,   0, 0),  # opp in obs at exactly strength 0
    ])
    obs_gone = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0, 200, 8),
        # opp eliminated — not in obs
    ])
    s_zero = v.favor_strategic(obs_zero_strength, 0, num_seats=2, gamma=0.99)
    s_gone = v.favor_strategic(obs_gone, 0, num_seats=2, gamma=0.99)
    # In `obs_zero_strength`: opps = [1] with strength 0 → Term C +=
    #   50*(1 - 0/200) = 50. dead_count = 0.
    # In `obs_gone`: opps = []. dead_count = 1. Term C += 50.
    # Both states → Term C contribution = 50. Continuity.
    # F1 and F2 are 0/200/200 and equivalent in both states (no opp
    # ships, no opp prod). So the leaves match almost exactly.
    assert abs(s_gone - s_zero) < 1.0, (
        f"Term C cliff at zero-strength → gone boundary: "
        f"s_zero={s_zero} s_gone={s_gone} diff={s_gone - s_zero}"
    )


# -- 2026-05-26 modeling-correctness rewrite invariants --------------


def test_hold_discounted_prod_helper_is_symmetric(monkeypatch):
    """Direct unit test of `_hold_discounted_prod`: the same physics
    applies whether we discount my prod (using opp planets as threats)
    or opp prod (using my planets as threats). This is the symmetry
    that fixes Bug 1 — the prior version only discounted my side.
    """
    v = _reload_value_with(monkeypatch, {})  # knobs irrelevant; helper is direct
    planets = [
        (0, 0, 30.0, 50.0, 1.0,  50, 3),   # me — 50 ships, 3 prod
        (1, 1, 60.0, 50.0, 1.0, 100, 2),   # opp — 100 ships, 2 prod
    ]
    # Discount me's prod, with opp's planet as the threat source.
    threats_to_me = [(60.0, 50.0, 100.0)]   # opp planet attacking
    my_disc = v._hold_discounted_prod(planets, 0, threats_to_me, 30)
    # Discount opp's prod, with my planet as the threat source.
    threats_to_opp = [(30.0, 50.0, 50.0)]   # my planet attacking
    opp_disc = v._hold_discounted_prod(planets, 1, threats_to_opp, 30)

    # Me defender (50 ships) → capture-size = 51. Opp has 100 ≥ 51.
    # v=fleet_speed(51)≈3.15, dist=30, eta≈9.5, hold_score≈0.32 → my_disc≈0.95.
    assert 0.5 < my_disc < 1.5, (
        f"my_disc={my_disc} (expected ≈0.95 — opp can capture me)"
    )
    # Opp defender (100 ships) → capture-size = 101. Me has 50 < 101.
    # Threat ETA = inf → hold_score = 1.0 → opp_disc = 2 (un-discounted).
    assert opp_disc == pytest.approx(2.0, abs=1e-6), (
        f"opp_disc={opp_disc} (expected 2.0 — me can't credibly capture opp)"
    )


def test_realistic_threat_eta_speed_increases_with_capture_size(monkeypatch):
    """`_realistic_threat_eta` uses fleet_speed(capture_size), so a
    defender with HIGHER garrison gets faster threats (attacker must
    send more, larger fleets fly faster). Prior MIN_FLEET_SIZE-floor
    treated all defenders identically.
    """
    v = _reload_value_with(monkeypatch, {})
    # Same attacker (200 ships) and same distance (30 units). Vary
    # defender ship count → vary capture-size → vary speed → vary ETA.
    eta_weak = v._realistic_threat_eta(  # defender at 2 ships → capture-size 3
        d_ships=2, d_x=0.0, d_y=0.0, a_ships=200, a_x=30.0, a_y=0.0,
    )
    eta_strong = v._realistic_threat_eta(  # defender at 100 ships → capture-size 101
        d_ships=100, d_x=0.0, d_y=0.0, a_ships=200, a_x=30.0, a_y=0.0,
    )
    assert eta_strong < eta_weak, (
        f"Threat-ETA didn't drop with capture-size: weak={eta_weak} "
        f"strong={eta_strong} (large-defender attack flies faster — "
        f"attacker must send capture-size 101 ships, faster than capture-size 3)"
    )


def test_realistic_threat_eta_inf_when_attacker_too_weak(monkeypatch):
    """Attacker with fewer ships than capture-size returns inf (no
    credible threat) — replaces the previous ships>=2 hard filter
    with a physics-derived condition."""
    v = _reload_value_with(monkeypatch, {})
    eta = v._realistic_threat_eta(
        d_ships=100, d_x=0.0, d_y=0.0, a_ships=5, a_x=30.0, a_y=0.0,
    )
    assert eta == float("inf"), (
        f"Expected inf (attacker has 5 ships, needs 101 for capture) got {eta}"
    )


def test_term_b_per_source_speed_not_mean_garrison(monkeypatch):
    """Term B's reach should depend on the actual source planet's ships,
    not on a global mean. A 2-ship planet and a 100-ship planet attacking
    the same target should produce DIFFERENT reach contributions because
    fleet_speed differs.
    """
    v = _reload_value_with(monkeypatch, {
        "BASELINE_FORWARD_REACH_WEIGHT": "1.0",
        "BASELINE_FORWARD_REACH_HORIZON": "15",
    })
    # Target at distance 30. fleet_speed(2) ≈ 1.16 → eta = 30/1.16 ≈ 26
    # turns. fleet_speed(101) ≈ 3.45 → eta = 30/3.45 ≈ 8.7 turns.
    # With HORIZON=15: only the 100-ship source reaches; the 2-ship one
    # does NOT.
    # Source 2-ship board: my-planet has 2 ships; target has 100 (so
    # capture-size = 101 but launch = min(2, 101) = 2).
    obs_2ship = _mk_obs([
        (0, 0, 50.0, 50.0, 1.0,   2, 3),
        (1, 1, 80.0, 50.0, 1.0, 100, 2),
    ])
    # Source 100-ship board: same target.
    obs_100ship = _mk_obs([
        (0, 0, 50.0, 50.0, 1.0, 100, 3),
        (1, 1, 80.0, 50.0, 1.0, 100, 2),
    ])
    base_2 = v.favor(obs_2ship, 0, num_seats=2, gamma=0.99)
    base_100 = v.favor(obs_100ship, 0, num_seats=2, gamma=0.99)
    strat_2 = v.favor_strategic(obs_2ship, 0, num_seats=2, gamma=0.99)
    strat_100 = v.favor_strategic(obs_100ship, 0, num_seats=2, gamma=0.99)
    term_b_2 = strat_2 - base_2
    term_b_100 = strat_100 - base_100
    # The 100-ship source can reach (eta ≈ 8.7 < 15) → reach_sum += 2 (target prod).
    # The 2-ship source can't (eta ≈ 26 > 15) → reach_sum stays 0.
    assert term_b_100 > term_b_2, (
        f"Term B did not scale with source ship count: "
        f"term_b_2={term_b_2} term_b_100={term_b_100} "
        f"(2-ship source should have ZERO reach; 100-ship source should have positive reach)"
    )
    # Spot-check magnitudes
    assert abs(term_b_2) < 1e-6, (
        f"2-ship source reached target it shouldn't: term_b_2={term_b_2}"
    )
    assert term_b_100 > 1.5, (
        f"100-ship source reach too small: term_b_100={term_b_100}"
    )


def test_term_a_threat_eta_uses_capture_size_speed(monkeypatch):
    """Term A's threat-ETA should use the realistic capture-size launch
    speed, not the MIN_FLEET_SIZE floor. A me-planet with HIGH garrison
    is more threatened (faster opp capture-size launch) than a me-planet
    with LOW garrison at the same distance — the prior version (v_opp =
    fleet_speed(2)) treated both identically.
    """
    v = _reload_value_with(monkeypatch, {"BASELINE_HOLD_HORIZON": "30"})
    # Both boards: opp planet at distance 30 with enough ships to launch
    # a capture-size fleet. Vary MY garrison.
    # Low-garrison me: capture-size = 3 → fleet_speed(3) ≈ 1.27
    #                  eta = 30/1.27 ≈ 23.6 turns → hold_score ≈ 0.79.
    # High-garrison me: capture-size = 101 → fleet_speed(101) ≈ 3.45
    #                   eta = 30/3.45 ≈ 8.7 turns → hold_score ≈ 0.29.
    # So the high-garrison planet's hold_score should be LOWER (more
    # threatened) → my_prod_discounted lower → leaf lower for high-garrison.
    obs_low = _mk_obs([
        (0, 0, 50.0, 50.0, 1.0,   2, 3),
        (1, 1, 80.0, 50.0, 1.0, 200, 2),
    ])
    obs_high = _mk_obs([
        (0, 0, 50.0, 50.0, 1.0, 100, 3),
        (1, 1, 80.0, 50.0, 1.0, 200, 2),
    ])
    # F1 differs (my_ships 2 vs 100) — strip it to isolate Term A.
    s_low = v.favor_strategic(obs_low, 0, num_seats=2, gamma=0.99)
    s_high = v.favor_strategic(obs_high, 0, num_seats=2, gamma=0.99)
    # F1_low = 2 - 200 = -198. F1_high = 100 - 200 = -100.
    # ΔF1 = -100 - (-198) = 98.
    # Adjusted: (s_high - 98) compared to s_low → should be LOWER because
    # high-garrison version has Term A discount its 3-prod down more.
    s_high_minus_f1 = s_high - 98.0
    assert s_high_minus_f1 < s_low, (
        f"Term A did not fire stronger discount on high-garrison defender: "
        f"s_low={s_low} (effective Term A: hold ≈ 0.79) "
        f"s_high - ΔF1 = {s_high_minus_f1} (effective Term A: hold ≈ 0.29) — "
        f"expected the high-garrison version to be more discounted"
    )


# -- Dispatch --------------------------------------------------------


def test_select_favor_fn_routes_strategic(monkeypatch):
    """BASELINE_VALUE_HEAD=strategic dispatches to favor_strategic."""
    v = _reload_value_with(monkeypatch, {
        "BASELINE_VALUE_HEAD": "strategic",
    })
    f = v.select_favor_fn()
    assert f.__name__ == "favor_strategic"
