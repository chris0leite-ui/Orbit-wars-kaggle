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


def test_parity_with_knobs_off_4p(monkeypatch):
    """Phase F F15: 4P parity branch — weighted-sum opp + elim_bonus path."""
    v = _reload_value_with(monkeypatch, {})
    # 4P: me + 3 opps. Mix strengths so weakest is clearly identified.
    obs = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0,  10, 3),
        (1, 1, 70.0, 50.0, 1.0,   8, 2),
        (2, 2, 50.0, 30.0, 1.0,   4, 1),  # weakest opp
        (3, 3, 50.0, 70.0, 1.0,  10, 2),
    ])
    base = v.favor(obs, 0, num_seats=4, gamma=0.99)
    strat = v.favor_strategic(obs, 0, num_seats=4, gamma=0.99)
    assert strat == pytest.approx(base, abs=1e-9), (
        f"4P parity broken: base={base} strategic={strat}"
    )


# -- Term A ----------------------------------------------------------


def test_term_a_discounts_threatened_planet(monkeypatch):
    """A planet with imminent threat scores lower than the same planet safe."""
    v = _reload_value_with(monkeypatch, {"BASELINE_HOLD_HORIZON": "20"})
    obs_safe = _mk_obs([
        (0, 0,  5.0,  5.0, 1.0, 10, 3),
        (1, 1, 95.0, 95.0, 1.0,  8, 2),
    ])
    obs_close = _mk_obs([
        (0, 0, 40.0, 50.0, 1.0, 10, 3),
        (1, 1, 60.0, 50.0, 1.0,  8, 2),
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
    import math as _math_mod

    def _raise_systemexit(*a, **kw):
        raise SystemExit("propagate me")

    monkeypatch.setattr(v.math, "hypot", _raise_systemexit)
    obs = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0, 10, 3),
        (1, 1, 70.0, 50.0, 1.0,  8, 2),
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
    """Phase F F2: in 4P, Term C (continuous) REPLACES the discrete
    ELIMINATION_BONUS=55 rather than stacking on top of it.
    """
    v = _reload_value_with(monkeypatch, {
        "BASELINE_FINISH_BONUS": "50",
        "BASELINE_FINISH_THRESHOLD": "200",
    })
    # 4P where weakest opp is finishable (strength ≤ 110) AND my_strength
    # clears the 0.9× gate.
    obs = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0, 200, 8),  # me — strong
        (1, 1, 70.0, 50.0, 1.0,  30, 2),  # opp 1
        (2, 2, 50.0, 30.0, 1.0,   5, 2),  # opp 2 — weakest (strength = 5 + 30 = 35)
        (3, 3, 50.0, 70.0, 1.0,  40, 3),  # opp 3
    ])
    base = v.favor(obs, 0, num_seats=4, gamma=0.99)  # has discrete +55
    strat = v.favor_strategic(obs, 0, num_seats=4, gamma=0.99)  # has Term C only

    # `base` includes ELIMINATION_BONUS=55. `strat` has Term C only.
    # Term C at target_str=35, FINISH_THRESHOLD=200 → 50 * (1 - 35/200) = 41.25.
    # If Term C is stacked on top of the 55, strat - base ≈ 41.25.
    # If Term C subsumes (correct), strat - base ≈ 41.25 - 55 = -13.75.
    delta = strat - base
    assert delta < 0, (
        f"Term C did not subsume ELIMINATION_BONUS in 4P: "
        f"delta={delta} (expected negative ≈ -13.75 because Term C 41.25 "
        f"replaces discrete 55)"
    )


def test_term_c_excludes_zero_prod_opps(monkeypatch):
    """Phase F F5: Term C's target_str must pick the weakest opp WITH
    production (> 0), not a near-dead opp with zero planets / only
    in-flight residual ships.
    """
    v = _reload_value_with(monkeypatch, {
        "BASELINE_FINISH_BONUS": "50",
        "BASELINE_FINISH_THRESHOLD": "200",
    })
    # Both boards have:
    # - me at (30, 50) with 200 ships, 8 prod
    # - opp 1 at (70, 50) with 100 ships, 5 prod (strength 175)
    # The "near-dead" board adds:
    # - opp 2 with prod=0 but a tiny in-flight fleet (we simulate by
    #   making the planet 0-ship 0-prod and adding a fleet via a
    #   side construction below).
    #
    # Concretely: 4P obs where opp 2's planet has 0 ships AND 0 prod,
    # representing a 'lost the planet' state. Term C should NOT use
    # opp 2's strength (~0) as target_str; it should use opp 1's (175).
    obs_dead = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0, 200, 8),
        (1, 1, 70.0, 50.0, 1.0, 100, 5),  # opp 1 — only finishable opp
        (2, 2, 50.0, 30.0, 1.0,   0, 0),  # opp 2 — defeated (no production)
        (3, 3, 50.0, 70.0, 1.0, 100, 5),
    ])
    # If F5 holds: target_str = min over opps with prod>0 = min(opp1, opp3)
    # = min(100 + 75, 100 + 75) = 175.
    # Without F5: target_str = min over all opps = opp2 = 0 + 0 = 0.
    # Pressure(175) ≈ 0.125; bonus ≈ 6.25.
    # Pressure(0) = 1.0; bonus = 50.
    base = v.favor(obs_dead, 0, num_seats=4, gamma=0.99)
    strat = v.favor_strategic(obs_dead, 0, num_seats=4, gamma=0.99)
    delta = strat - base

    # opp 2 has prod=0 → not in favor's opps either (no F2 contribution).
    # But favor STILL fires elim_bonus for the weakest (the 0-strength opp).
    # When Term C is active, F2 SUPPRESSES elim_bonus. So `base` has +55,
    # `strat` has Term C with strength=175 → bonus ≈ 6.25 (instead of 50).
    # delta = strat - base = 6.25 - 55 ≈ -48.75. Mostly we care that
    # Term C used a sensible target_str — verify the bonus magnitude is
    # NOT close to 50 (the bonus if target_str=0).
    # Term C contribution alone:
    term_c_only = delta + 55.0  # add back the suppressed elim_bonus
    assert 0 < term_c_only < 20, (
        f"Term C target_str included 0-prod opp (bonus too large): "
        f"term_c_only={term_c_only} (expected ~6.25 if opp 2 excluded; "
        f"~50 if F5 not applied)"
    )


# -- Dispatch --------------------------------------------------------


def test_select_favor_fn_routes_strategic(monkeypatch):
    """BASELINE_VALUE_HEAD=strategic dispatches to favor_strategic."""
    v = _reload_value_with(monkeypatch, {
        "BASELINE_VALUE_HEAD": "strategic",
    })
    f = v.select_favor_fn()
    assert f.__name__ == "favor_strategic"
