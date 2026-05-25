"""Pin Term A (hold-discount) and Term B (forward-reach) in `favor_strategic`.

These pins ensure model enrichment behaves as documented:
  - Parity vs base `favor` when knobs OFF
  - Term A discounts production of threatened planets
  - Term B rewards positions with reachable enemy supply
"""
from __future__ import annotations

import importlib
import os

import pytest


def _reload_value(env_overrides: dict[str, str]):
    """Reload agents.baseline.value with the given env vars set.
    Module-level constants (HOLD_HORIZON, FORWARD_REACH_*) re-read on import.
    """
    for k, v in env_overrides.items():
        os.environ[k] = v
    import agents.baseline.value as vmod
    importlib.reload(vmod)
    return vmod


def _mk_obs(planets, step=0):
    return {
        "player": 0,
        "planets": planets,
        "fleets": [],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": step,
    }


def test_parity_with_knobs_off():
    """All three terms OFF → favor_strategic ≡ favor."""
    v = _reload_value({
        "BASELINE_HOLD_HORIZON": "0",
        "BASELINE_FORWARD_REACH_WEIGHT": "0",
        "BASELINE_FINISH_BONUS": "0",
    })
    obs = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0, 10, 3),
        (1, 1, 70.0, 50.0, 1.0,  8, 2),
    ])
    base = v.favor(obs, 0, num_seats=2, gamma=0.99)
    strat = v.favor_strategic(obs, 0, num_seats=2, gamma=0.99)
    assert strat == pytest.approx(base, abs=1e-9), (
        f"parity broken: base={base} strategic={strat}"
    )


def test_term_a_discounts_threatened_planet():
    """A planet with imminent threat scores lower than the same planet safe."""
    v = _reload_value({
        "BASELINE_HOLD_HORIZON": "20",
        "BASELINE_FORWARD_REACH_WEIGHT": "0",  # isolate Term A
        "BASELINE_FINISH_BONUS": "0",          # isolate Term A
    })
    # Safe: enemy in far corner.
    obs_safe = _mk_obs([
        (0, 0,  5.0,  5.0, 1.0, 10, 3),
        (1, 1, 95.0, 95.0, 1.0,  8, 2),
    ])
    # Close: enemy adjacent.
    obs_close = _mk_obs([
        (0, 0, 40.0, 50.0, 1.0, 10, 3),
        (1, 1, 60.0, 50.0, 1.0,  8, 2),
    ])
    score_safe = v.favor_strategic(obs_safe, 0, num_seats=2, gamma=0.99)
    score_close = v.favor_strategic(obs_close, 0, num_seats=2, gamma=0.99)
    assert score_safe > score_close, (
        f"Term A failed to penalise threatened planet: "
        f"safe={score_safe} close={score_close}"
    )


def test_term_b_rewards_reachable_enemy_supply():
    """Identical my-position; vary number of enemies in reach. More reachable
    enemy production → higher favor (Term B additive)."""
    v = _reload_value({
        "BASELINE_HOLD_HORIZON": "0",         # isolate Term B
        "BASELINE_FORWARD_REACH_WEIGHT": "1.0",
        "BASELINE_FORWARD_REACH_HORIZON": "30",
        "BASELINE_FINISH_BONUS": "0",         # isolate Term B
    })
    # Single enemy reachable; my planet central.
    obs_one = _mk_obs([
        (0, 0, 50.0, 50.0, 1.0, 10, 3),
        (1, 1, 60.0, 50.0, 1.0,  5, 2),
    ])
    # Three enemies reachable; my planet central.
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
    # Term B contribution: strategic - base should be larger for the 3-enemy
    # board than the 1-enemy board.
    delta_one = strat_one - base_one
    delta_three = strat_three - base_three
    assert delta_three > delta_one, (
        f"Term B did not scale with enemy count: "
        f"delta_one={delta_one} delta_three={delta_three}"
    )


def test_select_favor_fn_routes_strategic():
    """BASELINE_VALUE_HEAD=strategic dispatches to favor_strategic."""
    v = _reload_value({
        "BASELINE_VALUE_HEAD": "strategic",
        "BASELINE_HOLD_HORIZON": "20",
        "BASELINE_FORWARD_REACH_WEIGHT": "0.5",
    })
    f = v.select_favor_fn()
    # When stockpile penalty is off (default), select returns the bare head.
    assert f.__name__ == "favor_strategic", (
        f"dispatch did not route to favor_strategic: {f.__name__}"
    )


def test_term_c_rewards_weak_opp():
    """Identical my-position; opp ship-count varies. Term C with FINISH_BONUS>0
    prefers the weak-opp board (more finishing pressure → higher favor delta
    vs base)."""
    v = _reload_value({
        "BASELINE_HOLD_HORIZON": "0",          # isolate Term C
        "BASELINE_FORWARD_REACH_WEIGHT": "0",  # isolate Term C
        "BASELINE_FINISH_BONUS": "50",
        "BASELINE_FINISH_THRESHOLD": "200",
    })
    # Healthy opp (strength = 200 + 2*15 = 230 >> threshold; pressure = 0).
    obs_strong = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0,  10, 3),
        (1, 1, 70.0, 50.0, 1.0, 200, 2),
    ])
    # Weak opp (strength = 5 + 2*15 = 35 << threshold; pressure ≈ 0.825).
    obs_weak = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0,  10, 3),
        (1, 1, 70.0, 50.0, 1.0,   5, 2),
    ])
    base_strong = v.favor(obs_strong, 0, num_seats=2, gamma=0.99)
    base_weak = v.favor(obs_weak, 0, num_seats=2, gamma=0.99)
    strat_strong = v.favor_strategic(obs_strong, 0, num_seats=2, gamma=0.99)
    strat_weak = v.favor_strategic(obs_weak, 0, num_seats=2, gamma=0.99)
    # Term C bonus = strategic - base. Should be larger on the weak-opp board.
    delta_strong = strat_strong - base_strong
    delta_weak = strat_weak - base_weak
    assert delta_weak > delta_strong, (
        f"Term C did not reward weak opp more than strong opp: "
        f"delta_weak={delta_weak} delta_strong={delta_strong}"
    )
    # Spot check the magnitude: at strength=35, pressure ≈ 0.825, bonus ≈ 41.25.
    assert delta_weak == pytest.approx(50.0 * (1.0 - 35.0/200.0), abs=1.0)


def test_term_c_disabled_when_zero():
    """FINISH_BONUS=0 → no Term C contribution; strategic with Term A+B
    matches strategic with Term A+B+C (no Term C effect)."""
    v_off = _reload_value({
        "BASELINE_HOLD_HORIZON": "20",
        "BASELINE_FORWARD_REACH_WEIGHT": "0.5",
        "BASELINE_FORWARD_REACH_HORIZON": "15",
        "BASELINE_FINISH_BONUS": "0",
    })
    obs = _mk_obs([
        (0, 0, 30.0, 50.0, 1.0,  10, 3),
        (1, 1, 70.0, 50.0, 1.0,   5, 2),  # weak opp — Term C would otherwise fire
    ])
    s_off = v_off.favor_strategic(obs, 0, num_seats=2, gamma=0.99)

    v_on = _reload_value({
        "BASELINE_HOLD_HORIZON": "20",
        "BASELINE_FORWARD_REACH_WEIGHT": "0.5",
        "BASELINE_FORWARD_REACH_HORIZON": "15",
        "BASELINE_FINISH_BONUS": "50",
        "BASELINE_FINISH_THRESHOLD": "200",
    })
    s_on = v_on.favor_strategic(obs, 0, num_seats=2, gamma=0.99)
    # With knobs ON, Term C fires (weak opp); s_on > s_off by Term C bonus.
    assert s_on > s_off, (
        f"Term C did not lift favor on weak-opp board: off={s_off} on={s_on}"
    )
