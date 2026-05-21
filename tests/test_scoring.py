"""Unit tests for lib.scoring — projected-arrival helpers used by ROI variants."""

from __future__ import annotations

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib import scoring


def _planet(pid, owner, x, y, ships, production):
    return Planet(pid, owner, x, y, 1.0, ships, production)


def test_eta_proxy_zero_distance_is_zero():
    p = _planet(0, 0, 50, 50, 100, 1)
    assert scoring.eta_proxy(p, p) == 0


def test_eta_proxy_positive_for_typical_pair():
    src = _planet(0, 0, 10, 10, 100, 1)
    tgt = _planet(1, -1, 50, 50, 30, 2)
    assert scoring.eta_proxy(src, tgt) > 0


def test_projected_garrison_neutral_does_not_grow():
    tgt = _planet(1, -1, 50, 50, 20, 5)
    assert scoring.projected_garrison(tgt, eta=10) == 20


def test_projected_garrison_enemy_grows_by_production_times_eta():
    tgt = _planet(1, 1, 50, 50, 20, 3)
    assert scoring.projected_garrison(tgt, eta=10) == 20 + 30


def test_s_needed_strict_win():
    tgt = _planet(1, 1, 50, 50, 10, 2)
    assert scoring.s_needed(tgt, eta=5) == 10 + 10 + 1


def test_horizon_clamps_to_zero_late_game():
    assert scoring.horizon(step=500, eta=10) == 0
    assert scoring.horizon(step=499, eta=10) == 0
    assert scoring.horizon(step=400, eta=10) == 90


def test_margin_multiplier_enemy_is_2_neutral_is_1_self_is_0():
    enemy = _planet(1, 1, 0, 0, 0, 0)
    neutral = _planet(2, -1, 0, 0, 0, 0)
    own = _planet(3, 0, 0, 0, 0, 0)
    assert scoring.margin_multiplier(enemy, my_id=0) == 2
    assert scoring.margin_multiplier(neutral, my_id=0) == 1
    assert scoring.margin_multiplier(own, my_id=0) == 0


# ---------------------------------------------------------------------------
# pv_horizon — present-value discount of future production
# (TID 699003 / hypothesis H16)
# ---------------------------------------------------------------------------


def test_pv_horizon_gamma_one_matches_linear_horizon():
    # At γ = 1.0 pv_horizon should reduce to the integer `horizon()`
    # function (modulo float cast) so existing snipe/reinforce score
    # values stay numerically identical.
    for step, eta in [(0, 5), (100, 20), (400, 10), (499, 1), (500, 0)]:
        linear = scoring.horizon(step=step, eta=eta)
        pv = scoring.pv_horizon(step=step, eta=eta, gamma=1.0)
        assert pv == float(linear), (step, eta, linear, pv)


def test_pv_horizon_zero_at_or_past_game_end():
    assert scoring.pv_horizon(step=500, eta=0, gamma=0.99) == 0.0
    assert scoring.pv_horizon(step=400, eta=200, gamma=0.99) == 0.0
    assert scoring.pv_horizon(step=300, eta=300, gamma=0.99) == 0.0


def test_pv_horizon_discounts_far_arrivals_more_than_near_arrivals():
    # Same target lifetime, different eta — near-eta is worth more in PV.
    near = scoring.pv_horizon(step=0, eta=5, gamma=0.99)
    far = scoring.pv_horizon(step=0, eta=80, gamma=0.99)
    assert near > far


def test_pv_horizon_asymptotes_to_one_over_one_minus_gamma():
    # At γ = 0.99 with eta=0 and a 500-turn horizon the geometric
    # series sum approaches 1/(1−γ) = 100. Linear horizon at the same
    # point is 500 — five times larger. PV compresses long horizons.
    pv = scoring.pv_horizon(step=0, eta=0, gamma=0.99)
    assert 95.0 < pv < 100.0
    linear = scoring.horizon(step=0, eta=0)
    assert linear == 500
    assert pv < linear / 4


def test_pv_horizon_default_gamma_is_linear_for_backwards_compat():
    # Behaviour test independent of module-level PV_GAMMA (which is now
    # env-var configurable; another agent imported in the same process
    # might have set PV_GAMMA=0.99). Pass gamma=1.0 explicitly so this
    # test pins the linear-horizon equivalence regardless of env state.
    pv = scoring.pv_horizon(step=100, eta=10, gamma=1.0)
    assert pv == 390.0


def test_pv_horizon_strictly_monotone_in_gamma_for_fixed_step_eta():
    # For fixed (step, eta) with h > 0, pv_horizon is monotone non-
    # decreasing in γ — more patient discounting => more total value.
    vals = [scoring.pv_horizon(step=0, eta=10, gamma=g) for g in [0.90, 0.95, 0.99, 1.0]]
    assert vals == sorted(vals)


# ---------------------------------------------------------------------------
# expected_hold — HAV horizon cap (2026-05-14 plan, HAV-1)
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal WorldModel stub: time_to_enemy_threat returns whatever
    is preset; other methods unused by expected_hold. Records the
    `arrival_eta` kwarg seen so tests can pin the orbital-safety
    wiring (f1774a7 / 2026-05-22)."""
    def __init__(self, threat_eta=None):
        self._threat = threat_eta
        self.last_arrival_eta = None
    def time_to_enemy_threat(self, planet_id, my_id, world, arrival_eta=0):
        self.last_arrival_eta = arrival_eta
        return self._threat


class _FakeWorld:
    def __init__(self, step=10, my_id=0):
        self.step = step
        self.my_id = my_id


def test_expected_hold_saturates_at_remaining_game_when_no_threat():
    world = _FakeWorld(step=100)
    model = _FakeModel(threat_eta=None)
    # remaining_game = 500 - 100 - 20 = 380.
    assert scoring.expected_hold(target_id=7, eta=20, world=world, model=model) == 380


def test_expected_hold_caps_by_threat_eta_when_threat_arrives_soon():
    world = _FakeWorld(step=100)
    model = _FakeModel(threat_eta=30)  # threat in 30 turns from now
    # eta=20, threat=30 → hold = 30 - 20 = 10.
    assert scoring.expected_hold(target_id=7, eta=20, world=world, model=model) == 10


def test_expected_hold_zero_when_threat_arrives_before_us():
    world = _FakeWorld(step=100)
    model = _FakeModel(threat_eta=15)  # threat arrives BEFORE our eta
    assert scoring.expected_hold(target_id=7, eta=20, world=world, model=model) == 0


def test_expected_hold_zero_at_game_end():
    world = _FakeWorld(step=499)
    model = _FakeModel(threat_eta=None)
    assert scoring.expected_hold(target_id=7, eta=1, world=world, model=model) == 0


def test_expected_hold_respects_smaller_remaining_game():
    # Game has 50 turns left after our arrival; threat would saturate
    # at 200 turns out. Hold caps at remaining game.
    world = _FakeWorld(step=440)
    model = _FakeModel(threat_eta=200)
    # remaining_game = 500 - 440 - 10 = 50; hold via threat = 200 - 10 = 190.
    # min(50, 190) = 50.
    assert scoring.expected_hold(target_id=7, eta=10, world=world, model=model) == 50


# ---------------------------------------------------------------------------
# Orbital-safety wiring pin (f1774a7 / 2026-05-22)
# ---------------------------------------------------------------------------


def test_expected_hold_passes_arrival_eta_when_env_set(monkeypatch):
    """When BASELINE_ORBITAL_SAFETY=1, expected_hold must thread `eta`
    into `time_to_enemy_threat` as `arrival_eta`. Pins the wiring
    landed in f1774a7."""
    world = _FakeWorld(step=100)
    model = _FakeModel(threat_eta=30)
    monkeypatch.setenv("BASELINE_ORBITAL_SAFETY", "1")
    scoring.expected_hold(target_id=7, eta=20, world=world, model=model)
    assert model.last_arrival_eta == 20, (
        f"Expected arrival_eta=20, got {model.last_arrival_eta}"
    )


def test_expected_hold_does_not_pass_arrival_eta_when_env_unset(monkeypatch):
    """When BASELINE_ORBITAL_SAFETY unset (default), `arrival_eta`
    defaults to 0 in the stub — preserves backwards compat with sub
    52882014."""
    world = _FakeWorld(step=100)
    model = _FakeModel(threat_eta=30)
    monkeypatch.delenv("BASELINE_ORBITAL_SAFETY", raising=False)
    scoring.expected_hold(target_id=7, eta=20, world=world, model=model)
    assert model.last_arrival_eta == 0


def test_expected_hold_does_not_pass_arrival_eta_when_env_zero(monkeypatch):
    """`BASELINE_ORBITAL_SAFETY=0` (explicit) behaves identical to unset."""
    world = _FakeWorld(step=100)
    model = _FakeModel(threat_eta=30)
    monkeypatch.setenv("BASELINE_ORBITAL_SAFETY", "0")
    scoring.expected_hold(target_id=7, eta=20, world=world, model=model)
    assert model.last_arrival_eta == 0
