"""Tests for the large-idle-stockpile spend-down post-pass (PI 2026-06-03).

Mechanism: after the normal pipeline + enforce_launch_rules settles, any of
MY planets that (a) qualifies as a stockpile under the geometry-adaptive
threshold (3x avg OR >25% of my total fleet) and (b) hasn't fired this turn
and (c) has no inbound enemy fleet gets ONE forced launch at an opponent —
positive-EV preferred, nearest-opp fallback. K-eta cap bypassed by
post-enforce slotting.

These unit tests verify the helpers and the post-pass under controlled
synthetic worlds. End-to-end (replay reproduces failure state) verification
is done via a fast.py play smoke + a diagnostic probe — Rule 38.
"""

from __future__ import annotations

import pytest

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


# ---------------------------------------------------------------------------
# Fixtures: tiny stub world / model / planets matching what main.py expects.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    """Clean env state before each test so module-level constants do not leak."""
    for var in (
        "BASELINE_IDLE_STOCKPILE_DRAIN",
        "BASELINE_IDLE_STOCKPILE_REL_MULT",
        "BASELINE_IDLE_STOCKPILE_SHARE",
        "BASELINE_IDLE_STOCKPILE_FLOOR",
        "BASELINE_IDLE_STOCKPILE_GARRISON",
        "BASELINE_IDLE_STOCKPILE_MIN_SEND",
        "BASELINE_IDLE_STOCKPILE_MAX",
        "BASELINE_LAUNCH_RULES",
        "BASELINE_ADAPTIVE_K",
    ):
        monkeypatch.delenv(var, raising=False)


def _planet(pid, owner, x, y, ships, production=2, radius=2.0):
    """Build a real Planet so `list(p)` (used inside aim_and_eta) and
    Planet's __iter__ work. Field order is
    (id, owner, x, y, radius, ships, production)."""
    return Planet(pid, owner, float(x), float(y), float(radius),
                  int(ships), int(production))


class _StubModel:
    """Minimal model stub. `threatened` planet ids return non-None from both
    incoming_enemy_eta and time_to_enemy_threat (mimicking either an in-flight
    enemy fleet OR a stationary opp planet that could plausibly launch at us).
    `flipping` planet ids return a non-my owner from owner_at (mimicking a
    multi-wave attack in the ledger that flips the planet)."""

    def __init__(self, ledger=None, threatened=None, flipping=None,
                 flip_owner=1):
        self.ledger = ledger or {}
        self._threatened = threatened or set()
        self._flipping = flipping or set()
        self._flip_owner = flip_owner

    def incoming_enemy_eta(self, planet_id, my_id):
        return 7 if int(planet_id) in self._threatened else None

    def time_to_enemy_threat(self, planet_id, my_id, world, arrival_eta=0):
        return 7 if int(planet_id) in self._threatened else None

    def owner_at(self, planet_id, step):
        if int(planet_id) in self._flipping:
            return self._flip_owner
        return None  # unknown -> caller treats as safe


class _StubWorld:
    """Minimal world stub. Module functions in main.py call aim_and_eta which
    routes through proposer's geometry; we re-import that function so we get
    the real behavior. The world only needs angular_velocity and a step."""

    def __init__(self, omega=0.0, step=20):
        self.omega = omega
        self.step = step
        self.obs_raw = {"angular_velocity": omega}


# ---------------------------------------------------------------------------
# _pick_idle_stockpile_target: positive-EV preference & fallback.
# ---------------------------------------------------------------------------


def test_pick_target_returns_positive_ev_opp(monkeypatch):
    monkeypatch.setenv("BASELINE_IDLE_STOCKPILE_DRAIN", "1")
    from agents.baseline.main import _pick_idle_stockpile_target

    src = _planet(0, 0, 0.0, 0.0, 200)
    # Weak opp at moderate distance — capture feasible
    weak_opp = _planet(1, 1, 30.0, 0.0, 10, production=2)
    # Strong opp closer — capture NOT feasible (200 vs 500)
    strong_opp = _planet(2, 1, 20.0, 0.0, 500, production=2)
    planets = [src, weak_opp, strong_opp]
    model = _StubModel()
    world = _StubWorld()

    tgt, angle, eta = _pick_idle_stockpile_target(
        src, send=195, planets=planets, my_id=0, world=world, model=model,
        omega=0.0,
    )
    assert tgt is not None
    assert int(tgt.id) == 1, f"expected weak opp (id=1), got {tgt.id}"


def test_pick_target_returns_positive_ev_neutral(monkeypatch):
    monkeypatch.setenv("BASELINE_IDLE_STOCKPILE_DRAIN", "1")
    from agents.baseline.main import _pick_idle_stockpile_target

    src = _planet(0, 0, 0.0, 0.0, 100)
    # Strong opp — uncapturable
    strong_opp = _planet(1, 1, 20.0, 0.0, 500, production=3)
    # Neutral with low garrison — easy capture
    weak_neutral = _planet(2, -1, 25.0, 0.0, 5, production=1)
    planets = [src, strong_opp, weak_neutral]
    model = _StubModel()
    world = _StubWorld()

    tgt, angle, eta = _pick_idle_stockpile_target(
        src, send=95, planets=planets, my_id=0, world=world, model=model,
        omega=0.0,
    )
    assert tgt is not None
    assert int(tgt.id) == 2, f"expected weak neutral (id=2), got {tgt.id}"


def test_pick_target_falls_back_to_nearest_opp(monkeypatch):
    monkeypatch.setenv("BASELINE_IDLE_STOCKPILE_DRAIN", "1")
    from agents.baseline.main import _pick_idle_stockpile_target

    src = _planet(0, 0, 0.0, 0.0, 50)
    # Strong opp close — uncapturable
    near_strong = _planet(1, 1, 20.0, 0.0, 500, production=3)
    # Stronger opp farther — also uncapturable
    far_strong = _planet(2, 1, 60.0, 0.0, 800, production=3)
    # Neutral with HIGH garrison — also uncapturable
    big_neutral = _planet(3, -1, 25.0, 0.0, 200, production=1)
    planets = [src, near_strong, far_strong, big_neutral]
    model = _StubModel()
    world = _StubWorld()

    tgt, angle, eta = _pick_idle_stockpile_target(
        src, send=45, planets=planets, my_id=0, world=world, model=model,
        omega=0.0,
    )
    # Fallback should be the nearest OPPONENT (not the neutral).
    assert tgt is not None
    assert int(tgt.id) == 1, f"expected nearest opp (id=1), got {tgt.id}"


def test_pick_target_returns_none_when_no_opps(monkeypatch):
    monkeypatch.setenv("BASELINE_IDLE_STOCKPILE_DRAIN", "1")
    from agents.baseline.main import _pick_idle_stockpile_target

    src = _planet(0, 0, 0.0, 0.0, 100)
    # Only a strong neutral exists — no opponent at all.
    strong_neutral = _planet(1, -1, 30.0, 0.0, 500, production=1)
    planets = [src, strong_neutral]
    model = _StubModel()
    world = _StubWorld()

    tgt, angle, eta = _pick_idle_stockpile_target(
        src, send=95, planets=planets, my_id=0, world=world, model=model,
        omega=0.0,
    )
    assert tgt is None and angle is None and eta is None


# ---------------------------------------------------------------------------
# drain_idle_stockpile_to_opp: trigger semantics & gating.
# ---------------------------------------------------------------------------


def test_drain_triggers_on_3x_avg_stockpile(monkeypatch):
    monkeypatch.setenv("BASELINE_IDLE_STOCKPILE_DRAIN", "1")
    from agents.baseline.main import drain_idle_stockpile_to_opp

    # Average across my 4 planets = (20+20+20+200)/4 = 65
    # Stockpile @ 200 ships > 3 * 65 = 195 → qualifies on REL_MULT.
    stockpile = _planet(0, 0, 0.0, 0.0, 200)
    small_a = _planet(1, 0, 5.0, 0.0, 20)
    small_b = _planet(2, 0, 0.0, 5.0, 20)
    small_c = _planet(3, 0, -5.0, 0.0, 20)
    weak_opp = _planet(4, 1, 40.0, 0.0, 15, production=2)
    planets = [stockpile, small_a, small_b, small_c, weak_opp]
    model = _StubModel()
    world = _StubWorld()

    moves = []  # nothing has launched this turn
    result = drain_idle_stockpile_to_opp(
        moves, planets, my_id=0, world=world, model=model, omega=0.0,
    )
    # Expected: exactly one forced launch from the stockpile.
    src_launches = [m for m in result if int(m[0]) == 0]
    assert len(src_launches) == 1
    assert int(src_launches[0][2]) == 200 - 5  # GARRISON=5


def test_drain_triggers_on_share_of_total_clause(monkeypatch):
    """Single mega-planet game: avg is huge so the 3x clause never fires.
    The 25%-share-of-total OR-clause must catch this case."""
    monkeypatch.setenv("BASELINE_IDLE_STOCKPILE_DRAIN", "1")
    from agents.baseline.main import drain_idle_stockpile_to_opp

    # Two MY planets, both large. avg=350, 3*avg=1050, neither hits REL_MULT.
    # But the 400-ship planet holds 400/700 = 57% > 25% of total → SHARE clause.
    big = _planet(0, 0, 0.0, 0.0, 400)
    bigger = _planet(1, 0, 5.0, 0.0, 300)
    # Wait — let's tune: big=400 holds 400/700 = 57% (qualifies);
    # bigger=300 holds 300/700 = 43% (also qualifies). Make first launch
    # check: at least one fires.
    weak_opp = _planet(2, 1, 60.0, 0.0, 10, production=2)
    planets = [big, bigger, weak_opp]
    model = _StubModel()
    world = _StubWorld()

    result = drain_idle_stockpile_to_opp(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    assert len(result) >= 1, "expected ≥1 forced launch via share-of-total clause"


def test_drain_skips_threatened_source(monkeypatch):
    monkeypatch.setenv("BASELINE_IDLE_STOCKPILE_DRAIN", "1")
    from agents.baseline.main import drain_idle_stockpile_to_opp

    stockpile = _planet(0, 0, 0.0, 0.0, 200)
    smalls = [_planet(i, 0, float(i), 0.0, 20) for i in (1, 2, 3)]
    weak_opp = _planet(4, 1, 40.0, 0.0, 15, production=2)
    planets = [stockpile] + smalls + [weak_opp]
    # Mark the stockpile as having an inbound enemy fleet.
    model = _StubModel(threatened={0})
    world = _StubWorld()

    result = drain_idle_stockpile_to_opp(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    src_launches = [m for m in result if int(m[0]) == 0]
    assert src_launches == [], "must not drain a threatened source"


def test_drain_skips_source_that_flips_at_lookahead(monkeypatch):
    """Belt-and-suspenders gate: even with no current threat, a source
    predicted to belong to opp at horizon K (multi-wave attack visible
    in the ledger) must not be drained."""
    monkeypatch.setenv("BASELINE_IDLE_STOCKPILE_DRAIN", "1")
    from agents.baseline.main import drain_idle_stockpile_to_opp

    stockpile = _planet(0, 0, 0.0, 0.0, 200)
    smalls = [_planet(i, 0, float(i), 0.0, 20) for i in (1, 2, 3)]
    weak_opp = _planet(4, 1, 40.0, 0.0, 15, production=2)
    planets = [stockpile] + smalls + [weak_opp]
    # No CURRENT threat, but the ledger predicts the planet flips to opp.
    model = _StubModel(flipping={0}, flip_owner=1)
    world = _StubWorld()

    result = drain_idle_stockpile_to_opp(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    src_launches = [m for m in result if int(m[0]) == 0]
    assert src_launches == [], "must not drain a source predicted to flip"


def test_drain_skips_source_already_in_moves(monkeypatch):
    monkeypatch.setenv("BASELINE_IDLE_STOCKPILE_DRAIN", "1")
    from agents.baseline.main import drain_idle_stockpile_to_opp

    stockpile = _planet(0, 0, 0.0, 0.0, 200)
    smalls = [_planet(i, 0, float(i), 0.0, 20) for i in (1, 2, 3)]
    weak_opp = _planet(4, 1, 40.0, 0.0, 15, production=2)
    planets = [stockpile] + smalls + [weak_opp]
    model = _StubModel()
    world = _StubWorld()

    # The chooser already fired src 0 this turn (some existing launch).
    pre_moves = [[0, 0.0, 50]]
    result = drain_idle_stockpile_to_opp(
        pre_moves, planets, my_id=0, world=world, model=model, omega=0.0,
    )
    extras = [m for m in result if m not in pre_moves]
    assert all(int(m[0]) != 0 for m in extras), \
        "must not double-fire the same source in the same turn"


def test_drain_is_noop_when_disabled(monkeypatch):
    # Env var NOT set → default OFF → pass-through.
    # Reload main.py to re-evaluate the module-level constant after the
    # reset_env autouse fixture has cleared the env var (a prior test
    # may have set it ON and the constant would otherwise still be True).
    import importlib
    import agents.baseline.main as main_mod
    importlib.reload(main_mod)
    drain_idle_stockpile_to_opp = main_mod.drain_idle_stockpile_to_opp

    stockpile = _planet(0, 0, 0.0, 0.0, 200)
    smalls = [_planet(i, 0, float(i), 0.0, 20) for i in (1, 2, 3)]
    weak_opp = _planet(4, 1, 40.0, 0.0, 15, production=2)
    planets = [stockpile] + smalls + [weak_opp]
    model = _StubModel()
    world = _StubWorld()

    result = drain_idle_stockpile_to_opp(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    assert result == [], "lever-OFF must return the input moves unchanged"


def test_drain_respects_max_per_turn(monkeypatch):
    monkeypatch.setenv("BASELINE_IDLE_STOCKPILE_DRAIN", "1")
    monkeypatch.setenv("BASELINE_IDLE_STOCKPILE_MAX", "2")
    # IDLE_STOCKPILE_MAX_PER_TURN is read at module-import time. We need to
    # re-import to pick up the env override.
    import importlib
    import agents.baseline.main as main_mod
    importlib.reload(main_mod)
    drain_idle_stockpile_to_opp = main_mod.drain_idle_stockpile_to_opp

    # 5 stockpile planets, all qualifying. Cap = 2 → at most 2 forced launches.
    stockpiles = [_planet(i, 0, float(i) * 3.0, 0.0, 300) for i in range(5)]
    # Add a small "average puller" so the REL_MULT branch fires
    smalls = [_planet(100 + i, 0, float(i), 5.0, 20) for i in range(3)]
    weak_opp = _planet(200, 1, 50.0, 0.0, 10, production=2)
    planets = stockpiles + smalls + [weak_opp]
    model = _StubModel()
    world = _StubWorld()

    result = drain_idle_stockpile_to_opp(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    assert len(result) <= 2, \
        f"MAX_PER_TURN=2 must cap forced launches, got {len(result)}"
