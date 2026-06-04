"""Tests for the frontier-circulation post-pass v2 (PI 2026-06-03 + Biel).

v2 mechanism: per-friendly-planet enemy pressure = distance-decayed enemy
ship mass. Ships flow UP the pressure gradient. Launches fire only when
destination pressure exceeds source pressure by DELTA_MIN AND destination
is reachable within ETA_CAP turns.

v1 (centroid-based) failed two ways at n=16: 5/16 wins AND turn-ms blowup
(p95=1452 > 1000ms cap). v2's short ETA cap caps the in-flight friendly
fleet population, fixing the wallclock cascade.

End-to-end verification (replay reproduces failure state) is done via a
fast.py play smoke + a diagnostic probe — Rule 38.
"""

from __future__ import annotations

import math

import pytest

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


def _reload_main():
    import importlib
    import agents.baseline.main as main_mod
    importlib.reload(main_mod)
    return main_mod


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    for var in (
        "BASELINE_FRONTIER_CIRCULATION",
        "BASELINE_CIRCULATION_PRESSURE_HORIZON",
        "BASELINE_CIRCULATION_PRESSURE_DELTA_MIN",
        "BASELINE_CIRCULATION_ETA_CAP",
        "BASELINE_CIRCULATION_GARRISON",
        "BASELINE_CIRCULATION_TRIGGER_MIN",
        "BASELINE_CIRCULATION_MIN_SEND",
        "BASELINE_CIRCULATION_MAX",
        "BASELINE_CIRCULATION_MIN_THREAT_ETA",
        "BASELINE_LAUNCH_RULES",
    ):
        monkeypatch.delenv(var, raising=False)
    yield
    # Teardown: previous tests may have reloaded the module while their env
    # overrides were set; reload again on the way out to leave a clean
    # module for the next test.
    _reload_main()


def _planet(pid, owner, x, y, ships, production=2, radius=2.0):
    return Planet(pid, owner, float(x), float(y), float(radius),
                  int(ships), int(production))


class _StubModel:
    def __init__(self, threatened=None, threat_eta=7, ships_snapshot=None):
        """Stub model for circulation tests.

        `ships_snapshot` is an optional dict {planet_id -> ships} consulted by
        `ships_at(planet_id, eta)`. The v3 destination-usefulness filter calls
        `capture_floor_arrival`, which calls `model.ships_at(tgt.id, eta)` to
        predict defender strength at arrival. We return a static snapshot for
        a no-production / no-in-flight world."""
        self.ledger = {}
        self._threatened = threatened or set()
        self._threat_eta = threat_eta
        self._ships_snapshot = ships_snapshot or {}

    def time_to_enemy_threat(self, planet_id, my_id, world, arrival_eta=0):
        return self._threat_eta if int(planet_id) in self._threatened else None

    def ships_at(self, planet_id, step):
        return self._ships_snapshot.get(int(planet_id))


class _StubWorld:
    def __init__(self):
        self.step = 20
        self.obs_raw = {"angular_velocity": 0.0}


# ---------------------------------------------------------------------------
# Pressure-field unit tests
# ---------------------------------------------------------------------------


def test_pressure_zero_with_no_enemies():
    from agents.baseline.main import _compute_enemy_pressure
    my = [_planet(0, 0, 0.0, 0.0, 50), _planet(1, 0, 5.0, 0.0, 50)]
    out = _compute_enemy_pressure(my, [], horizon=18)
    assert all(v == 0.0 for v in out.values())


def test_pressure_concentrates_near_enemies():
    """A friendly near a strong enemy has much higher pressure than a
    friendly far from any enemy."""
    from agents.baseline.main import _compute_enemy_pressure

    near = _planet(0, 0, 10.0, 0.0, 50)   # close to enemy at x=20
    far = _planet(1, 0, -200.0, 0.0, 50)  # far from enemy
    enemy = _planet(2, 1, 20.0, 0.0, 100)
    pressure = _compute_enemy_pressure([near, far], [enemy], horizon=18)
    assert pressure[0] > 0.0, "near friendly must have positive pressure"
    assert pressure[0] > pressure[1]
    # Decay verification: at d=10, reach = fleet_speed(100) * 18 (big), so
    # pressure(near) ≈ ships * (1 - 10/reach), strictly less than ships.
    assert pressure[0] < 100.0


def test_pressure_drops_outside_reach():
    """An enemy too far to plausibly reach contributes zero pressure."""
    from agents.baseline.main import _compute_enemy_pressure

    my = [_planet(0, 0, 0.0, 0.0, 50)]
    # fleet_speed of 1 ship is 1.0; reach at H=1 is 1.0. Enemy at d=100
    # is far outside reach -> contribution 0.
    far_enemy = _planet(1, 1, 100.0, 0.0, 1)
    out = _compute_enemy_pressure(my, [far_enemy], horizon=1)
    assert out[0] == 0.0


# ---------------------------------------------------------------------------
# emit_frontier_circulation behavior
# ---------------------------------------------------------------------------


def test_fires_toward_higher_pressure_friendly(monkeypatch):
    """src -> dst when dst has strictly higher pressure (it's nearer the
    enemy) AND dst is reachable within ETA_CAP AND v3 dst can capture an
    enemy today (weak neutral adjacent to front gives it a target)."""
    monkeypatch.setenv("BASELINE_FRONTIER_CIRCULATION", "1")
    main_mod = _reload_main()

    rear = _planet(0, 0, 0.0, 0.0, 100)
    front = _planet(1, 0, 20.0, 0.0, 30)
    # Strong enemy provides PRESSURE near front.
    enemy = _planet(2, 1, 30.0, 0.0, 200)
    # Weak NEUTRAL adjacent to front gives v3 a capture-feasible target.
    weak_neutral = _planet(3, -1, 22.0, 0.0, 5)
    planets = [rear, front, enemy, weak_neutral]
    model = _StubModel()
    world = _StubWorld()

    result = main_mod.emit_frontier_circulation(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    src_launches = [m for m in result if int(m[0]) == 0]
    assert len(src_launches) == 1, \
        f"expected 1 launch from rear, got {len(src_launches)}"
    _, angle, ships = src_launches[0]
    # Launch must aim at +x (toward front planet).
    assert abs(angle) < 0.1
    assert int(ships) == 100 - 5  # everything minus garrison


def test_skips_when_no_higher_pressure_destination(monkeypatch):
    """Front planet (highest pressure) must NOT fire — no forward friendly
    has strictly higher pressure."""
    monkeypatch.setenv("BASELINE_FRONTIER_CIRCULATION", "1")
    main_mod = _reload_main()
    emit_frontier_circulation = main_mod.emit_frontier_circulation

    rear = _planet(0, 0, 0.0, 0.0, 100)
    front = _planet(1, 0, 20.0, 0.0, 100)
    enemy = _planet(2, 1, 30.0, 0.0, 200)
    planets = [rear, front, enemy]
    model = _StubModel()
    world = _StubWorld()

    result = emit_frontier_circulation(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    src_ids = {int(m[0]) for m in result}
    assert int(front.id) not in src_ids


def test_skips_when_destination_outside_eta_cap(monkeypatch):
    """A higher-pressure friendly that is too FAR (eta > ETA_CAP) must
    not be picked as destination. This is the wallclock-safety mechanism."""
    monkeypatch.setenv("BASELINE_FRONTIER_CIRCULATION", "1")
    monkeypatch.setenv("BASELINE_CIRCULATION_ETA_CAP", "5")
    import importlib
    import agents.baseline.main as main_mod
    importlib.reload(main_mod)

    rear = _planet(0, 0, 0.0, 0.0, 100)
    # Forward friendly is 500 units away — ETA easily > 5.
    far_front = _planet(1, 0, 500.0, 0.0, 30)
    # Enemy near the far friendly: gives the far friendly high pressure.
    enemy = _planet(2, 1, 510.0, 0.0, 200)
    planets = [rear, far_front, enemy]
    model = main_mod_StubModel = _StubModel()
    world = _StubWorld()

    result = main_mod.emit_frontier_circulation(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    src_launches = [m for m in result if int(m[0]) == 0]
    assert src_launches == [], \
        "ETA-out-of-range destination must NOT fire (wallclock guard)"


def test_pressure_delta_gate_blocks_marginal_trips(monkeypatch):
    """If dst_pressure - src_pressure < DELTA_MIN the launch is blocked.
    Set DELTA_MIN very high so even genuine deltas don't clear."""
    monkeypatch.setenv("BASELINE_FRONTIER_CIRCULATION", "1")
    monkeypatch.setenv("BASELINE_CIRCULATION_PRESSURE_DELTA_MIN", "1e9")
    import importlib
    import agents.baseline.main as main_mod
    importlib.reload(main_mod)

    rear = _planet(0, 0, 0.0, 0.0, 100)
    front = _planet(1, 0, 20.0, 0.0, 30)
    enemy = _planet(2, 1, 30.0, 0.0, 200)
    planets = [rear, front, enemy]
    model = _StubModel()
    world = _StubWorld()

    result = main_mod.emit_frontier_circulation(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    assert result == [], "huge DELTA_MIN must block all launches"


def test_skips_imminent_threat_source(monkeypatch):
    """Source-safety: planet whose threat ETA is < MIN_THREAT_ETA must
    not be drained, preserving local defence."""
    monkeypatch.setenv("BASELINE_FRONTIER_CIRCULATION", "1")
    main_mod = _reload_main()
    emit_frontier_circulation = main_mod.emit_frontier_circulation

    rear = _planet(0, 0, 0.0, 0.0, 100)
    front = _planet(1, 0, 20.0, 0.0, 30)
    enemy = _planet(2, 1, 30.0, 0.0, 200)
    planets = [rear, front, enemy]
    model = _StubModel(threatened={0}, threat_eta=7)
    world = _StubWorld()

    result = emit_frontier_circulation(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    src_ids = {int(m[0]) for m in result}
    assert 0 not in src_ids


def test_fires_when_threat_is_distant(monkeypatch):
    monkeypatch.setenv("BASELINE_FRONTIER_CIRCULATION", "1")
    main_mod = _reload_main()
    emit_frontier_circulation = main_mod.emit_frontier_circulation

    rear = _planet(0, 0, 0.0, 0.0, 100)
    front = _planet(1, 0, 20.0, 0.0, 30)
    enemy = _planet(2, 1, 30.0, 0.0, 200)
    # Weak neutral keeps the v3 destination-usefulness filter open.
    weak_neutral = _planet(3, -1, 22.0, 0.0, 5)
    planets = [rear, front, enemy, weak_neutral]
    model = _StubModel(threatened={0}, threat_eta=30)  # 30 >= 15 default
    world = _StubWorld()

    result = emit_frontier_circulation(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    src_ids = {int(m[0]) for m in result}
    assert 0 in src_ids


def test_skips_source_already_in_moves(monkeypatch):
    monkeypatch.setenv("BASELINE_FRONTIER_CIRCULATION", "1")
    main_mod = _reload_main()
    emit_frontier_circulation = main_mod.emit_frontier_circulation

    rear = _planet(0, 0, 0.0, 0.0, 100)
    front = _planet(1, 0, 20.0, 0.0, 30)
    enemy = _planet(2, 1, 30.0, 0.0, 200)
    planets = [rear, front, enemy]
    model = _StubModel()
    world = _StubWorld()

    pre_moves = [[0, 0.0, 50]]
    result = emit_frontier_circulation(
        pre_moves, planets, my_id=0, world=world, model=model, omega=0.0,
    )
    extras = [m for m in result if m not in pre_moves]
    assert all(int(m[0]) != 0 for m in extras)


def test_respects_max_per_turn(monkeypatch):
    monkeypatch.setenv("BASELINE_FRONTIER_CIRCULATION", "1")
    monkeypatch.setenv("BASELINE_CIRCULATION_MAX", "2")
    import importlib
    import agents.baseline.main as main_mod
    importlib.reload(main_mod)
    emit_frontier_circulation = main_mod.emit_frontier_circulation

    # 5 rear planets in a column, 1 forward friendly near the enemy.
    rears = [_planet(i, 0, 0.0, float(i * 2), 50) for i in range(5)]
    front = _planet(99, 0, 20.0, 0.0, 30)
    enemy = _planet(100, 1, 30.0, 0.0, 200)
    # Weak neutral keeps the v3 destination-usefulness filter open.
    weak_neutral = _planet(101, -1, 22.0, 0.0, 5)
    planets = rears + [front, enemy, weak_neutral]
    model = _StubModel()
    world = _StubWorld()

    result = emit_frontier_circulation(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    assert len(result) <= 2


def test_noop_when_disabled():
    import importlib
    import agents.baseline.main as main_mod
    importlib.reload(main_mod)

    rear = _planet(0, 0, 0.0, 0.0, 100)
    front = _planet(1, 0, 20.0, 0.0, 30)
    enemy = _planet(2, 1, 30.0, 0.0, 200)
    planets = [rear, front, enemy]
    model = _StubModel()
    world = _StubWorld()

    result = main_mod.emit_frontier_circulation(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    assert result == []


def test_noop_with_no_opponents(monkeypatch):
    monkeypatch.setenv("BASELINE_FRONTIER_CIRCULATION", "1")
    main_mod = _reload_main()
    emit_frontier_circulation = main_mod.emit_frontier_circulation

    rear = _planet(0, 0, 0.0, 0.0, 100)
    front = _planet(1, 0, 20.0, 0.0, 30)
    neutral = _planet(2, -1, 30.0, 0.0, 200)
    planets = [rear, front, neutral]
    model = _StubModel()
    world = _StubWorld()

    result = emit_frontier_circulation(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    assert result == []


# ---------------------------------------------------------------------------
# v3: destination-usefulness filter
# ---------------------------------------------------------------------------


def test_v3_destination_with_capturable_enemy_fires(monkeypatch):
    """v3: dst with at least one capturable enemy in its 8 nearest
    qualifies. A high-pressure forward friendly + weak neutral target =>
    dst can fire today => circulation lands there."""
    monkeypatch.setenv("BASELINE_FRONTIER_CIRCULATION", "1")
    main_mod = _reload_main()

    rear = _planet(0, 0, 0.0, 0.0, 100)
    front = _planet(1, 0, 20.0, 0.0, 30)
    pressure_enemy = _planet(2, 1, 30.0, 0.0, 200)  # gives front pressure
    weak_neutral = _planet(3, -1, 22.0, 0.0, 5)     # capturable by front
    planets = [rear, front, pressure_enemy, weak_neutral]
    model = _StubModel()
    world = _StubWorld()

    result = main_mod.emit_frontier_circulation(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    src_launches = [m for m in result if int(m[0]) == 0]
    assert len(src_launches) == 1, \
        "v3: dst with capturable enemy must qualify; circulation fires"


def test_v3_destination_without_capturable_enemy_blocks(monkeypatch):
    """v3: dst whose 8 nearest non-our planets are all UNCAPTURABLE (too
    strong) must be skipped, even when pressure-delta and ETA gates pass.
    Ships shouldn't pile up at a forward planet our chooser ignores."""
    monkeypatch.setenv("BASELINE_FRONTIER_CIRCULATION", "1")
    main_mod = _reload_main()

    rear = _planet(0, 0, 0.0, 0.0, 100)
    front = _planet(1, 0, 20.0, 0.0, 30)
    # Strong enemy gives front high pressure but is uncapturable (200 vs 30).
    strong_enemy = _planet(2, 1, 30.0, 0.0, 200)
    # The only neutral nearby is also too strong for front to capture.
    strong_neutral = _planet(3, -1, 25.0, 0.0, 500)
    planets = [rear, front, strong_enemy, strong_neutral]
    # ships_snapshot lets capture_floor_arrival predict defender strength.
    model = _StubModel(
        ships_snapshot={2: 200, 3: 500},
    )
    world = _StubWorld()

    result = main_mod.emit_frontier_circulation(
        [], planets, my_id=0, world=world, model=model, omega=0.0,
    )
    src_launches = [m for m in result if int(m[0]) == 0]
    assert src_launches == [], (
        "v3: dst with no capturable enemy must block circulation; "
        "we don't pile ships at planets our chooser ignores"
    )
