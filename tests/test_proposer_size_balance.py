"""Oracles for the unified fire-now size-balance fix (BASELINE_SIZE_BALANCE).

Targets failure modes D (under-delivery) and A (source over-drain), which
together are 58.3% of the current champion's lost-episode capture failures
(audit/2026-06-01-champion-failure-mix.md). The fix lives in the SOLO
proposer path — `enumerate_ship_counts` — because the chooser passes the
emitted ship count through untouched, so a correctly-sized candidate must
be GENERATED there or it never exists.

Three new units under test (agents/baseline/proposer.py):
  - `source_keep_floor`      — min residue to keep a source safe (A).
  - `capture_floor_arrival`  — arrival-correct capture size with a combat
                               -resolution margin (D).
  - `enumerate_ship_counts`  — flag-gated unified emission of both.

The flag is default-OFF and read at call time; OFF must be byte-identical
to the pre-fix set (Rule 46 parity).
"""

from __future__ import annotations

import math

from lib.intent import World
from lib.world_model import WorldModel, simulate_planet_timeline

from agents.baseline.proposer import (
    MIN_FLEET_SIZE,
    SIZE_BALANCE_CAPTURE_MARGIN,
    _source_survives_launch,
    aim_and_eta,
    capture_floor_arrival,
    capture_size,
    enumerate_ship_counts,
    source_keep_floor,
)


def _build(planets, *, step=0, omega=0.0, ledger=None, horizon=80):
    """Synthetic World + WorldModel from [pid, owner, x, y, r, ships, prod]
    rows. `ledger` is `{pid: [(eta, owner, ships), ...]}`. Mirrors the
    builder in test_proposer_bundling.py so model.ships_at /
    time_to_enemy_threat return realistic values."""
    obs = {
        "player": 0,
        "planets": planets,
        "fleets": [],
        "angular_velocity": omega,
        "initial_planets": [],
        "comet_planet_ids": [],
        "comets": [],
        "step": step,
    }
    world = World.from_obs(obs)
    base = {int(p[0]): [] for p in planets}
    if ledger:
        for pid, arr in ledger.items():
            base[int(pid)] = arr
    timelines = {
        p.id: simulate_planet_timeline(p, base.get(p.id, []), horizon)
        for p in world.planets_by_id.values()
    }
    model = WorldModel(ledger=base, timelines=timelines, horizon=horizon)
    return world, model


# ---------------------------------------------------------------------------
# Oracle 1 — flag OFF is byte-identical to the pre-fix set (Rule 46 parity)
# ---------------------------------------------------------------------------


def test_off_is_byte_identical_to_prefix_set(monkeypatch):
    """With BASELINE_SIZE_BALANCE unset, enumerate_ship_counts returns the
    exact pre-fix {cap, 2*cap, budget}-filtered set."""
    monkeypatch.delenv("BASELINE_SIZE_BALANCE", raising=False)
    planets = [
        [0, 0, 20.0, 20.0, 1.69, 60, 2],    # my source
        [10, -1, 70.0, 70.0, 1.69, 14, 2],  # neutral target
        [99, 1, 5.0, 90.0, 1.69, 10, 2],    # distant opp (plumbing)
    ]
    world, model = _build(planets)
    src, tgt = world.planets_by_id[0], world.planets_by_id[10]

    cap = capture_size(src, tgt, model, 0.0, 0, world)
    budget = int(src.ships)
    expected = set()
    if MIN_FLEET_SIZE <= cap <= budget:
        expected.add(cap)
    if 2 * cap <= budget:
        expected.add(2 * cap)
    if budget >= MIN_FLEET_SIZE:
        expected.add(budget)

    got = enumerate_ship_counts(src, tgt, model, 0.0, 0, world)
    assert got == sorted(expected), (got, sorted(expected))


def test_off_reinforce_path_unaffected(monkeypatch):
    """Owned (reinforce) targets must be untouched even with the flag ON —
    the balance block only fires for non-owned targets."""
    planets = [
        [0, 0, 20.0, 20.0, 1.69, 60, 2],   # my source
        [10, 0, 70.0, 70.0, 1.69, 8, 5],   # MY high-prod planet (reinforce)
        [99, 1, 5.0, 90.0, 1.69, 40, 3],   # opp threat to plumbing
    ]
    world, model = _build(planets)
    src, tgt = world.planets_by_id[0], world.planets_by_id[10]
    monkeypatch.delenv("BASELINE_SIZE_BALANCE", raising=False)
    off = enumerate_ship_counts(src, tgt, model, 0.0, 0, world)
    monkeypatch.setenv("BASELINE_SIZE_BALANCE", "1")
    on = enumerate_ship_counts(src, tgt, model, 0.0, 0, world)
    assert on == off, (on, off)


# ---------------------------------------------------------------------------
# Oracle 2 — D: arrival-correct capture floor with combat-resolution margin
# ---------------------------------------------------------------------------


def test_capture_floor_arrival_adds_combat_margin_on_neutral(monkeypatch):
    """For a neutral target (garrison constant across etas), the arrival
    floor is exactly capture_size + 1 — the extra combat-resolution ship
    (MARGIN=2 vs the legacy +1) that prevents the integer 'largest-minus-
    second' bounce. The arrival-correct column is emitted when feasible."""
    planets = [
        [0, 0, 20.0, 20.0, 1.69, 80, 2],    # my source (ample budget)
        [10, -1, 68.0, 68.0, 1.69, 8, 0],   # neutral target, 8 ships
        [99, 1, 5.0, 90.0, 1.69, 10, 2],    # distant opp (no source threat)
    ]
    world, model = _build(planets)
    src, tgt = world.planets_by_id[0], world.planets_by_id[10]

    cap = capture_size(src, tgt, model, 0.0, 0, world)
    cap_arr = capture_floor_arrival(src, tgt, model, 0.0, 0, world)
    # Neutral garrison is constant, so the only difference is the margin.
    assert cap_arr == cap + 1, (cap_arr, cap)
    assert cap_arr == int(tgt.ships) + SIZE_BALANCE_CAPTURE_MARGIN

    monkeypatch.setenv("BASELINE_SIZE_BALANCE", "1")
    on = enumerate_ship_counts(src, tgt, model, 0.0, 0, world)
    assert cap_arr in on, (cap_arr, on)


def test_capture_floor_arrival_is_arrival_correct(monkeypatch):
    """Arrival-correctness invariant (the D fix): the floor covers the
    predicted garrison at its OWN arrival tick (not the slow probe-eta
    garrison capture_size sizes against)."""
    planets = [
        [0, 0, 20.0, 20.0, 1.69, 200, 2],   # my source, big budget
        [10, 1, 78.0, 78.0, 2.61, 12, 3],   # opp target, grows in transit
        [99, 1, 5.0, 90.0, 1.69, 10, 2],
    ]
    world, model = _build(planets)
    src, tgt = world.planets_by_id[0], world.planets_by_id[10]

    cap_arr = capture_floor_arrival(src, tgt, model, 0.0, 0, world)
    _a, eta = aim_and_eta(src, tgt, cap_arr, 0.0, world=world)
    garrison_at_arrival = math.ceil(model.ships_at(int(tgt.id), eta) or 0.0)
    assert cap_arr >= garrison_at_arrival + 1, (cap_arr, garrison_at_arrival, eta)


# ---------------------------------------------------------------------------
# Oracle 3 — A: every emitted size respects the source-keep floor
# ---------------------------------------------------------------------------


def test_source_keep_floor_respected_by_all_sizes(monkeypatch):
    """Source under an in-flight enemy threat: every fire-now size emitted
    with the flag ON leaves residue >= source_keep_floor and passes
    _source_survives_launch. The pre-fix budget column drained below it."""
    planets = [
        [0, 0, 50.0, 20.0, 1.69, 40, 2],    # my source, threatened
        [10, -1, 55.0, 25.0, 1.69, 8, 1],   # neutral target nearby
        [99, 1, 50.0, 12.0, 1.69, 30, 2],   # opp near source
    ]
    # enemy fleet inbound to source 0: 25 ships at eta=6
    world, model = _build(planets, ledger={0: [(6, 1, 25)]})
    src, tgt = world.planets_by_id[0], world.planets_by_id[10]

    keep = source_keep_floor(src, 0, world, model, 0)
    assert keep > 0, keep

    monkeypatch.setenv("BASELINE_SIZE_BALANCE", "1")
    on = enumerate_ship_counts(src, tgt, model, 0.0, 0, world)
    assert on, "expected at least one feasible balanced size"
    for size in on:
        assert int(src.ships) - size >= keep, (size, keep)
        assert _source_survives_launch(src, size, 0, world, model, 0)

    # Pre-fix budget column (full src.ships) violates the keep floor.
    monkeypatch.delenv("BASELINE_SIZE_BALANCE", raising=False)
    off = enumerate_ship_counts(src, tgt, model, 0.0, 0, world)
    assert int(src.ships) in off
    assert not _source_survives_launch(src, int(src.ships), 0, world, model, 0)


# ---------------------------------------------------------------------------
# Oracle 4 — infeasible launch is suppressed (can't win AND keep source)
# ---------------------------------------------------------------------------


def test_infeasible_launch_emits_no_fire_now(monkeypatch):
    """When the arrival-correct capture floor exceeds what the source can
    spare, the candidate is correctly invalid: flag ON emits no fire-now
    column, while the pre-fix path emits a doomed (under-sized) launch."""
    planets = [
        [0, 0, 20.0, 20.0, 1.69, 60, 2],    # my source, only 60 ships
        [10, 1, 78.0, 78.0, 2.61, 12, 3],   # opp target needs ~83 at arrival
        [99, 1, 5.0, 90.0, 1.69, 10, 2],
    ]
    world, model = _build(planets)
    src, tgt = world.planets_by_id[0], world.planets_by_id[10]

    cap_arr = capture_floor_arrival(src, tgt, model, 0.0, 0, world)
    keep = source_keep_floor(src, 0, world, model, 0)
    assert cap_arr > int(src.ships) - keep, (cap_arr, src.ships, keep)

    monkeypatch.setenv("BASELINE_SIZE_BALANCE", "1")
    on = enumerate_ship_counts(src, tgt, model, 0.0, 0, world)
    assert on == [], on

    # Pre-fix: a doomed under-sized launch is emitted (the mode-D failure).
    monkeypatch.delenv("BASELINE_SIZE_BALANCE", raising=False)
    off = enumerate_ship_counts(src, tgt, model, 0.0, 0, world)
    assert off, "pre-fix should emit the (doomed) budget column"
    assert max(off) < cap_arr, (off, cap_arr)


# ---------------------------------------------------------------------------
# Oracle 5 — source_keep_floor self-consistency at the residue boundary
# ---------------------------------------------------------------------------


def test_source_keep_floor_self_consistency():
    """A launch leaving exactly `source_keep_floor` residue passes the
    filter; one ship more sent fails it. And no in-flight threat → floor 0."""
    planets = [
        [0, 0, 50.0, 20.0, 1.69, 40, 2],
        [10, -1, 55.0, 25.0, 1.69, 8, 1],
        [99, 1, 50.0, 12.0, 1.69, 30, 2],
    ]
    world, model = _build(planets, ledger={0: [(6, 1, 25)]})
    src = world.planets_by_id[0]
    keep = source_keep_floor(src, 0, world, model, 0)
    assert keep > 0

    send_exact = int(src.ships) - keep            # residue == keep
    assert _source_survives_launch(src, send_exact, 0, world, model, 0)
    assert not _source_survives_launch(src, send_exact + 1, 0, world, model, 0)

    # No in-flight threat (empty ledger) → keep floor is 0.
    world2, model2 = _build(planets)  # no ledger entry on source
    src2 = world2.planets_by_id[0]
    assert source_keep_floor(src2, 0, world2, model2, 0) == 0
