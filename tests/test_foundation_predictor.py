"""Step 6 — Foundation `Predictor` API tests.

Two tiers tested:
- **O(1) position queries** — assert `planet_position` / `comet_
  position` / `fleet_position` match forward-simulated positions
  from `lib.foundation.jax_engine.step`.
- **O(horizon) timeline queries** — assert `arrival_ledger` and
  `ships_at_planet` match the world-model timeline, with and without
  hypothetical launches.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from lib.foundation.actions import ActionSpec
from lib.foundation.jax_engine import step
from lib.foundation.memory import EmptyMemory
from lib.foundation.predictor import Arrival, Predictor
from lib.game.jax.conversions import actions_to_jax, jax_to_scalar_obs, scalar_to_jax
from lib.game.jax.jax_types import MAX_PLANETS


def _build_state(seed: int = 42, num_agents: int = 2):
    """Build a JAX GameState from a fresh env. Convenience wrapper
    reused by every test below."""
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=num_agents)
    return scalar_to_jax(env.state, env.info["seed"])


def _forward_state(state, turns: int, num_agents: int = 2):
    """Forward-simulate `state` by `turns` no-op steps; returns the
    resulting state. Used as a ground-truth oracle for the
    closed-form predictor."""
    pids, angles, ships = actions_to_jax([[]] * num_agents, num_agents)
    for _ in range(turns):
        state = step(state, pids, angles, ships)
    return state


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_predictor_construction():
    """Predictor builds from a state; exposes state/memory/omega."""
    state = _build_state()
    p = Predictor(state)
    assert p.state is state
    assert isinstance(p.memory, EmptyMemory)
    assert p.omega == float(state.angular_velocity)


def test_predictor_with_explicit_memory():
    """When memory is supplied, the Predictor exposes it."""
    state = _build_state()
    custom_mem = EmptyMemory()
    p = Predictor(state, memory=custom_mem)
    assert p.memory is custom_mem


# ---------------------------------------------------------------------------
# O(1) position queries: planets
# ---------------------------------------------------------------------------


def test_planet_position_t0_returns_current():
    """At t=0, planet_position returns the current state's
    (planets_x, planets_y) — no projection needed."""
    state = _build_state(seed=42)
    p = Predictor(state)
    planets = p._planets_list()
    for planet in planets[:5]:
        pos = p.planet_position(planet.id, t=0)
        assert pos is not None
        assert math.isclose(pos[0], planet.x, abs_tol=1e-6)
        assert math.isclose(pos[1], planet.y, abs_tol=1e-6)


def test_planet_position_matches_forward_sim_orbiting():
    """Closed-form planet_position at relative turn t matches the
    forward-simulated state's planet position. Tolerance 1e-3 (float
    drift between Python-double orbital math and JAX float32 step)."""
    state = _build_state(seed=42)
    p = Predictor(state)
    for t in [5, 15, 30]:
        future = _forward_state(state, turns=t)
        # Compare every alive non-comet planet.
        alive = np.asarray(future.planets_alive)
        is_comet = np.asarray(future.is_comet)
        ids = np.asarray(future.planets_id)
        x = np.asarray(future.planets_x)
        y = np.asarray(future.planets_y)
        for i in range(MAX_PLANETS):
            if not alive[i] or is_comet[i]:
                continue
            pid = int(ids[i])
            pred = p.planet_position(pid, t=t)
            assert pred is not None, f"pid={pid} t={t}: predictor returned None"
            assert abs(pred[0] - float(x[i])) < 1e-3, (
                f"pid={pid} t={t} x: pred={pred[0]} truth={float(x[i])}"
            )
            assert abs(pred[1] - float(y[i])) < 1e-3, (
                f"pid={pid} t={t} y: pred={pred[1]} truth={float(y[i])}"
            )


def test_planet_position_unknown_id_returns_none():
    """planet_position(unknown_id) returns None."""
    state = _build_state()
    p = Predictor(state)
    assert p.planet_position(9999, t=5) is None


def test_planet_position_by_index_out_of_range_returns_none():
    """planet_position with by_id=False rejects bad indices."""
    state = _build_state()
    p = Predictor(state)
    assert p.planet_position(-1, t=5, by_id=False) is None
    assert p.planet_position(MAX_PLANETS, t=5, by_id=False) is None


# ---------------------------------------------------------------------------
# O(1) position queries: comets
# ---------------------------------------------------------------------------


def test_comet_position_matches_forward_sim():
    """After a comet spawn boundary, predict the comet's future
    position via comet_position and compare to forward-sim. Match
    must be bit-exact (path-array lookup)."""
    state = _build_state(seed=42)
    # Step past the step-50 comet spawn boundary.
    state = _forward_state(state, turns=55)
    p = Predictor(state)

    # Find a comet planet id.
    is_comet = np.asarray(state.is_comet)
    alive = np.asarray(state.planets_alive)
    ids = np.asarray(state.planets_id)
    comet_ids = [int(ids[i]) for i in range(MAX_PLANETS) if alive[i] and is_comet[i]]
    if not comet_ids:
        pytest.skip("no comets after step-55; seed-dependent")

    # Predict at t=3 turns.
    for cid in comet_ids[:2]:
        pred = p.comet_position(cid, t=3)
        if pred is None:
            continue  # expired before t=3
        future = _forward_state(state, turns=3)
        # Find the same comet in the future state (it may have
        # expired or stayed; only check if it's still alive).
        future_alive = np.asarray(future.planets_alive)
        future_ids = np.asarray(future.planets_id)
        future_x = np.asarray(future.planets_x)
        future_y = np.asarray(future.planets_y)
        match = None
        for i in range(MAX_PLANETS):
            if future_alive[i] and int(future_ids[i]) == cid:
                match = (float(future_x[i]), float(future_y[i]))
                break
        if match is None:
            continue  # expired between now and t=3
        assert abs(pred[0] - match[0]) < 1e-3
        assert abs(pred[1] - match[1]) < 1e-3


def test_comet_position_far_future_returns_none():
    """A comet's predicted position past its path end returns None
    (expired)."""
    state = _build_state(seed=42)
    state = _forward_state(state, turns=55)
    p = Predictor(state)
    is_comet = np.asarray(state.is_comet)
    alive = np.asarray(state.planets_alive)
    ids = np.asarray(state.planets_id)
    comet_ids = [int(ids[i]) for i in range(MAX_PLANETS) if alive[i] and is_comet[i]]
    if not comet_ids:
        pytest.skip("no comets after step-55; seed-dependent")
    # Comet paths are validated to len <= 40; t=200 is well past.
    assert p.comet_position(comet_ids[0], t=200) is None


def test_comet_position_on_non_comet_returns_none():
    """Asking for the comet position of a non-comet planet returns
    None (cleaner than throwing)."""
    state = _build_state()
    p = Predictor(state)
    planets = p._planets_list()
    non_comets = [pl for pl in planets if pl.id < 20]  # initial planets are pid < 20
    if not non_comets:
        pytest.skip("no non-comet planets; unusual seed")
    assert p.comet_position(non_comets[0].id, t=5) is None


# ---------------------------------------------------------------------------
# O(1) position queries: fleets
# ---------------------------------------------------------------------------


def test_fleet_position_straight_line_projection():
    """fleet_position is a closed-form straight-line projection. With
    no collisions in the way, it matches forward-sim position."""
    state = _build_state(seed=42)
    # Launch a fleet so there's something to track.
    planets = Predictor(state)._planets_list()
    src = next((p for p in planets if p.owner == 0 and p.ships > 5), None)
    if src is None:
        pytest.skip("no seat-0 planet with >5 ships at seed=42 init")

    # Build a launch action via JAX.
    action = [[[src.id, 0.0, src.ships // 2]], []]  # seat 0 fires; seat 1 empty
    pids, angles, ships = actions_to_jax(action, 2)
    state = step(state, pids, angles, ships)

    p = Predictor(state)
    fleets = p._fleets_list()
    if not fleets:
        pytest.skip("fleet did not survive the launch step (terrain?)")
    fleet = fleets[0]
    # Project 3 turns ahead via predictor.
    pred = p.fleet_position(fleet.id, t=3)
    assert pred is not None

    # Forward-sim 3 turns and compare. (May diverge if fleet hit
    # something; check fleet still alive.)
    future = _forward_state(state, turns=3, num_agents=2)
    f_alive = np.asarray(future.fleets_alive)
    f_ids = np.asarray(future.fleets_id)
    f_x = np.asarray(future.fleets_x)
    f_y = np.asarray(future.fleets_y)
    match = None
    for i in range(len(f_alive)):
        if f_alive[i] and int(f_ids[i]) == fleet.id:
            match = (float(f_x[i]), float(f_y[i]))
            break
    if match is None:
        pytest.skip("fleet hit something within 3 turns; closed-form would diverge")
    # Tolerance accommodates float32 vs float64 drift.
    assert abs(pred[0] - match[0]) < 1e-2
    assert abs(pred[1] - match[1]) < 1e-2


def test_fleet_position_unknown_id_returns_none():
    state = _build_state()
    p = Predictor(state)
    assert p.fleet_position(9999, t=5) is None


# ---------------------------------------------------------------------------
# positions_at — aggregate query
# ---------------------------------------------------------------------------


def test_positions_at_returns_expected_shapes():
    """positions_at returns three numpy arrays with shape (N, 2)."""
    state = _build_state(seed=42)
    p = Predictor(state)
    pos = p.positions_at(t=5)
    assert set(pos.keys()) == {"planets", "comets", "fleets"}
    for k, v in pos.items():
        assert isinstance(v, np.ndarray), f"{k}: not ndarray"
        assert v.dtype == np.float32, f"{k}: wrong dtype"
        assert v.ndim == 2 and v.shape[1] == 2, f"{k}: shape={v.shape}"


def test_positions_at_planet_entries_match_planet_position():
    """positions_at['planets'] entries match per-planet
    planet_position calls."""
    state = _build_state(seed=42)
    p = Predictor(state)
    t = 7
    pos = p.positions_at(t=t)

    # Build the same list manually.
    is_comet = np.asarray(state.is_comet)
    alive = np.asarray(state.planets_alive)
    manual: list[tuple[float, float]] = []
    for i in range(MAX_PLANETS):
        if not alive[i] or is_comet[i]:
            continue
        pi = p.planet_position(i, t=t, by_id=False)
        if pi is not None:
            manual.append(pi)

    np.testing.assert_allclose(
        pos["planets"], np.array(manual, dtype=np.float32), atol=1e-5,
    )


# ---------------------------------------------------------------------------
# arrival_ledger — empty hypothetical
# ---------------------------------------------------------------------------


def test_arrival_ledger_no_hypothetical_matches_world_model():
    """Without any hypothetical launches, Predictor.arrival_ledger
    must produce the same ledger as a direct call to
    lib.world_model.build_arrival_ledger with the same planets and
    fleets."""
    from lib.world_model import build_arrival_ledger

    # Use a state with some in-flight fleets — step past a few turns
    # of random play.
    state = _build_state(seed=42)
    rng = random.Random(7919)
    for _ in range(10):
        actions = [[] for _ in range(2)]
        # Random launches from owned planets.
        from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet as _F
        # Just no-op — keep it simple; world_model still has comets
        # to track once spawn fires.
        pids, angles, ships = actions_to_jax(actions, 2)
        state = step(state, pids, angles, ships)

    p = Predictor(state)
    omega = p.omega
    ledger_predictor = p.arrival_ledger(horizon=80)

    planets = p._planets_list()
    fleets = p._fleets_list()
    ledger_direct = build_arrival_ledger(fleets, planets, horizon=80, omega=omega)

    assert ledger_predictor == ledger_direct


# ---------------------------------------------------------------------------
# arrival_ledger — with hypothetical launches
# ---------------------------------------------------------------------------


def test_arrival_ledger_with_hypothetical_launch_turn0():
    """Injecting a hypothetical launch with launch_turn=0 adds an
    arrival entry on the target planet."""
    state = _build_state(seed=42)
    p = Predictor(state)

    # Find a seat-0 planet with ships, and any other planet to aim at.
    planets = p._planets_list()
    owned = next((pl for pl in planets if pl.owner == 0 and pl.ships > 5), None)
    target = next((pl for pl in planets if pl.id != (owned.id if owned else -1)), None)
    if owned is None or target is None:
        pytest.skip("setup planets not available")

    # Aim direction from src to target.
    angle = math.atan2(target.y - owned.y, target.x - owned.x)
    spec = ActionSpec(
        from_planet_id=owned.id,
        dir_angle=angle,
        ships=min(owned.ships, 10),
        launch_turn=0,
        agent_id=0,
    )

    ledger_no = p.arrival_ledger(horizon=80)
    ledger_yes = p.arrival_ledger(horizon=80, hypothetical=[spec])

    # The target's arrivals must INCLUDE a seat-0 entry with our ships.
    target_arrivals_yes = ledger_yes.get(target.id, [])
    target_arrivals_no = ledger_no.get(target.id, [])
    new_entries = [a for a in target_arrivals_yes if a not in target_arrivals_no]
    assert len(new_entries) >= 1, (
        f"hypothetical launch should add an arrival on target; "
        f"no={target_arrivals_no} yes={target_arrivals_yes}"
    )
    # The new entry should be seat-0, with `spec.ships`.
    assert any(e[1] == 0 and e[2] == spec.ships for e in new_entries)


def test_arrival_ledger_hypothetical_launch_turn_nonzero_raises():
    """launch_turn>0 is documented as not-yet-supported."""
    state = _build_state()
    p = Predictor(state)
    spec = ActionSpec(
        from_planet_id=0, dir_angle=0.0, ships=1, launch_turn=5, agent_id=0,
    )
    with pytest.raises(NotImplementedError, match="launch_turn>0"):
        p.arrival_ledger(horizon=30, hypothetical=[spec])


def test_arrival_ledger_hypothetical_unknown_source_is_skipped():
    """A hypothetical launch from an unknown planet id is silently
    skipped (no synthetic fleet added) — same behavior as the env
    rejecting an invalid action."""
    state = _build_state()
    p = Predictor(state)
    spec = ActionSpec(
        from_planet_id=9999, dir_angle=0.0, ships=1, launch_turn=0, agent_id=0,
    )
    ledger_no = p.arrival_ledger(horizon=30)
    ledger_yes = p.arrival_ledger(horizon=30, hypothetical=[spec])
    assert ledger_no == ledger_yes


# ---------------------------------------------------------------------------
# ships_at_planet
# ---------------------------------------------------------------------------


def test_ships_at_planet_t0_returns_current():
    """At t=0, ships_at_planet returns the planet's current (owner,
    ships)."""
    state = _build_state(seed=42)
    p = Predictor(state)
    planets = p._planets_list()
    for planet in planets[:3]:
        owner, ships = p.ships_at_planet(planet.id, t=0)
        assert owner == planet.owner
        assert ships == planet.ships


def test_ships_at_planet_accounts_for_production():
    """An owned planet's ship count grows by `production` per turn."""
    state = _build_state(seed=42)
    p = Predictor(state)
    planets = p._planets_list()
    owned = next((pl for pl in planets if pl.owner == 0 and pl.production > 0), None)
    if owned is None:
        pytest.skip("no owned seat-0 producer at seed=42 init")
    owner_at_0, ships_at_0 = p.ships_at_planet(owned.id, t=0)
    owner_at_10, ships_at_10 = p.ships_at_planet(owned.id, t=10)
    assert owner_at_0 == owner_at_10  # ownership stable if no arrivals
    # Ships grow by production * 10 (no incoming arrivals expected).
    assert math.isclose(ships_at_10, ships_at_0 + owned.production * 10, abs_tol=0.5)


def test_ships_at_planet_negative_t_raises():
    state = _build_state()
    p = Predictor(state)
    planets = p._planets_list()
    with pytest.raises(ValueError, match="t must be"):
        p.ships_at_planet(planets[0].id, t=-1)


def test_ships_at_planet_unknown_returns_neutral_zero():
    state = _build_state()
    p = Predictor(state)
    owner, ships = p.ships_at_planet(9999, t=5)
    assert owner == -1
    assert ships == 0.0


# ---------------------------------------------------------------------------
# Arrival namedtuple-ish wrapper
# ---------------------------------------------------------------------------


def test_arrival_dataclass_round_trip():
    """`Arrival.from_tuple` and field access round-trip."""
    t = (15, 0, 42)
    a = Arrival.from_tuple(t)
    assert a.eta == 15
    assert a.owner == 0
    assert a.ships == 42


# ---------------------------------------------------------------------------
# Immutability: predictor must not mutate input state
# ---------------------------------------------------------------------------


def test_predictor_does_not_mutate_state():
    """Construct Predictor, run a bunch of queries, assert the input
    state's array contents haven't changed (Pytree-immutable
    contract)."""
    state = _build_state(seed=42)
    snapshot_x = np.asarray(state.planets_x).copy()
    snapshot_owner = np.asarray(state.planets_owner).copy()

    p = Predictor(state)
    # Exercise the API.
    p.positions_at(t=5)
    p.positions_at(t=20)
    planets = p._planets_list()
    if planets:
        p.ships_at_planet(planets[0].id, t=10)
    # Hypothetical injection — does NOT mutate state.
    if planets and any(pl.owner == 0 and pl.ships > 0 for pl in planets):
        src = next(pl for pl in planets if pl.owner == 0 and pl.ships > 0)
        p.arrival_ledger(horizon=30, hypothetical=[
            ActionSpec(from_planet_id=src.id, dir_angle=0.5, ships=1,
                       launch_turn=0, agent_id=0)
        ])

    np.testing.assert_array_equal(np.asarray(state.planets_x), snapshot_x)
    np.testing.assert_array_equal(np.asarray(state.planets_owner), snapshot_owner)
